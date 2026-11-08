# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DTV-flip: reverse negative-alignment full sample gradients."""

from __future__ import annotations

import json
import os
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import trainer as rl_trainer


def configuration_from_environment(environ):
  """Read the opt-in configuration without changing the training CLI."""
  enabled = lambda key: environ.get(key, "").strip().lower() in (
      "1", "true", "yes", "on"
  )
  for key in (
      "TUNIX_DBC_SELF_INF_POLICY",
      "TUNIX_DBC_SELF_INF_LOO_POLICY",
      "TUNIX_DBC_SELF_INF_LOO_POLICY_ONLY",
  ):
    if enabled(key):
      raise ValueError(f"dtv-flip uses full gradients; unset {key}.")
  scope = environ.get("TUNIX_DBC_SELF_INF_SCOPE", "batch").strip().lower()
  generations = int(environ.get("TUNIX_GRPO_NUM_GENERATIONS", "0"))
  if scope not in ("batch", "group"):
    raise ValueError(f"Invalid dtv-flip scope: {scope!r}")
  if scope == "group" and generations <= 0:
    raise ValueError("dtv-flip group scope requires num_generations > 0.")
  return dict(
      scope=scope,
      num_generations=generations,
      loo=enabled("TUNIX_DBC_SELF_INF_LOO"),
      decisions_path=environ.get("TUNIX_DTV_FLIP_DECISIONS_PATH"),
  )


def alignment_scores(grads, *, scope="batch", num_generations=0, loo=False):
  """Score full gradients against their batch/group mean (optionally LOO).

  Ordinary DTV uses the same mean-and-dot operations as SelfInfTrainer.
  LOO uses float32 cross sums / (population - 1), as SelfInfLooTrainer.
  Invalid group layouts fail explicitly instead of silently using batch scope.
  """
  leaves = jax.tree_util.tree_leaves(grads)
  if not leaves:
    raise ValueError("No gradients found for dtv-flip.")
  batch_size = int(leaves[0].shape[0])
  if scope not in ("batch", "group"):
    raise ValueError(f"Invalid dtv-flip scope: {scope!r}")
  size = batch_size if scope == "batch" else int(num_generations or 0)
  if batch_size == 0 or size <= 0 or batch_size % size:
    raise ValueError("dtv-flip requires nonempty, complete scoring groups.")
  if loo and size <= 1:
    raise ValueError("DTV-LOO requires at least two samples per scoring group.")
  scores = []
  for leaf in leaves:
    if leaf.shape[0] != batch_size:
      raise ValueError("Inconsistent sample axes in gradient tree.")
    g = leaf.astype(jnp.float32) if loo else leaf
    g = g.reshape((batch_size // size, size) + g.shape[1:])
    axes = tuple(range(2, g.ndim))
    if loo:
      total = jnp.sum(g, axis=1, keepdims=True)
      cross = jnp.sum(g * total, axis=axes) - jnp.sum(g * g, axis=axes)
      score = cross
    else:
      score = jnp.sum(g * jnp.mean(g, axis=1, keepdims=True), axis=axes)
    scores.append(score.reshape(batch_size))
  result = sum(scores)
  return result / float(size - 1) if loo else result


def signed_mean(grads, weights):
  """Average signed contributions over N, never over sum(weights)."""
  def reduce(leaf):
    shape = (-1,) + (1,) * (leaf.ndim - 1)
    return jnp.mean(leaf * weights.reshape(shape), axis=0)
  return jax.tree_util.tree_map(reduce, grads)


def _reject_nonfinite(scores):
  """Abort a bad step under eager or JIT execution; never label NaN positive."""
  def fail(_):
    def raise_error():
      raise FloatingPointError("dtv-flip encountered a nonfinite alignment score")
    # Unordered callbacks also support multi-device SPMD compilation.
    jax.debug.callback(raise_error)
  jax.lax.cond(jnp.all(jnp.isfinite(scores)), lambda _: None, fail, None)


class DTVFlipTrainer(rl_trainer.Trainer):
  """Use total-loss alignment to flip the same total gradients DTV masks."""

  def __init__(self, *args, scope="batch", num_generations=None,
               loo=False, decisions_path=None, **kwargs):
    if scope not in ("batch", "group"):
      raise ValueError(f"Invalid dtv-flip scope: {scope!r}")
    if scope == "group" and int(num_generations or 0) <= int(loo):
      raise ValueError("Insufficient num_generations for dtv-flip group scope.")
    super().__init__(*args, **kwargs)
    self.scope = scope
    self.num_generations = num_generations
    self.loo = loo
    self.decisions_path = decisions_path
    if decisions_path:
      os.makedirs(os.path.dirname(os.path.abspath(decisions_path)), exist_ok=True)

  def _train_step(self, model, optimizer, inputs):
    inputs = self.gen_model_input_fn(inputs)

    def sample_loss(model, inputs):
      if isinstance(inputs, dict):
        return self.loss_fn(model, **inputs)
      return self.loss_fn(model, inputs)

    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    grad_fn = nnx.value_and_grad(
        sample_loss, argnums=nnx.DiffState(0, wrt), has_aux=self._has_aux
    )
    values, grads = jax.vmap(
        grad_fn, in_axes=(None, jax.tree_util.tree_map(lambda _: 0, inputs))
    )(model, inputs)
    if self._has_aux:
      losses, aux = values
    else:
      losses, aux = values, None
    scores = alignment_scores(
        grads, scope=self.scope, num_generations=self.num_generations, loo=self.loo
    )
    _reject_nonfinite(scores)
    weights = jax.lax.stop_gradient(jnp.where(scores < 0, -1.0, 1.0))
    optimizer.update(model, signed_mean(grads, weights))
    final_loss = jnp.mean(losses * weights)
    if self._has_aux:
      # Raw reward/KL diagnostics remain ordinary averages, not signed metrics.
      final_aux = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), aux)
      if isinstance(final_aux, dict):
        final_aux.update({
            "skipped_samples": jnp.asarray(0.0),
            "self_inf_dot_mean": jnp.mean(scores),
            "self_inf_dot_std": jnp.std(scores),
            "self_inf_kept_fraction": jnp.asarray(1.0),
            "dtv_flip_count": jnp.sum(scores < 0),
            "dtv_flip_fraction": jnp.mean(scores < 0),
            "dtv_flip_raw_loss": jnp.mean(losses),
        })
        if self.decisions_path:
          final_aux["dtv_flip_scores"] = scores
          final_aux["dtv_flip_weights"] = weights
      return final_loss, final_aux
    return final_loss, None

  def _post_process_train_step(self, aux: Any) -> None:
    if isinstance(aux, dict):
      if self.decisions_path and "dtv_flip_scores" in aux:
        record = {
            "method": "dtv-flip", "score": "dtv-loo" if self.loo else "dtv",
            "scope": self.scope, "score_objective": "total",
            "flip_application": "full", "num_generations": self.num_generations,
            "train_step": int(self._train_steps) + 1,
            "scores": np.asarray(aux.pop("dtv_flip_scores")).tolist(),
            "weights": np.asarray(aux.pop("dtv_flip_weights")).tolist(),
        }
        with open(self.decisions_path, "a", encoding="utf-8") as output:
          output.write(json.dumps(record, allow_nan=False) + "\n")
      for key in ("dtv_flip_count", "dtv_flip_fraction", "dtv_flip_raw_loss"):
        self.rl_metrics_to_log.setdefault(key, np.mean)
    super()._post_process_train_step(aux)
