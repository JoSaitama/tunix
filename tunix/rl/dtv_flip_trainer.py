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
from fractions import Fraction
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
      max_flip_fraction=environ.get("TUNIX_DTV_FLIP_MAX_FRACTION", "0.1"),
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


def ratio_terms(value):
  """Use exact rational arithmetic for the cumulative quota, including 0/1."""
  try:
    ratio = Fraction(str(value))
  except (ValueError, ZeroDivisionError) as exc:
    raise ValueError("max_flip_fraction must be finite and in [0, 1].") from exc
  if not 0 <= ratio <= 1 or ratio.denominator > 1_000_000:
    raise ValueError("max_flip_fraction must be in [0, 1], denominator <= 1000000.")
  return ratio.numerator, ratio.denominator


def select_flips(scores, remainder, numerator, denominator):
  """Issue this batch's quota; unused negative-candidate slots expire.

  Only fractional rounding remainder carries over. Stable sorting resolves
  equal scores by original completion index. Quota scope is the entire batch,
  independent of batch/group alignment score scope.
  """
  tokens = remainder + int(scores.shape[0]) * numerator
  quota, next_remainder = tokens // denominator, tokens % denominator
  order = jnp.argsort(scores, stable=True)
  ranks = jnp.empty_like(order).at[order].set(jnp.arange(scores.shape[0]))
  selected = (scores < 0) & (ranks < quota)
  return jnp.where(selected, -1.0, 1.0), quota, next_remainder


class FlipBudget(nnx.Module):
  """Dynamic JIT state attached to the optimizer, saved in custom metadata."""

  def __init__(self, seen=0, flipped=0, remainder=0):
    self.seen = nnx.Variable(jnp.asarray(seen, dtype=jnp.int32))
    self.flipped = nnx.Variable(jnp.asarray(flipped, dtype=jnp.int32))
    self.remainder = nnx.Variable(jnp.asarray(remainder, dtype=jnp.int32))


class DTVFlipTrainer(rl_trainer.Trainer):
  """Use total-loss alignment to flip the same total gradients DTV masks."""

  def __init__(self, *args, scope="batch", num_generations=None,
               loo=False, decisions_path=None, max_flip_fraction="0.1", **kwargs):
    if scope not in ("batch", "group"):
      raise ValueError(f"Invalid dtv-flip scope: {scope!r}")
    if scope == "group" and int(num_generations or 0) <= int(loo):
      raise ValueError("Insufficient num_generations for dtv-flip group scope.")
    self.flip_numerator, self.flip_denominator = ratio_terms(max_flip_fraction)
    super().__init__(*args, **kwargs)
    self.scope = scope
    self.num_generations = num_generations
    self.loo = loo
    self.decisions_path = decisions_path
    self.initialize_budget_state(
        self.optimizer, self._restored_custom_metadata, self._train_steps
    )
    if decisions_path:
      os.makedirs(os.path.dirname(os.path.abspath(decisions_path)), exist_ok=True)

  def initialize_budget_state(self, optimizer, metadata, restored_step):
    """Restore the exact budget; reject legacy or incompatible continuations."""
    state = (metadata or {}).get("dtv_flip_budget")
    if state is None:
      if restored_step:
        raise ValueError("Checkpoint has no dtv-flip quota state; use a new checkpoint root.")
      state = dict(seen=0, flipped=0, remainder=0)
    else:
      expected = dict(version=1, numerator=self.flip_numerator,
                      denominator=self.flip_denominator, scope=self.scope, loo=self.loo,
                      num_generations=self.num_generations)
      if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint dtv-flip ratio or scoring configuration differs.")
      seen, flipped, remainder = (state.get(key) for key in ("seen", "flipped", "remainder"))
      if (any(type(x) is not int for x in (seen, flipped, remainder))
          or not 0 <= seen <= np.iinfo(np.int32).max
          or not 0 <= flipped <= seen * self.flip_numerator // self.flip_denominator
          or remainder != seen * self.flip_numerator % self.flip_denominator):
        raise ValueError("Invalid checkpoint dtv-flip budget counters.")
    optimizer.dtv_flip_budget = FlipBudget(
        state["seen"], state["flipped"], state["remainder"]
    )

  def custom_checkpoint_metadata(self):
    metadata = dict(self.custom_checkpoint_metadata_fn())
    budget = self.optimizer.dtv_flip_budget
    metadata["dtv_flip_budget"] = dict(
        version=1, numerator=self.flip_numerator, denominator=self.flip_denominator,
        scope=self.scope, loo=self.loo, num_generations=self.num_generations,
        seen=int(np.asarray(budget.seen[...])),
        flipped=int(np.asarray(budget.flipped[...])),
        remainder=int(np.asarray(budget.remainder[...])),
    )
    return metadata

  def _save_last_checkpoint(self):
    # The base final-save path omits custom metadata. Keep this fix local to
    # dtv-flip so both interval and final checkpoints carry the quota state.
    if self._train_steps <= 0:
      return
    latest = self.checkpoint_manager.latest_step()
    if latest is None or latest < self._train_steps:
      self.checkpoint_manager.save(
          self._train_steps, self.model, save_only_lora_params=self._lora_enabled,
          force=True, custom_metadata=self.custom_checkpoint_metadata(),
      )

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
    budget = optimizer.dtv_flip_budget
    weights, quota, remainder = select_flips(
        scores, budget.remainder[...], self.flip_numerator, self.flip_denominator
    )
    weights = jax.lax.stop_gradient(weights)
    optimizer.update(model, signed_mean(grads, weights))
    budget.seen[...] += int(scores.shape[0])
    budget.flipped[...] += jnp.sum(weights < 0).astype(jnp.int32)
    budget.remainder[...] = remainder
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
            "dtv_flip_count": jnp.sum(weights < 0),
            "dtv_flip_fraction": jnp.mean(weights < 0),
            "dtv_flip_negative_count": jnp.sum(scores < 0),
            "dtv_flip_negative_fraction": jnp.mean(scores < 0),
            "dtv_flip_quota": quota,
            "dtv_flip_seen": budget.seen[...],
            "dtv_flip_total": budget.flipped[...],
            "dtv_flip_cumulative_fraction": budget.flipped[...] / budget.seen[...],
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
            "quota_scope": "batch", "quota_policy": "cumulative_floor_no_catchup",
            "max_flip_fraction": self.flip_numerator / self.flip_denominator,
            **{key: np.asarray(aux[key]).item() for key in (
                "dtv_flip_count", "dtv_flip_negative_count", "dtv_flip_quota",
                "dtv_flip_seen", "dtv_flip_total", "dtv_flip_cumulative_fraction",
            )},
            "scores": np.asarray(aux.pop("dtv_flip_scores")).tolist(),
            "weights": np.asarray(aux.pop("dtv_flip_weights")).tolist(),
        }
        with open(self.decisions_path, "a", encoding="utf-8") as output:
          output.write(json.dumps(record, allow_nan=False) + "\n")
      for key in (
          "dtv_flip_count", "dtv_flip_fraction", "dtv_flip_raw_loss",
          "dtv_flip_negative_count", "dtv_flip_negative_fraction", "dtv_flip_quota",
          "dtv_flip_seen", "dtv_flip_total", "dtv_flip_cumulative_fraction",
      ):
        self.rl_metrics_to_log.setdefault(key, np.mean)
    super()._post_process_train_step(aux)
