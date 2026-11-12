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

"""Flip full completion gradients selected by alignment or random sampling."""

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
      batch_flip_fraction=environ.get("TUNIX_DTV_FLIP_BATCH_FRACTION") or None,
      frequency=flip_frequency(environ.get("TUNIX_DTV_FLIP_FREQUENCY", "1")),
      random_flip_seed=(int(environ.get("TUNIX_EXPERIMENT_SEED") or "0")
                        if enabled("TUNIX_DTV_FLIP_RANDOM") else None),
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
  """Parse an exact ratio for the quota, including 0 and 1."""
  try:
    ratio = Fraction(str(value))
  except (ValueError, ZeroDivisionError) as exc:
    raise ValueError("Flip fraction must be finite and in [0, 1].") from exc
  if not 0 <= ratio <= 1 or ratio.denominator > 1_000_000:
    raise ValueError("Flip fraction must be in [0, 1], denominator <= 1000000.")
  return ratio.numerator, ratio.denominator


def flip_frequency(value):
  text = str(value)
  if (len(text) > 10 or not text.isascii() or not text.isdecimal()
      or text.startswith("0") or int(text) > np.iinfo(np.int32).max):
    raise ValueError("TUNIX_DTV_FLIP_FREQUENCY must be an integer in [1, 2147483647].")
  return int(text)


def nearest_even_quota(batch_size, numerator, denominator):
  """Round to the nearest even count, with ties upward and a batch size cap."""
  pairs = (batch_size * numerator + denominator) // (2 * denominator)
  return 2 * min(pairs, batch_size // 2)


def _lowest_negative_weights(scores, quota):
  order = jnp.argsort(scores, stable=True)
  ranks = jnp.empty_like(order).at[order].set(jnp.arange(scores.shape[0]))
  return jnp.where((scores < 0) & (ranks < quota), -1.0, 1.0)


def select_flips(scores, remainder, numerator, denominator):
  """Issue this batch's quota; unused negative-candidate slots expire.

  Only fractional rounding remainder carries over. Stable sorting resolves
  equal scores by original completion index. Quota scope is the entire batch,
  independent of batch/group alignment score scope.
  """
  whole, fractional = divmod(int(scores.shape[0]) * numerator, denominator)
  tokens = remainder + fractional
  quota, next_remainder = whole + tokens // denominator, tokens % denominator
  return _lowest_negative_weights(scores, quota), quota, next_remainder


class FlipBudget(nnx.Module):
  """Dynamic JIT state attached to the optimizer, saved in custom metadata."""

  def __init__(self, seen=0, flipped=0, remainder=0, total_seen=0):
    self.seen = nnx.Variable(jnp.asarray(seen, dtype=jnp.int32))
    self.flipped = nnx.Variable(jnp.asarray(flipped, dtype=jnp.int32))
    self.remainder = nnx.Variable(jnp.asarray(remainder, dtype=jnp.int32))
    self.total_seen = nnx.Variable(jnp.asarray(total_seen, dtype=jnp.int32))


class DTVFlipTrainer(rl_trainer.Trainer):
  """Apply signed GRPO updates at a fixed actor-update frequency."""

  def __init__(self, *args, scope="batch", num_generations=None,
               loo=False, decisions_path=None, max_flip_fraction="0.1",
               batch_flip_fraction=None, frequency=1, random_flip_seed=None,
               **kwargs):
    if scope not in ("batch", "group"):
      raise ValueError(f"Invalid dtv-flip scope: {scope!r}")
    if scope == "group" and int(num_generations or 0) <= int(loo):
      raise ValueError("Insufficient num_generations for dtv-flip group scope.")
    if random_flip_seed is not None:
      if scope != "batch" or loo:
        raise ValueError("Random flip requires batch scope and no LOO scoring.")
      if type(random_flip_seed) is not int or not 0 <= random_flip_seed < 2**32:
        raise ValueError("Random flip requires a seed in [0, 4294967295].")
      if batch_flip_fraction is None:
        batch_flip_fraction = "0.1"
    self.quota_policy = (
        "cumulative_floor_no_catchup" if batch_flip_fraction is None else "batch_nearest_even"
    )
    fraction = max_flip_fraction if batch_flip_fraction is None else batch_flip_fraction
    self.flip_numerator, self.flip_denominator = ratio_terms(fraction)
    self.frequency = flip_frequency(frequency)
    self.random_flip_seed = random_flip_seed
    self._random_flip_loss = None
    super().__init__(*args, **kwargs)
    self._grad_acc_steps = self.config.get_with_default("gradient_accumulation_steps", 1)
    self.scope = scope
    self.num_generations = num_generations
    self.loo = loo
    self.decisions_path = decisions_path
    self.initialize_budget_state(
        self.optimizer, self._restored_custom_metadata, self._train_steps
    )
    if decisions_path:
      os.makedirs(os.path.dirname(os.path.abspath(decisions_path)), exist_ok=True)

  def with_random_flip_loss_fn(self, loss_fn):
    """Attach the GRPO loss that accepts completion signs."""
    self.clear_jit_cache()
    self._random_flip_loss = loss_fn

  def _budget_configuration(self):
    configuration = dict(
        version=2, numerator=self.flip_numerator, denominator=self.flip_denominator,
        scope=self.scope, loo=self.loo, num_generations=self.num_generations,
        quota_policy=self.quota_policy, frequency=self.frequency,
        grad_acc_steps=self._grad_acc_steps,
    )
    if self.random_flip_seed is not None:
      configuration.update(version=3, random_flip_seed=self.random_flip_seed)
    return configuration

  def initialize_budget_state(self, optimizer, metadata, restored_step):
    """Restore quota counters and continue the schedule from the saved actor step."""
    state = (metadata or {}).get("dtv_flip_budget")
    if state is None:
      if restored_step:
        raise ValueError("Checkpoint has no dtv-flip quota state; use a new checkpoint root.")
      state = dict(seen=0, flipped=0, remainder=0, total_seen=0)
    else:
      if state.get("version") == 1:
        state = dict(state, version=2, quota_policy="cumulative_floor_no_catchup",
                     frequency=1, grad_acc_steps=self._grad_acc_steps,
                     total_seen=state.get("seen"))
      expected = self._budget_configuration()
      if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint dtv-flip configuration differs.")
      seen, flipped, remainder, total_seen = (
          state.get(key) for key in ("seen", "flipped", "remainder", "total_seen")
      )
      if (any(type(x) is not int for x in (seen, flipped, remainder, total_seen))
          or not 0 <= flipped <= seen <= total_seen <= np.iinfo(np.int32).max
          or (self.frequency == 1 and seen != total_seen)):
        raise ValueError("Invalid checkpoint dtv-flip budget counters.")
      if self.quota_policy == "cumulative_floor_no_catchup":
        cap, expected_remainder = divmod(seen * self.flip_numerator, self.flip_denominator)
      else:
        cap, expected_remainder = seen, 0
      if flipped > cap or remainder != expected_remainder:
        raise ValueError("Invalid checkpoint dtv-flip budget counters.")
    self._initial_iter_step = restored_step * self._grad_acc_steps
    optimizer.dtv_flip_budget = FlipBudget(
        state["seen"], state["flipped"], state["remainder"], state["total_seen"]
    )

  def custom_checkpoint_metadata(self):
    metadata = dict(self.custom_checkpoint_metadata_fn())
    budget = self.optimizer.dtv_flip_budget
    metadata["dtv_flip_budget"] = dict(
        self._budget_configuration(),
        seen=int(np.asarray(budget.seen[...])),
        flipped=int(np.asarray(budget.flipped[...])),
        remainder=int(np.asarray(budget.remainder[...])),
        total_seen=int(np.asarray(budget.total_seen[...])),
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
    batch_size = jax.tree.leaves(inputs)[0].shape[0]

    def sample_loss(model, inputs):
      if isinstance(inputs, dict):
        return self.loss_fn(model, **inputs)
      return self.loss_fn(model, inputs)

    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    grad_fn = nnx.value_and_grad(
        sample_loss, argnums=nnx.DiffState(0, wrt), has_aux=self._has_aux
    )
    budget = optimizer.dtv_flip_budget
    remainder = budget.remainder[...]
    iter_step = optimizer.step[...] + self._initial_iter_step

    def flip_step(model, inputs):
      values, grads = jax.vmap(grad_fn, in_axes=(None, 0))(model, inputs)
      losses, aux = values if self._has_aux else (values, None)
      scores = alignment_scores(
          grads, scope=self.scope, num_generations=self.num_generations, loo=self.loo
      )
      _reject_nonfinite(scores)
      if self.quota_policy == "batch_nearest_even":
        quota = jnp.asarray(nearest_even_quota(
            batch_size, self.flip_numerator, self.flip_denominator
        ), dtype=jnp.int32)
        weights = _lowest_negative_weights(scores, quota)
        next_remainder = remainder
      else:
        weights, quota, next_remainder = select_flips(
            scores, remainder, self.flip_numerator, self.flip_denominator
        )
      weights = jax.lax.stop_gradient(weights)
      raw_loss = jnp.mean(losses)
      aux = jax.tree.map(lambda x: jnp.mean(x, axis=0), aux)
      if self.frequency != 1:
        scores = scores.astype(jnp.float32)
      return (signed_mean(grads, weights), jnp.mean(losses * weights), aux,
              raw_loss, scores, weights, quota, next_remainder)

    def ordinary_step(model, inputs):
      values, grads = grad_fn(model, inputs)
      loss, aux = values if self._has_aux else (values, None)
      scores = (jnp.zeros(batch_size, dtype=jnp.float32)
                if self.random_flip_seed is None else None)
      weights = jnp.broadcast_to(1.0, (batch_size,))
      return grads, loss, aux, loss, scores, weights, jnp.asarray(0, jnp.int32), remainder

    def random_step(model, inputs):
      quota = nearest_even_quota(batch_size, self.flip_numerator, self.flip_denominator)
      if quota == 0:
        return ordinary_step(model, inputs)

      # Use a separate PRNG stream for flip selection, indexed by micro-step.
      key = jax.random.fold_in(jax.random.key(self.random_flip_seed), 0x464C4950)
      key = jax.random.fold_in(key, iter_step)
      indices = jax.random.permutation(key, batch_size)[:quota]
      weights = jnp.ones(batch_size).at[indices].set(-1.0)

      def signed_loss(model):
        return self._random_flip_loss(model, **inputs, completion_weights=weights)

      (loss, aux), grads = nnx.value_and_grad(
          signed_loss, argnums=nnx.DiffState(0, wrt), has_aux=True
      )(model)
      raw_loss = aux.pop("dtv_flip_raw_loss")
      return grads, loss, aux, raw_loss, None, weights, jnp.asarray(quota, jnp.int32), remainder

    if self.random_flip_seed is not None and self._random_flip_loss is None:
      raise ValueError("Random flip requires the GRPO loss with completion signs.")
    active_step = flip_step if self.random_flip_seed is None else random_step
    if self.frequency == 1:
      enabled = jnp.asarray(True)
      result = active_step(model, inputs)
    else:
      # All microbatches in one accumulated optimizer update use the same phase.
      enabled = (iter_step // self._grad_acc_steps) % self.frequency == 0
      result = nnx.cond(enabled, active_step, ordinary_step, model, inputs)
    grads, loss, aux, raw_loss, scores, weights, quota, remainder = result
    optimizer.update(model, grads)

    flip_count = jnp.sum(weights < 0).astype(jnp.int32)
    budget.total_seen[...] += batch_size
    budget.seen[...] += jnp.where(enabled, batch_size, 0)
    budget.flipped[...] += flip_count
    budget.remainder[...] = remainder
    if isinstance(aux, dict):
      aux.update({
          "skipped_samples": jnp.asarray(0.0),
          "self_inf_kept_fraction": jnp.asarray(1.0),
          "dtv_flip_enabled": enabled.astype(jnp.int32),
          "dtv_flip_count": flip_count,
          "dtv_flip_fraction": jnp.mean(weights < 0),
          "dtv_flip_quota": quota,
          "dtv_flip_seen": budget.seen[...],
          "dtv_flip_total_seen": budget.total_seen[...],
          "dtv_flip_total": budget.flipped[...],
          "dtv_flip_active_fraction": budget.flipped[...] / jnp.maximum(budget.seen[...], 1),
          "dtv_flip_cumulative_fraction": budget.flipped[...] / budget.total_seen[...],
          "dtv_flip_raw_loss": raw_loss,
      })
      if scores is not None:
        aux.update({
            "self_inf_dot_mean": jnp.mean(scores),
            "self_inf_dot_std": jnp.std(scores),
            "dtv_flip_negative_count": jnp.sum(scores < 0),
            "dtv_flip_negative_fraction": jnp.mean(scores < 0),
        })
      if self.decisions_path:
        aux["dtv_flip_scores"] = scores
        aux["dtv_flip_weights"] = weights
    return loss, aux

  def _post_process_train_step(self, aux: Any) -> None:
    if isinstance(aux, dict):
      enabled = self.frequency == 1 or bool(aux["dtv_flip_enabled"])
      if self.decisions_path and "dtv_flip_weights" in aux:
        scores = aux.pop("dtv_flip_scores")
        weights = aux.pop("dtv_flip_weights")
        record = {
            "method": "dtv-flip" if scores is not None else "random-flip",
            "score": ("dtv-loo" if self.loo else "dtv") if scores is not None else "random",
            "scope": self.scope, "score_objective": "total" if scores is not None else None,
            "flip_application": "full", "num_generations": self.num_generations,
            "train_step": int(self._train_steps) + 1,
            "quota_scope": "batch", "quota_policy": self.quota_policy,
            "frequency": self.frequency, "flip_enabled": enabled,
            "batch_completion_count": int(weights.shape[0]),
            "configured_fraction": self.flip_numerator / self.flip_denominator,
            **{key: np.asarray(aux[key]).item() for key in (
                "dtv_flip_count", "dtv_flip_fraction", "dtv_flip_quota",
                "dtv_flip_seen", "dtv_flip_total_seen", "dtv_flip_total",
                "dtv_flip_active_fraction", "dtv_flip_cumulative_fraction",
            )},
            "dtv_flip_negative_count": (
                int(aux["dtv_flip_negative_count"]) if enabled and scores is not None else None
            ),
            "scores": np.asarray(scores).tolist() if enabled and scores is not None else None,
            "weights": np.asarray(weights).tolist() if enabled else None,
        }
        if self.quota_policy == "cumulative_floor_no_catchup":
          record["max_flip_fraction"] = record["configured_fraction"]
        if self.random_flip_seed is not None:
          record["random_flip_seed"] = self.random_flip_seed
          record["flipped_indices"] = np.flatnonzero(np.asarray(weights) < 0).tolist()
        with open(self.decisions_path, "a", encoding="utf-8") as output:
          output.write(json.dumps(record, allow_nan=False) + "\n")
      if not enabled and self.random_flip_seed is None:
        for key in ("self_inf_dot_mean", "self_inf_dot_std", "dtv_flip_negative_count",
                    "dtv_flip_negative_fraction"):
          aux.pop(key)
      for key in (
          "dtv_flip_enabled", "dtv_flip_count", "dtv_flip_fraction", "dtv_flip_raw_loss",
          "dtv_flip_negative_count", "dtv_flip_negative_fraction", "dtv_flip_quota",
          "dtv_flip_seen", "dtv_flip_total_seen", "dtv_flip_total",
          "dtv_flip_active_fraction", "dtv_flip_cumulative_fraction",
      ):
        self.rl_metrics_to_log.setdefault(key, np.mean)
    super()._post_process_train_step(aux)
