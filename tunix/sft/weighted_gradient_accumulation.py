"""Gradient accumulation with one effective-example denominator."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class WeightedMultiStepsState(NamedTuple):
  mini_step: jax.Array
  gradient_step: jax.Array
  inner_opt_state: Any
  acc_gradient_sum: Any
  acc_effective_count: jax.Array


def _promote_optimizer_float_value(value):
  """Promotes low-precision floating leaves used by optimizer state."""
  if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact):
    if value.dtype in (jnp.bfloat16, jnp.float16):
      return value.astype(jnp.float32)
  return value


def weighted_multisteps(
    inner: optax.GradientTransformation,
    every_k_schedule: int,
) -> optax.GradientTransformationExtraArgs:
  """Accumulates gradient numerators and divides once per optimizer update."""
  if every_k_schedule < 1:
    raise ValueError("every_k_schedule must be positive.")

  def init(params):
    optimizer_init_params = jax.tree.map(
        _promote_optimizer_float_value, params
    )
    inner_opt_state = jax.tree.map(
        _promote_optimizer_float_value, inner.init(optimizer_init_params)
    )
    return WeightedMultiStepsState(
        mini_step=jnp.zeros([], dtype=jnp.int32),
        gradient_step=jnp.zeros([], dtype=jnp.int32),
        inner_opt_state=inner_opt_state,
        acc_gradient_sum=jax.tree.map(jnp.zeros_like, params),
        acc_effective_count=jnp.zeros([], dtype=jnp.float32),
    )

  def update(updates, state, params=None, *, effective_count=None, **extra_args):
    del extra_args
    if effective_count is None:
      raise ValueError("weighted_multisteps requires effective_count.")
    count = jnp.asarray(effective_count, dtype=jnp.float32)
    gradient_sum = jax.tree.map(lambda x: x * count, updates)
    accumulated = jax.tree.map(
        lambda x, y: x + y, state.acc_gradient_sum, gradient_sum
    )
    accumulated_count = state.acc_effective_count + count
    should_emit = state.mini_step == every_k_schedule - 1

    def emit(_):
      has_signal = accumulated_count > 0
      denominator = jnp.maximum(accumulated_count, 1.0)
      mean_gradient = jax.tree.map(lambda x: x / denominator, accumulated)

      def apply_inner(_):
        inner_updates, inner_state = inner.update(
            mean_gradient, state.inner_opt_state, params
        )
        return inner_updates, inner_state, state.gradient_step + 1

      def skip_inner(_):
        return (
            jax.tree.map(jnp.zeros_like, mean_gradient),
            state.inner_opt_state,
            state.gradient_step,
        )

      output_updates, inner_state, gradient_step = jax.lax.cond(
          has_signal, apply_inner, skip_inner, operand=None
      )
      next_state = WeightedMultiStepsState(
          mini_step=jnp.zeros([], dtype=jnp.int32),
          gradient_step=gradient_step,
          inner_opt_state=inner_state,
          acc_gradient_sum=jax.tree.map(jnp.zeros_like, accumulated),
          acc_effective_count=jnp.zeros([], dtype=jnp.float32),
      )
      return output_updates, next_state

    def accumulate(_):
      return (
          jax.tree.map(jnp.zeros_like, updates),
          WeightedMultiStepsState(
              mini_step=state.mini_step + 1,
              gradient_step=state.gradient_step,
              inner_opt_state=state.inner_opt_state,
              acc_gradient_sum=accumulated,
              acc_effective_count=accumulated_count,
          ),
      )

    return jax.lax.cond(should_emit, emit, accumulate, operand=None)

  return optax.GradientTransformationExtraArgs(init, update)
