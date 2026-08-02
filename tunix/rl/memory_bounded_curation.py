"""Memory-bounded exact gradients for dynamic trajectory curation."""

from __future__ import annotations

from typing import Any, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


def batch_size(inputs: Any) -> int:
  for leaf in jax.tree_util.tree_leaves(inputs):
    if isinstance(leaf, (jax.Array, np.ndarray)):
      return int(leaf.shape[0])
  raise ValueError("No batched array found in trainer inputs.")


def sample_inputs(inputs: Any, index: jax.Array) -> Any:
  """Selects one example while retaining the loss function's batch axis."""
  return jax.tree_util.tree_map(
      lambda x: jax.lax.dynamic_index_in_dim(x, index, keepdims=True)
      if isinstance(x, (jax.Array, np.ndarray))
      else x,
      inputs,
  )


def tree_add(left: Any, right: Any) -> Any:
  return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def tree_scale(tree: Any, scale: jax.Array) -> Any:
  return jax.tree_util.tree_map(lambda x: x * scale, tree)


def tree_dot(left: Any, right: Any) -> jax.Array:
  products = jax.tree_util.tree_map(
      lambda x, y: jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32)),
      left,
      right,
  )
  return jax.tree_util.tree_reduce(
      lambda x, y: x + y, products, initializer=jnp.asarray(0.0)
  )


_staged_tree_add = jax.jit(tree_add)
_staged_tree_scale = jax.jit(tree_scale)
_staged_tree_dot = jax.jit(tree_dot)


def make_grad_fn(
    loss_fn: Callable[..., Any], *, wrt: type, has_aux: bool
) -> Callable[..., Any]:
  def call_loss(model, inputs):
    return loss_fn(model, **inputs)

  return nnx.value_and_grad(
      call_loss,
      argnums=nnx.DiffState(0, wrt),
      has_aux=has_aux,
  )


def apply_trajectory_mask(inputs: Any, mask: jax.Array) -> Any:
  """Applies a trajectory mask before the configured loss reduction.

  Agentic GRPO represents valid response tokens with
  ``TrainExample.completion_mask``. Zeroing every response token for a rejected
  trajectory makes the original batched GRPO loss exclude that trajectory from
  policy, KL, entropy, and metric reductions without changing any retained
  trajectory data.
  """
  if not isinstance(inputs, dict) or "train_example" not in inputs:
    raise ValueError(
        "Masked aggregate updates require inputs['train_example']."
    )
  train_example = inputs["train_example"]
  completion_mask = getattr(train_example, "completion_mask", None)
  if completion_mask is None:
    raise ValueError(
        "Masked aggregate updates require TrainExample.completion_mask."
    )
  if completion_mask.shape[0] != mask.shape[0]:
    raise ValueError(
        "Trajectory mask and completion mask batch dimensions differ: "
        f"{mask.shape[0]} != {completion_mask.shape[0]}."
    )
  row_mask = mask.astype(completion_mask.dtype).reshape(
      (mask.shape[0],) + (1,) * (completion_mask.ndim - 1)
  )
  masked_example = train_example.replace(
      completion_mask=completion_mask * row_mask
  )
  return {**inputs, "train_example": masked_example}


def make_masked_aggregate_grad_fn(
    loss_fn: Callable[..., Any], *, wrt: type, has_aux: bool
) -> Callable[..., Any]:
  """Builds one gradient call for a masked aggregate batched loss."""

  def call_loss(model, inputs, mask):
    masked_inputs = apply_trajectory_mask(inputs, mask)
    return loss_fn(model, **masked_inputs)

  return nnx.value_and_grad(
      call_loss,
      argnums=nnx.DiffState(0, wrt),
      has_aux=has_aux,
  )


def gradient_sum(
    grad_fn: Callable[..., Any],
    model: nnx.Module,
    inputs: Any,
    start: int | jax.Array,
    count: int,
) -> Any:
  """Returns an exact sum without stacking per-example gradient trees."""
  _, first = grad_fn(model, sample_inputs(inputs, start))
  accumulated = first
  # Keep NNX graph extraction at the train step's trace level. A lax loop would
  # close over ``model`` from a different trace level, which NNX rejects.
  for offset in range(1, count):
    _, gradient = grad_fn(model, sample_inputs(inputs, start + offset))
    accumulated = tree_add(accumulated, gradient)
  return accumulated


def dtv_statistics(
    grad_fn: Callable[..., Any],
    model: nnx.Module,
    inputs: Any,
    *,
    scope: str,
    group_size: int | None,
) -> dict[str, jax.Array]:
  """Computes ordinary and LOO DTV statistics with exact dot products.

  The score gradients are evaluated twice: once to form each reference sum and
  once to take per-example dot products. This trades compute for bounded HBM.
  """
  size = batch_size(inputs)
  if scope == "group":
    if group_size is None or group_size <= 1 or size % group_size:
      raise ValueError(
          "Group DTV requires a complete group and num_generations > 1; "
          f"received batch_size={size}, num_generations={group_size}."
      )
    reference_size = group_size
  else:
    if size <= 1:
      raise ValueError("Batch DTV requires at least two samples.")
    reference_size = size

  raw_self = jnp.zeros((size,), dtype=jnp.float32)
  raw_cross = jnp.zeros((size,), dtype=jnp.float32)

  def process_reference(start, count, self_values, cross_values):
    total = gradient_sum(grad_fn, model, inputs, start, count)
    for offset in range(count):
      _, gradient = grad_fn(model, sample_inputs(inputs, start + offset))
      self_term = tree_dot(gradient, gradient)
      total_term = tree_dot(gradient, total)
      index = start + offset
      self_values = self_values.at[index].set(self_term)
      cross_values = cross_values.at[index].set(total_term - self_term)
    return self_values, cross_values

  if scope == "group":
    num_groups = size // reference_size
    for group_index in range(num_groups):
      raw_self, raw_cross = process_reference(
          group_index * reference_size,
          reference_size,
          raw_self,
          raw_cross,
      )
  else:
    raw_self, raw_cross = process_reference(
        0, reference_size, raw_self, raw_cross
    )

  denominator = float(reference_size)
  return {
      "raw_self": raw_self,
      "raw_cross_sum": raw_cross,
      "standard_self_term": raw_self / denominator,
      "standard_cross_term": raw_cross / denominator,
      "standard_score": (raw_self + raw_cross) / denominator,
      "loo_score": raw_cross / float(reference_size - 1),
  }


def masked_value_and_grad(
    grad_fn: Callable[..., Any],
    model: nnx.Module,
    inputs: Any,
    mask: jax.Array,
    *,
    has_aux: bool,
) -> tuple[jax.Array, Any | None, Any]:
  """Computes the masked mean gradient without stacking sample gradients."""
  size = batch_size(inputs)
  mask_f = mask.astype(jnp.float32)
  denominator = jnp.clip(jnp.sum(mask_f), 1.0)

  def evaluate(index):
    out, gradient = grad_fn(model, sample_inputs(inputs, index))
    if has_aux:
      loss, aux = out
    else:
      loss, aux = out, None
    return jnp.squeeze(loss), aux, gradient

  first_loss, first_aux, first_gradient = evaluate(0)
  loss_sum = first_loss * mask_f[0]
  gradient_sum_value = tree_scale(first_gradient, mask_f[0])
  aux_sum = tree_scale(first_aux, mask_f[0]) if has_aux else None

  for index in range(1, size):
    current_loss, current_aux, current_gradient = evaluate(index)
    weight = mask_f[index]
    loss_sum = loss_sum + current_loss * weight
    if has_aux:
      aux_sum = tree_add(aux_sum, tree_scale(current_aux, weight))
    gradient_sum_value = tree_add(
        gradient_sum_value, tree_scale(current_gradient, weight)
    )
  mean_gradient = tree_scale(gradient_sum_value, 1.0 / denominator)
  mean_aux = tree_scale(aux_sum, 1.0 / denominator) if has_aux else None
  return loss_sum / denominator, mean_aux, mean_gradient


def staged_dtv_statistics(
    score_step: Callable[[Any], Any],
    inputs: Any,
    *,
    scope: str,
    group_size: int | None,
) -> dict[str, jax.Array]:
  """Computes exact DTV statistics across separately compiled score calls."""
  size = batch_size(inputs)
  if scope == "group":
    if group_size is None or group_size <= 1 or size % group_size:
      raise ValueError(
          "Group DTV requires complete groups and num_generations > 1."
      )
    reference_size = group_size
  else:
    if size <= 1:
      raise ValueError("Batch DTV requires at least two samples.")
    reference_size = size

  raw_self_values = []
  raw_cross_values = []
  starts = range(0, size, reference_size) if scope == "group" else (0,)
  for start in starts:
    total = None
    for offset in range(reference_size):
      gradient = score_step(sample_inputs(inputs, start + offset))
      total = gradient if total is None else _staged_tree_add(total, gradient)
    for offset in range(reference_size):
      gradient = score_step(sample_inputs(inputs, start + offset))
      self_term = _staged_tree_dot(gradient, gradient)
      raw_self_values.append(self_term)
      raw_cross_values.append(_staged_tree_dot(gradient, total) - self_term)

  raw_self = jnp.stack(raw_self_values)
  raw_cross = jnp.stack(raw_cross_values)
  denominator = float(reference_size)
  return {
      "raw_self": raw_self,
      "raw_cross_sum": raw_cross,
      "standard_self_term": raw_self / denominator,
      "standard_cross_term": raw_cross / denominator,
      "standard_score": (raw_self + raw_cross) / denominator,
      "loo_score": raw_cross / float(reference_size - 1),
  }


def staged_masked_value_and_grad(
    update_step: Callable[[Any], tuple[Any, Any]],
    inputs: Any,
    mask: jax.Array,
    *,
    has_aux: bool,
) -> tuple[jax.Array, Any | None, Any]:
  """Accumulates separately compiled update gradients using an exact mask."""
  size = batch_size(inputs)
  mask_f = mask.astype(jnp.float32)
  denominator = jnp.clip(jnp.sum(mask_f), 1.0)
  loss_sum = None
  aux_sum = None
  gradient_sum_value = None
  for index in range(size):
    out, gradient = update_step(sample_inputs(inputs, index))
    if has_aux:
      loss, aux = out
    else:
      loss, aux = out, None
    weight = mask_f[index]
    weighted_loss = jnp.squeeze(loss) * weight
    weighted_gradient = _staged_tree_scale(gradient, weight)
    loss_sum = weighted_loss if loss_sum is None else loss_sum + weighted_loss
    gradient_sum_value = (
        weighted_gradient
        if gradient_sum_value is None
        else _staged_tree_add(gradient_sum_value, weighted_gradient)
    )
    if has_aux:
      weighted_aux = _staged_tree_scale(aux, weight)
      aux_sum = weighted_aux if aux_sum is None else tree_add(aux_sum, weighted_aux)
  mean_aux = _staged_tree_scale(aux_sum, 1.0 / denominator) if has_aux else None
  return (
      loss_sum / denominator,
      mean_aux,
      _staged_tree_scale(gradient_sum_value, 1.0 / denominator),
  )
