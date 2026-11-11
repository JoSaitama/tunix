"""Opt-in GRPO actor update using groupwise PCGrad or CAGrad."""

from __future__ import annotations

from collections.abc import Mapping

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import cagrad
from tunix.rl import pcgrad
from tunix.rl import trainer as rl_trainer


def configuration_from_environment(environ: Mapping[str, str]):
  method = environ.get("TUNIX_GRADIENT_CONFLICT_METHOD", "")
  include_kl = environ.get("TUNIX_GRADIENT_CONFLICT_INCLUDE_KL", "0")
  c = float(environ.get("TUNIX_CAGRAD_C", "0.5"))
  seed = int(environ.get("TUNIX_EXPERIMENT_SEED") or "0")
  if method not in ("pcgrad", "cagrad"):
    raise ValueError("Gradient conflict method must be pcgrad or cagrad.")
  if include_kl not in ("0", "1"):
    raise ValueError("TUNIX_GRADIENT_CONFLICT_INCLUDE_KL must be 0 or 1.")
  if not 0 <= c < 1:
    raise ValueError("CAGrad c must be in [0, 1).")
  if not 0 <= seed < 2**32:
    raise ValueError("Gradient conflict seed must fit in an unsigned 32-bit integer.")
  if (
      environ.get("TUNIX_DYNAMIC_SAMPLING") == "1"
      or environ.get("TUNIX_PODS_KEEP_COUNT", "").strip()
  ):
    raise ValueError(
        "Gradient conflict baselines cannot be combined with DS or PODS."
    )
  return dict(method=method, include_kl=include_kl == "1", c=c, seed=seed)


def group_gram(grads):
  """Return one FP32 Gram matrix for each contiguous four-completion group."""
  leaves = jax.tree.leaves(grads)
  if not leaves or not leaves[0].shape or not leaves[0].shape[0]:
    raise ValueError("Expected per-completion gradients.")
  count = leaves[0].shape[0]
  if count % 4 or any(not x.shape or x.shape[0] != count for x in leaves):
    raise ValueError("Expected complete four-completion gradient groups.")
  gram = jnp.zeros((count // 4, 4, 4), jnp.float32)
  for leaf in leaves:
    rows = leaf.reshape(count // 4, 4, -1).astype(jnp.float32)
    gram += jnp.matmul(
        rows, rows.swapaxes(1, 2), precision=jax.lax.Precision.HIGHEST
    )
  return gram


def _check_finite(loss, gram, direction):
  finite = jnp.isfinite(loss) & jnp.all(jnp.isfinite(gram))
  for leaf in jax.tree.leaves(direction):
    finite &= jnp.all(jnp.isfinite(leaf))

  def fail():
    def raise_error():
      raise FloatingPointError("Nonfinite loss or gradient in PCGrad/CAGrad.")

    jax.debug.callback(raise_error)

  jax.lax.cond(finite, lambda: None, fail)


class GradientConflictTrainer(rl_trainer.Trainer):
  """Average per-prompt directions, then apply the original actor optimizer."""

  def __init__(self, *args, method, include_kl=False, c=0.5, seed=0, **kwargs):
    super().__init__(*args, **kwargs)
    self.method = method
    self.include_kl = include_kl
    self.c = c
    self.seed = seed
    self._policy_loss = None
    self._policy_has_aux = False
    self._initial_iter_step = self.iter_steps

  def with_policy_score_loss_fn(self, loss_fn, *, has_aux=False):
    """Reuse the existing GRPO beta=0 loss hook."""
    self.clear_jit_cache()
    self._policy_loss, self._policy_has_aux = loss_fn, has_aux

  def _train_step(self, model, optimizer, inputs):
    if self._policy_loss is None:
      raise RuntimeError("Gradient conflict baselines require the GRPO loss hook.")
    inputs = self.gen_model_input_fn(inputs)
    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param

    def gradient(loss_fn, has_aux):
      def loss(model, example):
        if isinstance(example, dict):
          return loss_fn(model, **example)
        return loss_fn(model, example)

      return nnx.value_and_grad(
          loss, argnums=nnx.DiffState(0, wrt), has_aux=has_aux
      )

    def per_completion(loss_fn, has_aux):
      return jax.vmap(gradient(loss_fn, has_aux), in_axes=(None, 0))(model, inputs)

    if self.include_kl:
      output, grads = per_completion(self.loss_fn, self._has_aux)
      output = jax.tree.map(lambda value: jnp.mean(value, axis=0), output)
    else:
      # Compute the ordinary full batch gradient, then replace only its
      # policy mean by the transformed direction. This keeps the full KL
      # contribution without a second per-completion gradient tree.
      output, full_batch_gradient = gradient(self.loss_fn, self._has_aux)(
          model, inputs
      )
      _, grads = per_completion(self._policy_loss, self._policy_has_aux)
    loss, aux = output if self._has_aux else (output, None)

    gram = group_gram(grads)
    if self.method == "pcgrad":
      step = optimizer.step[...] + self._initial_iter_step
      key = jax.random.fold_in(jax.random.key(self.seed), step)
      weights, projections = jax.vmap(pcgrad.coefficients)(
          gram, jax.random.split(key, gram.shape[0])
      )
    else:
      weights = jax.vmap(lambda matrix: cagrad.coefficients(matrix, self.c))(
          gram
      )
    change = weights - 0.25
    applied = weights if self.include_kl else change
    applied = applied.reshape(-1) / gram.shape[0]
    direction = jax.tree.map(
        lambda leaf: jnp.tensordot(
            applied, leaf, axes=1, precision=jax.lax.Precision.HIGHEST
        ),
        grads,
    )
    if not self.include_kl:
      direction = jax.tree.map(jnp.add, full_batch_gradient, direction)
    _check_finite(loss, gram, direction)
    optimizer.update(model, direction)

    if getattr(self, "_benchmark_quiet", False):
      return loss, None
    if isinstance(aux, dict):
      aux["skipped_samples"] = jnp.asarray(0.0)
      aux["gradient_conflict_negative_pairs"] = jnp.mean(
          jnp.sum(jnp.triu(gram < 0, k=1), axis=(1, 2))
      )
      shift_squared = jnp.einsum(
          "bi,bij,bj->b", change, gram, change,
          precision=jax.lax.Precision.HIGHEST,
      )
      aux["gradient_conflict_shift_norm"] = jnp.mean(
          jnp.sqrt(jnp.maximum(shift_squared, 0))
      )
      if self.method == "pcgrad":
        aux["pcgrad_projections"] = jnp.mean(projections)
    return loss, aux

  def _post_process_train_step(self, aux):
    for key in (
        "gradient_conflict_negative_pairs",
        "gradient_conflict_shift_norm",
        "pcgrad_projections",
    ):
      self.rl_metrics_to_log.setdefault(key, np.mean)
    super()._post_process_train_step(aux)
