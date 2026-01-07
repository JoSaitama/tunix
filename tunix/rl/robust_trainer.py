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

"""Robust RL Trainer with Dynamic Batch Curation."""

from typing import Any, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax
from tunix.rl import trainer as rl_trainer


class RobustTrainer(rl_trainer.Trainer):
  """Trainer that implements Dynamic Batch Curation to filter gradient outliers."""

  def __init__(
      self,
      *args,
      curation_threshold: float = 3.0,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.curation_threshold = curation_threshold

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    """Train step with dynamic batch curation."""
    inputs = self.gen_model_input_fn(inputs)

    # 1. Define per-sample loss and grad function
    def per_sample_loss_fn(model, inputs):
      if isinstance(inputs, dict):
          return self.loss_fn(model, **inputs)
      return self.loss_fn(model, inputs)

    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    
    # 2. Compute per-sample gradients using vmap
    grad_fn = nnx.value_and_grad(per_sample_loss_fn, argnums=nnx.DiffState(0, wrt), has_aux=self._has_aux)
    
    if isinstance(inputs, dict):
        inputs_in_axes = jax.tree_util.tree_map(lambda _: 0, inputs)
        vmapped_grad_fn = jax.vmap(grad_fn, in_axes=(None, inputs_in_axes))
        (per_sample_loss, per_sample_aux), per_sample_grads = vmapped_grad_fn(model, inputs)
    else:
        inputs_in_axes = jax.tree_util.tree_map(lambda _: 0, inputs)
        vmapped_grad_fn = jax.vmap(grad_fn, in_axes=(None, inputs_in_axes))
        (per_sample_loss, per_sample_aux), per_sample_grads = vmapped_grad_fn(model, inputs)

    # 3. Compute Gradient Norms (L2)
    def compute_global_norm(grads):
        return optax.global_norm(grads)
    
    per_sample_grad_norms = jax.vmap(compute_global_norm)(per_sample_grads)
    
    # 4. Detect Outliers
    mean_norm = jnp.mean(per_sample_grad_norms)
    std_norm = jnp.std(per_sample_grad_norms)
    cutoff = mean_norm + self.curation_threshold * std_norm
    
    # Create mask: 1.0 if keep, 0.0 if outlier
    mask = per_sample_grad_norms <= cutoff
    
    num_skipped = jnp.sum(1.0 - mask)
    
    # 5. Aggregate Gradients
    def apply_mask_and_avg(grads, mask):
        return jax.tree_util.tree_map(
            lambda g: jnp.sum(g * mask.reshape((-1,) + (1,) * (g.ndim - 1)), axis=0) / jnp.clip(jnp.sum(mask), 1.0),
            grads
        )

    final_grads = apply_mask_and_avg(per_sample_grads, mask)
    
    # 6. Aggregate Loss (for logging)
    final_loss = jnp.sum(per_sample_loss * mask) / jnp.clip(jnp.sum(mask), 1.0)
    
    optimizer.update(model, final_grads)
    
    if self._has_aux:
        final_aux = jax.tree_util.tree_map(
            lambda x: jnp.sum(x * mask.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0) / jnp.clip(jnp.sum(mask), 1.0),
            per_sample_aux
        )
        if isinstance(final_aux, dict):
             final_aux["skipped_samples"] = num_skipped
             final_aux["grad_norm_mean"] = mean_norm
             final_aux["grad_norm_std"] = std_norm

        return final_loss, final_aux
    else:
        return final_loss, None
