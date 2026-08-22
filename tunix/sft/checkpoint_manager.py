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

"""Checkpoint manager for PEFT."""

import dataclasses
import os
import time
from typing import Any, Tuple

from absl import logging
from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

_DEFAULT_CHECKPOINTING_OPTIONS = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,
    ),
    max_to_keep=3,
    enable_async_checkpointing=False,
)


def _process_local_barrier_sync_fn(*, key: str, timeout_ms: int) -> None:
  """Skips a distributed barrier for a single active checkpoint process."""
  del key, timeout_ms


class CheckpointManager:
  """Checkpoint manager for PEFT."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: ocp.CheckpointManagerOptions | None = None,
  ):
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
    """
    self._is_primary_process = jax.process_index() == 0
    self._root_directory = root_directory
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    checkpoint_manager_options = options or _DEFAULT_CHECKPOINTING_OPTIONS
    if root_directory is not None and self._is_primary_process:
      os.makedirs(root_directory, exist_ok=True)
      if jax.process_count() > 1:
        async_options = checkpoint_manager_options.async_options
        if not checkpoint_manager_options.enable_async_checkpointing:
          # Orbax also uses AsyncOptions for its save-progress tracker during
          # synchronous saves. That tracker does not need a distributed barrier
          # when this manager has exactly one active process.
          if async_options is None:
            async_options = ocp.options.AsyncOptions()
          async_options = dataclasses.replace(
              async_options,
              barrier_sync_fn=_process_local_barrier_sync_fn,
          )
        checkpoint_manager_options = dataclasses.replace(
            checkpoint_manager_options,
            create=False,
            async_options=async_options,
            multiprocessing_options=ocp.options.MultiprocessingOptions(
                primary_host=0,
                active_processes={0},
                barrier_sync_key_prefix='tunix_peft_process_0',
            ),
        )
        if not checkpoint_manager_options.enable_async_checkpointing:
          logging.info(
              'Configured process-local Orbax checkpoint synchronization for '
              'JAX process 0 of %d.',
              jax.process_count(),
          )
    self._checkpoint_manager_options = checkpoint_manager_options
    self._item_handlers: dict[str, Any] | None = None
    if root_directory is not None and self._is_primary_process:
      # When using Pathways, the checkpoint manager only supports persistence
      # APIs now.
      if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(
                use_ocdbt=False,
                use_zarr3=False,
            ),
            'optimizer_state': ocp.PyTreeCheckpointHandler(
                use_ocdbt=False,
                use_zarr3=False,
            ),
        }
        logging.info('Using persistence APIs for checkpointing with Pathways.')
      else:
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(),
            'optimizer_state': ocp.PyTreeCheckpointHandler(),
        }
      item_handlers['custom_metadata'] = ocp.JsonCheckpointHandler()
      self._item_handlers = item_handlers

  def _has_checkpoint_entries(self) -> bool:
    if not self._root_directory or not os.path.isdir(self._root_directory):
      return False
    with os.scandir(self._root_directory) as entries:
      return next(entries, None) is not None

  def _ensure_checkpoint_manager(self) -> ocp.CheckpointManager | None:
    if self._checkpoint_manager is not None:
      return self._checkpoint_manager
    if not self._is_primary_process or self._root_directory is None:
      return None
    if self._item_handlers is None:
      return None
    self._checkpoint_manager = ocp.CheckpointManager(
        self._root_directory,
        item_handlers=self._item_handlers,
        options=self._checkpoint_manager_options,
    )
    return self._checkpoint_manager

  def _prepare_save_item(self, item: Any) -> Any:
    return jax.tree.map(
        lambda x: np.asarray(x)
        if isinstance(x, jax.Array) and x.is_fully_addressable
        else x,
        item,
    )

  def _detach_external_restored_checkpoint(
      self, manager: ocp.CheckpointManager, step: int
  ) -> None:
    """Stops Orbax retention from deleting an external resume symlink."""
    if self._root_directory is None:
      return
    checkpoint_path = os.path.join(self._root_directory, str(step))
    if not os.path.islink(checkpoint_path):
      return
    checkpoint_target = os.path.realpath(checkpoint_path)
    root_directory = os.path.realpath(self._root_directory)
    if (
        os.path.commonpath((root_directory, checkpoint_target))
        == root_directory
    ):
      return
    os.unlink(checkpoint_path)
    manager.reload()
    logging.info(
        'Detached restored checkpoint symlink %s -> %s after successful '
        'restore; the source checkpoint remains unchanged.',
        checkpoint_path,
        checkpoint_target,
    )

  def latest_step(self) -> int | None:
    """Returns the latest step."""
    if not self._has_checkpoint_entries():
      return None
    manager = self._ensure_checkpoint_manager()
    if manager is None:
      return None
    return manager.latest_step()

  def save(
      self,
      step: int,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
  ) -> bool:
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save the params for.
      optimizer: The optimizer to save the params for. If None, the optimizer
        will not be saved.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save the checkpoint regardless of the save decision
        policy.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    manager = self._ensure_checkpoint_manager()
    if manager is None:
      return False
    if not force and not manager.should_save(step):
      return False
    logging.info(
        'Saving checkpoint for step %d to %s (force=%s).',
        step,
        self._root_directory,
        force,
    )
    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)
    params = self._prepare_save_item(params)

    model_cp_args = ocp.args.PyTreeSave(
        item=params, save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params)
    )

    cp_save_args = {
        'model_params': model_cp_args,
    }
    if optimizer is not None:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      optimizer_state = self._prepare_save_item(optimizer_state)
      optimizer_cp_args = ocp.args.PyTreeSave(
          item=optimizer_state,
          save_args=jax.tree.map(lambda _: ocp.SaveArgs(), optimizer_state),
      )
      cp_save_args['optimizer_state'] = optimizer_cp_args
    saved = manager.save(
        step,
        args=ocp.args.Composite(**cp_save_args),
        custom_metadata=custom_metadata or {},
        force=force,
    )
    logging.info(
        'Checkpoint save finished for step %d to %s: saved=%s.',
        step,
        self._root_directory,
        saved,
    )
    return saved

  def maybe_restore(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> Tuple[int, dict[str, Any]]:
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      optimizer: The optimizer to restore the params for. If None or if
        optimizer state is not found in the checkpoint, the optimizer will not
        be restored.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      The step of the restored checkpoint or 0 if no checkpoint is available.

    Raises:
      RuntimeError: If the checkpoint cannot be restored.
    """
    restore_start = time.time()
    if not self._has_checkpoint_entries():
      logging.info(
          'No checkpoint entries found under %s. Skipping restore.',
          self._root_directory,
      )
      return 0, {}
    manager = self._ensure_checkpoint_manager()
    if manager is None:
      return 0, {}
    if step is None:
      step = manager.latest_step()
      # If no checkpoint is available, return 0.
      if step is None:
        return 0, {}

    logging.info(
        'Restoring checkpoint from step %d under %s.',
        step,
        self._root_directory,
    )

    metadata = manager.metadata(step)

    # Load the params from the checkpoint.
    if restore_only_lora_params:
      abstract_params = nnx.state(model, nnx.LoRAParam)
    else:
      abstract_params = nnx.state(model)

    model_cp_args = ocp.args.PyTreeRestore(
        item=abstract_params,
        restore_args=ocp.checkpoint_utils.construct_restore_args(
            target=abstract_params
        ),
    )

    def fix_sharding(state):
      # Scalar values in optimizer states like step and count is initialized as
      # SingleDeviceSharding, which will fail if optimizer is sharded. To fix
      # it, we will replicate the scalar values.
      shardings = jax.tree_util.tree_map(lambda x: x.sharding, state)
      try:
        named_sharding = next(
            s
            for s in jax.tree_util.tree_leaves(shardings)
            if isinstance(s, jax.sharding.NamedSharding)
        )
        return nnx.get_named_sharding(optimizer_state, named_sharding.mesh)
      except StopIteration:
        return shardings

    if optimizer is not None and 'optimizer_state' in metadata.item_metadata:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      fixed_sharding = fix_sharding(optimizer_state)
      optimizer_cp_args = ocp.args.PyTreeRestore(
          item=optimizer_state,
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              target=optimizer_state, sharding_tree=fixed_sharding
          ),
      )
      ckpt = manager.restore(
          step,
          args=ocp.args.Composite(
              model_params=model_cp_args,
              optimizer_state=optimizer_cp_args,
          ),
      )
      nnx.update(optimizer, ckpt.optimizer_state)
    else:
      ckpt = manager.restore(
          step,
          args=ocp.args.Composite(
              model_params=model_cp_args,
          ),
      )
    # Update the model state with params from the restored checkpoint.
    nnx.update(model, ckpt.model_params)
    logging.info(
        'Restored params from step: %d in %.3f seconds',
        step,
        time.time() - restore_start,
    )
    custom_metadata = metadata.custom_metadata if metadata else {}
    self._detach_external_restored_checkpoint(manager, step)
    return step, custom_metadata

  def close(self):
    """Closes the checkpoint manager."""
    if self._checkpoint_manager is None:
      return
    self._checkpoint_manager.close()
