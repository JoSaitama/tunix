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

"""Utilities for sharding tensors."""

from typing import Tuple

import jax
from jax.interpreters import pxla
import jax.sharding as shd
import numpy as np


def shard_input(
    input_data: jax.Array, data_sharding_axis: Tuple[str, ...]
) -> jax.Array:
  """Shards the input data across the available devices.

  Args:
    input_data: The input data to be sharded, expected to be a TrainingInput
      dataclass.
    data_sharding_axis: The sharding axis for the input data, e.g. ("fsdp",).

  Returns:
    The sharded TrainingInput.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty:
    return input_data

  pspec = shd.PartitionSpec(*data_sharding_axis)
  # Check if the input is already sharded with the target mesh to avoid
  # re-sharding.
  is_sharded = jax.tree.map(
      lambda x: isinstance(x, jax.Array)
      and hasattr(x, "sharding")
      and hasattr(x.sharding, "mesh")
      and x.sharding.mesh == mesh
      and hasattr(x.sharding, "spec")
      and x.sharding.spec == pspec,
      input_data,
  )
  if all(jax.tree.leaves(is_sharded)):
    return input_data

  with jax.transfer_guard("allow"):
      # If x is already a JAX array, we might not need to do anything or just reshard.
      # device_put requires fully addressable data (host-local).
      # If x is a global sharded array, we should use jax.lax.with_sharding_constraint or similar,
      # but make_array_from_process_local_data is definitely wrong for global arrays.
      def _shard_leaf(x):
          if isinstance(x, jax.Array) and not x.is_fully_addressable:
               # It's a global array. Correctly resharding global arrays usually involves
               # jax.lax.with_sharding_constraint if we are inside jit, or device_put with sharding
               # if we are outside. However, device_put(global_array, sharding) *is* supported
               # and handles resharding. The issue was make_array_from_process_local_data.
               return jax.device_put(x, get_sharding(x, mesh=mesh, pspec=pspec))
          
          # For local data (processed on host) or fully addressable arrays:
          return jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), x
        )

      return jax.tree.map(_shard_leaf, input_data)


def get_sharding(x: jax.Array, mesh: shd.Mesh, pspec: shd.PartitionSpec):
  """Get a sharding for an tensor given a mesh and partition spec."""
  # Only shard arrays with rank > 0.
  if not isinstance(x, (np.ndarray, jax.Array)) or x.ndim == 0:
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Don't shard if rank is not sufficient.
  if x.ndim < len(pspec):
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Check for divisibility for all sharded axes.
  for i, axis_name in enumerate(pspec):
    if axis_name is not None:
      axis_names = axis_name if isinstance(axis_name, tuple) else (axis_name,)
      for name in axis_names:
        axis_size = mesh.shape[name]
        if x.shape[i] % axis_size != 0:
          # Replicate if not evenly divisible.
          return shd.NamedSharding(mesh, shd.PartitionSpec())
  return shd.NamedSharding(mesh, pspec)
