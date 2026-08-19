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

"""GRPO entry point that initializes multi-host JAX before running training."""

import os
import runpy
import socket

import jax


def main() -> None:
  if not jax.distributed.is_initialized():
    coordinator_address = os.getenv("TUNIX_JAX_COORDINATOR_ADDRESS")
    num_processes = os.getenv("TUNIX_JAX_NUM_PROCESSES")
    process_id = os.getenv("TUNIX_JAX_PROCESS_ID")
    explicit_values = (coordinator_address, num_processes, process_id)
    if any(value is not None for value in explicit_values):
      if not all(value is not None for value in explicit_values):
        raise ValueError(
            "TUNIX_JAX_COORDINATOR_ADDRESS, TUNIX_JAX_NUM_PROCESSES, and"
            " TUNIX_JAX_PROCESS_ID must be set together."
        )
      jax.distributed.initialize(
          coordinator_address=coordinator_address,
          num_processes=int(num_processes),
          process_id=int(process_id),
      )
    else:
      jax.distributed.initialize()
  print(
      "JAX_DISTRIBUTED_IDENTITY"
      f" hostname={socket.gethostname()}"
      f" process_index={jax.process_index()}"
      f" process_count={jax.process_count()}"
      f" coordinator={os.getenv('TUNIX_JAX_COORDINATOR_ADDRESS', 'auto')}",
      flush=True,
  )
  runpy.run_module("tunix.cli.grpo_main", run_name="__main__")


if __name__ == "__main__":
  main()
