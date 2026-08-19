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

import ipaddress
import os
import runpy
import socket

import jax
from jax.experimental import multihost_utils
import numpy as np


def _resolve_process_hosts() -> list[str] | None:
  """Resolves and validates the optional split-host routing candidates."""
  raw_hosts = os.getenv("TUNIX_PROCESS_HOSTS")
  if not raw_hosts:
    return None

  resolved_hosts = []
  for host in (item.strip() for item in raw_hosts.split(",")):
    if not host:
      continue
    try:
      addresses = {
          info[4][0]
          for info in socket.getaddrinfo(
              host, None, family=socket.AF_INET, type=socket.SOCK_STREAM
          )
      }
    except socket.gaierror as exc:
      raise ValueError(f"Cannot resolve TUNIX_PROCESS_HOSTS entry {host!r}.") from exc
    if len(addresses) != 1:
      raise ValueError(
          f"TUNIX_PROCESS_HOSTS entry {host!r} must resolve to exactly one"
          f" IPv4 address; got {sorted(addresses)}."
      )
    address = addresses.pop()
    ipaddress.IPv4Address(address)
    resolved_hosts.append(address)

  if not resolved_hosts:
    raise ValueError("TUNIX_PROCESS_HOSTS contains no usable hosts.")
  if len(set(resolved_hosts)) != len(resolved_hosts):
    raise ValueError(
        "TUNIX_PROCESS_HOSTS entries must resolve to distinct IPv4 addresses."
    )
  return resolved_hosts


def _discover_local_process_host(process_hosts: list[str]) -> str:
  """Finds this process's routable address without sending network traffic."""
  for peer_host in process_hosts:
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        # UDP connect performs only a local route lookup until data is sent.
        probe.connect((peer_host, 9))
        local_host = probe.getsockname()[0]
    except OSError:
      continue
    if local_host in process_hosts:
      return local_host
  raise RuntimeError(
      "Could not match this host's routable IPv4 address to"
      f" TUNIX_PROCESS_HOSTS={process_hosts}."
  )


def _rank_ordered_process_hosts(
    local_host: str, candidate_hosts: list[str]
) -> list[str]:
  """Collects host addresses in the TPU backend's actual process-rank order."""
  encoded_host = np.asarray(
      [int(ipaddress.IPv4Address(local_host))], dtype=np.uint32
  )
  gathered_hosts = np.asarray(
      multihost_utils.process_allgather(encoded_host, tiled=False)
  ).reshape(-1)
  process_hosts = [
      str(ipaddress.IPv4Address(int(encoded))) for encoded in gathered_hosts
  ]
  if len(process_hosts) != jax.process_count():
    raise RuntimeError(
        "Distributed host discovery returned the wrong number of hosts:"
        f" expected={jax.process_count()}, actual={len(process_hosts)},"
        f" hosts={process_hosts}."
    )
  if len(set(process_hosts)) != len(process_hosts):
    raise RuntimeError(
        f"Distributed host discovery returned duplicate hosts: {process_hosts}."
    )
  if set(process_hosts) != set(candidate_hosts):
    raise RuntimeError(
        "Distributed host discovery does not match TUNIX_PROCESS_HOSTS:"
        f" candidates={candidate_hosts}, discovered={process_hosts}."
    )
  return process_hosts


def _initialize_jax_distributed() -> tuple[str | None, list[str] | None]:
  """Initializes JAX and discovers transport hosts in actual rank order."""
  candidate_hosts = _resolve_process_hosts()
  if candidate_hosts is None:
    if not jax.distributed.is_initialized():
      jax.distributed.initialize()
    return None, None

  local_host = _discover_local_process_host(candidate_hosts)
  if not jax.distributed.is_initialized():
    # TPU device process indices are assigned by the backend topology. Passing
    # a coordination task id does not renumber those devices, so initialize
    # natively and discover the resulting rank-to-host mapping afterwards.
    jax.distributed.initialize()

  if jax.process_count() != len(candidate_hosts):
    raise RuntimeError(
        "JAX process count does not match TUNIX_PROCESS_HOSTS:"
        f" local_host={local_host}, candidate_hosts={candidate_hosts},"
        f" expected_process_count={len(candidate_hosts)},"
        f" actual_process_count={jax.process_count()}."
    )

  process_hosts = _rank_ordered_process_hosts(local_host, candidate_hosts)
  if process_hosts[jax.process_index()] != local_host:
    raise RuntimeError(
        "Rank-ordered host discovery assigned the wrong local host:"
        f" process_index={jax.process_index()}, local_host={local_host},"
        f" process_hosts={process_hosts}."
    )

  if (
      os.getenv("TUNIX_RESUME_ACTOR_CHECKPOINT_ROOT")
      and process_hosts[0] != candidate_hosts[0]
  ):
    raise RuntimeError(
        "Checkpoint continuation requires the launcher host to remain JAX"
        " process 0 because process 0 owns the actor/checkpoint mesh:"
        f" launcher_host={candidate_hosts[0]},"
        f" process_hosts={process_hosts}."
    )

  normalized_hosts = ",".join(process_hosts)
  os.environ["TUNIX_PROCESS_HOSTS"] = normalized_hosts
  return local_host, process_hosts


def main() -> None:
  local_host, process_hosts = _initialize_jax_distributed()
  print(
      "JAX_DISTRIBUTED_IDENTITY"
      f" hostname={socket.gethostname()}"
      f" process_index={jax.process_index()}"
      f" process_count={jax.process_count()}"
      f" local_host={local_host or 'auto'}"
      f" process_hosts={process_hosts or 'auto'}",
      flush=True,
  )
  runpy.run_module("tunix.cli.grpo_main", run_name="__main__")


if __name__ == "__main__":
  main()
