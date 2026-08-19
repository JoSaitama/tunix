# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for deterministic multi-host GRPO process identity discovery."""

import os
import socket
from unittest import mock

from absl.testing import absltest
from tunix.cli import grpo_main_distributed


class GrpoMainDistributedTest(absltest.TestCase):

  def test_resolve_process_hosts_normalizes_ipv4_addresses(self):
    def fake_getaddrinfo(host, *args, **kwargs):
      del args, kwargs
      addresses = {"actor": "10.0.0.2", "rollout": "10.0.0.7"}
      return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addresses[host], 0))]

    with mock.patch.dict(
        os.environ, {"TUNIX_PROCESS_HOSTS": "actor,rollout"}
    ), mock.patch.object(socket, "getaddrinfo", side_effect=fake_getaddrinfo):
      hosts = grpo_main_distributed._resolve_process_hosts()

    self.assertEqual(hosts, ["10.0.0.2", "10.0.0.7"])

  def test_resolve_process_hosts_rejects_duplicate_addresses(self):
    with mock.patch.dict(
        os.environ, {"TUNIX_PROCESS_HOSTS": "10.0.0.2,10.0.0.2"}
    ):
      with self.assertRaisesRegex(ValueError, "distinct IPv4"):
        grpo_main_distributed._resolve_process_hosts()

  def test_initialize_derives_rank_from_local_address(self):
    hosts = ["10.128.0.25", "10.128.0.24"]
    with mock.patch.dict(
        os.environ,
        {
            "TUNIX_PROCESS_HOSTS": ",".join(hosts),
            "TUNIX_DISTRIBUTED_ROLLOUT_PORT": "29600",
        },
    ), mock.patch.object(
        grpo_main_distributed, "_resolve_process_hosts", return_value=hosts
    ), mock.patch.object(
        grpo_main_distributed,
        "_discover_local_process_host",
        return_value="10.128.0.24",
    ), mock.patch.object(
        grpo_main_distributed.jax.distributed,
        "is_initialized",
        return_value=False,
    ), mock.patch.object(
        grpo_main_distributed.jax.distributed, "initialize"
    ) as initialize, mock.patch.object(
        grpo_main_distributed.jax, "process_index", return_value=1
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_count", return_value=2
    ):
      local_host, resolved_hosts = (
          grpo_main_distributed._initialize_jax_distributed()
      )

    initialize.assert_called_once_with(
        coordinator_address="10.128.0.25:29599",
        num_processes=2,
        process_id=1,
    )
    self.assertEqual(local_host, "10.128.0.24")
    self.assertEqual(resolved_hosts, hosts)

  def test_initialize_preserves_native_jax_fallback(self):
    with mock.patch.object(
        grpo_main_distributed, "_resolve_process_hosts", return_value=None
    ), mock.patch.object(
        grpo_main_distributed.jax.distributed,
        "is_initialized",
        return_value=False,
    ), mock.patch.object(
        grpo_main_distributed.jax.distributed, "initialize"
    ) as initialize:
      local_host, process_hosts = (
          grpo_main_distributed._initialize_jax_distributed()
      )

    initialize.assert_called_once_with()
    self.assertIsNone(local_host)
    self.assertIsNone(process_hosts)


if __name__ == "__main__":
  absltest.main()
