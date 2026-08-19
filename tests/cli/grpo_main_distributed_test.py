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

from contextlib import redirect_stdout
import io
import os
import socket
from unittest import mock

from absl.testing import absltest
import numpy as np
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

  def test_rank_ordered_process_hosts_uses_backend_process_order(self):
    candidates = ["10.128.0.25", "10.128.0.24"]
    encoded = np.asarray(
        [
            [int(grpo_main_distributed.ipaddress.IPv4Address("10.128.0.24"))],
            [int(grpo_main_distributed.ipaddress.IPv4Address("10.128.0.25"))],
        ],
        dtype=np.uint32,
    )
    with mock.patch.object(
        grpo_main_distributed.multihost_utils,
        "process_allgather",
        return_value=encoded,
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_count", return_value=2
    ):
      hosts = grpo_main_distributed._rank_ordered_process_hosts(
          "10.128.0.24", candidates
      )

    self.assertEqual(hosts, ["10.128.0.24", "10.128.0.25"])

  def test_initialize_accepts_backend_assigned_rank_order(self):
    hosts = ["10.128.0.25", "10.128.0.24"]
    rank_ordered_hosts = ["10.128.0.24", "10.128.0.25"]
    with mock.patch.dict(
        os.environ,
        {"TUNIX_PROCESS_HOSTS": ",".join(hosts)},
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
        grpo_main_distributed,
        "_rank_ordered_process_hosts",
        return_value=rank_ordered_hosts,
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_index", return_value=0
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_count", return_value=2
    ):
      local_host, resolved_hosts = (
          grpo_main_distributed._initialize_jax_distributed()
      )
      normalized_hosts = os.environ["TUNIX_PROCESS_HOSTS"]

    initialize.assert_called_once_with()
    self.assertEqual(local_host, "10.128.0.24")
    self.assertEqual(resolved_hosts, rank_ordered_hosts)
    self.assertEqual(normalized_hosts, "10.128.0.24,10.128.0.25")

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

  def test_resume_rejects_actor_host_rank_change(self):
    candidates = ["10.128.0.25", "10.128.0.24"]
    rank_ordered_hosts = ["10.128.0.24", "10.128.0.25"]
    with mock.patch.dict(
        os.environ,
        {
            "TUNIX_PROCESS_HOSTS": ",".join(candidates),
            "TUNIX_RESUME_ACTOR_CHECKPOINT_ROOT": "/checkpoint/actor",
        },
    ), mock.patch.object(
        grpo_main_distributed,
        "_resolve_process_hosts",
        return_value=candidates,
    ), mock.patch.object(
        grpo_main_distributed,
        "_discover_local_process_host",
        return_value="10.128.0.24",
    ), mock.patch.object(
        grpo_main_distributed.jax.distributed,
        "is_initialized",
        return_value=True,
    ), mock.patch.object(
        grpo_main_distributed,
        "_rank_ordered_process_hosts",
        return_value=rank_ordered_hosts,
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_index", return_value=0
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_count", return_value=2
    ):
      with self.assertRaisesRegex(RuntimeError, "actor/checkpoint mesh"):
        grpo_main_distributed._initialize_jax_distributed()

  def test_main_emits_normal_teardown_boundaries(self):
    registered_callbacks = []
    with mock.patch.object(
        grpo_main_distributed,
        "_initialize_jax_distributed",
        return_value=("10.0.0.2", ["10.0.0.2", "10.0.0.7"]),
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_index", return_value=0
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_count", return_value=2
    ), mock.patch.object(
        grpo_main_distributed.socket,
        "gethostname",
        return_value="actor-host",
    ), mock.patch.object(
        grpo_main_distributed.runpy, "run_module"
    ) as run_module, mock.patch.object(
        grpo_main_distributed.atexit,
        "register",
        side_effect=lambda callback, *args, **kwargs: registered_callbacks.append(
            (callback, args, kwargs)
        ),
    ):
      output = io.StringIO()
      with redirect_stdout(output):
        grpo_main_distributed.main()

    run_module.assert_called_once_with(
        "tunix.cli.grpo_main", run_name="__main__"
    )
    self.assertLen(registered_callbacks, 1)
    self.assertIn("phase=runpy_start", output.getvalue())
    self.assertIn("phase=runpy_return", output.getvalue())
    self.assertIn("phase=runpy_finally", output.getvalue())

  def test_main_records_system_exit_code(self):
    with mock.patch.object(
        grpo_main_distributed,
        "_initialize_jax_distributed",
        return_value=("10.0.0.2", ["10.0.0.2", "10.0.0.7"]),
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_index", return_value=0
    ), mock.patch.object(
        grpo_main_distributed.jax, "process_count", return_value=2
    ), mock.patch.object(
        grpo_main_distributed.socket,
        "gethostname",
        return_value="actor-host",
    ), mock.patch.object(
        grpo_main_distributed.runpy,
        "run_module",
        side_effect=SystemExit(1),
    ), mock.patch.object(grpo_main_distributed.atexit, "register"):
      output = io.StringIO()
      with redirect_stdout(output), self.assertRaises(SystemExit):
        grpo_main_distributed.main()

    self.assertIn("phase=runpy_system_exit", output.getvalue())
    self.assertIn("exit_code=1", output.getvalue())
    self.assertIn("phase=runpy_finally", output.getvalue())


if __name__ == "__main__":
  absltest.main()
