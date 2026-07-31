#!/bin/bash
# Copyright 2026 Google LLC
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

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

maybe_launch_all_workers() {
  if [[ "${TUNIX_MULTIHOST_LAUNCHED:-0}" == "1" ]]; then
    return 1
  fi
  if [[ -n "${TUNIX_PROCESS_HOSTS:-}" || "${TUNIX_INIT_JAX_DISTRIBUTED:-0}" == "1" ]]; then
    return 1
  fi
  if [[ "${TUNIX_SKIP_MULTIHOST_LAUNCH:-0}" == "1" ]]; then
    return 1
  fi
  if ! command -v gcloud >/dev/null 2>&1; then
    return 1
  fi

  local fqdn
  fqdn="$(hostname -f 2>/dev/null || true)"
  if [[ ! "${fqdn}" =~ -w-0(\.|$) ]]; then
    return 1
  fi

  local zone current_ip tpu_name remote_args remote_cmd
  zone="$(
    curl -fs -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/zone \
      2>/dev/null | awk -F/ '{print $NF}'
  )"
  current_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -z "${zone}" || -z "${current_ip}" ]]; then
    return 1
  fi

  tpu_name="$(
    gcloud compute tpus tpu-vm list --zone="${zone}" --format=json 2>/dev/null \
      | CURRENT_IP="${current_ip}" python3 -c '
import json
import os
import sys

current_ip = os.environ["CURRENT_IP"]
for node in json.load(sys.stdin):
  for endpoint in node.get("networkEndpoints", []):
    if endpoint.get("ipAddress") == current_ip:
      print(node["name"])
      raise SystemExit(0)
raise SystemExit(1)
'
  )"
  if [[ -z "${tpu_name}" ]]; then
    return 1
  fi

  printf -v remote_args ' %q' "$@"
  printf -v remote_cmd \
    'cd %q && export TUNIX_INIT_JAX_DISTRIBUTED=1 TUNIX_MULTIHOST_LAUNCHED=1 PYTHONUNBUFFERED=1; bash %q%s' \
    "${repo_root}" \
    "${repo_root}/examples/deepscaler/run_deepscaler_disagg_v5p16_selfinf_group_1epoch.sh" \
    "${remote_args}"

  exec gcloud alpha compute tpus tpu-vm ssh "${tpu_name}" \
    --zone="${zone}" \
    --worker=all \
    --internal-ip \
    --command="${remote_cmd}"
}

maybe_launch_all_workers "$@" || true

bash "${repo_root}/examples/deepscaler/run_deepscaler_disagg_v5p16_1epoch.sh" \
  rl_training_config.dynamic_batch_curation_variant="self_inf_group" \
  rl_training_config.self_influence_dot_threshold=0.0 \
  "$@"
