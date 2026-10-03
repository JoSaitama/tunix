#!/usr/bin/env bash
# Deploy a committed Project_7B snapshot; never starts training or changes RPC.
set -euo pipefail

stage=initialization
trap 'echo "Deployment stopped during ${stage} (line ${LINENO}); no training was started." >&2' ERR
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd -- "$script_dir/../.." && pwd)}"
[[ "$REPO" == */Project_7B/tunix ]] || {
  echo "REPO must be the separate Project_7B/tunix checkout: $REPO" >&2
  exit 2
}
OLD_REPO="${OLD_REPO:-${REPO%/Project_7B/tunix}/Project/tunix}"
VENV="$REPO/.venv"
python_bin="$VENV/bin/python"
REMOTE_WORKER_INDEX="${REMOTE_WORKER_INDEX:-1}"
phase=all
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --phase)
      [[ $# -ge 2 ]] || { echo "--phase requires all, source or env" >&2; exit 2; }
      phase="$2"; shift 2 ;;
    *) echo "Usage: bash $0 [--dry-run] [--phase all|source|env]" >&2; exit 2 ;;
  esac
done
case "$phase" in all|source|env) ;; *) echo "Invalid phase: $phase" >&2; exit 2 ;; esac
[[ "$REMOTE_WORKER_INDEX" == 0 || "$REMOTE_WORKER_INDEX" == 1 ]] || {
  echo "REMOTE_WORKER_INDEX must be 0 or 1 for this two-worker deployment." >&2
  exit 2
}
test -x "$python_bin"
cd "$REPO"
if [[ -n "$(git status --porcelain)" ]]; then
  git status --short
  echo "New checkout has local changes; commit or inspect them before deployment." >&2
  exit 1
fi
DEPLOY_SHA="$(git rev-parse HEAD)"
DEPLOY_DIR="$REPO/runs_xuesong/cache/deploy/$DEPLOY_SHA"
mkdir -p "$DEPLOY_DIR"

metadata_value() {
  curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 10 \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/$1"
}
stage=target_discovery
ZONE="${ZONE:-}"
TPU_PROJECT="${TPU_PROJECT:-}"
TPU_NAME="${TPU_NAME:-}"
if [[ -z "$ZONE" ]]; then
  if ! zone_resource="$(metadata_value instance/zone)"; then
    echo "Cannot read this VM's zone. Set ZONE to its actual zone and retry." >&2
    exit 1
  fi
  ZONE="${zone_resource##*/}"
fi
if [[ -z "$TPU_PROJECT" ]]; then
  if ! TPU_PROJECT="$(metadata_value project/project-id)"; then
    echo "Cannot read this VM's project. Set TPU_PROJECT to its actual project and retry." >&2
    exit 1
  fi
fi
for value in "$ZONE" "$TPU_PROJECT" "$TPU_NAME"; do
  [[ "$value" != *'<'* && "$value" != *'>'* ]] || {
    echo "Replace placeholder TPU_NAME/ZONE/TPU_PROJECT values or unset them for discovery." >&2
    exit 2
  }
done
local_ips="$(hostname -I)"
gcloud alpha compute tpus tpu-vm list \
  --project="$TPU_PROJECT" --zone="$ZONE" --format=json > "$DEPLOY_DIR/tpu-nodes.json"
resolved="$("$python_bin" - "$DEPLOY_DIR/tpu-nodes.json" "$local_ips" \
  "$TPU_NAME" "$REMOTE_WORKER_INDEX" <<'PY'
import json
import sys

path, local_ips, requested, remote_index = sys.argv[1:]
local_ips = set(local_ips.split())
matches = []
for node in json.load(open(path, encoding="utf-8")):
    name = node.get("name", "").rsplit("/", 1)[-1]
    endpoints = node.get("networkEndpoints", [])
    ips = [endpoint.get("ipAddress", "") for endpoint in endpoints]
    if (not requested or requested == name) and local_ips.intersection(ips):
        matches.append((name, ips))
if len(matches) != 1:
    raise SystemExit(
        f"Expected one TPU matching this host's IPs {sorted(local_ips)} "
        f"and requested name {requested!r}; found {len(matches)}. "
        "Check the project/zone and the saved tpu-nodes.json; no SSH was attempted."
    )
name, ips = matches[0]
if len(ips) != 2 or not all(ips) or len(set(ips)) != 2:
    raise SystemExit(f"Expected two distinct TPU worker endpoints, found {ips!r}.")
remote_ip = ips[int(remote_index)]
if remote_ip in local_ips:
    raise SystemExit("Selected remote worker is this host; choose the other worker index.")
print(name + "\t" + remote_ip)
PY
)"
IFS=$'\t' read -r TPU_NAME remote_ip <<< "$resolved"
printf 'Resolved target: project=%s zone=%s TPU=%s worker=%s IP=%s\n' \
  "$TPU_PROJECT" "$ZONE" "$TPU_NAME" "$REMOTE_WORKER_INDEX" "$remote_ip"
printf 'Source: %s at %s\nRemote directory: %s\n' "$REPO" "$DEPLOY_SHA" "$REPO"
printf 'Use these same settings for later training/evaluation:\n'
printf -v target_settings 'export TPU_PROJECT=%q CLOUDSDK_CORE_PROJECT=%q TPU_NAME=%q ZONE=%q REMOTE_WORKER_INDEX=%q\n' \
  "$TPU_PROJECT" "$TPU_PROJECT" "$TPU_NAME" "$ZONE" "$REMOTE_WORKER_INDEX"
printf '%s' "$target_settings"
printf '%s' "$target_settings" > "$DEPLOY_DIR/target.env"
if [[ "$dry_run" == true ]]; then
  echo 'Dry run complete: no remote commands or file transfers were executed.'
  exit 0
fi

cloud_args=("--project=$TPU_PROJECT" "--zone=$ZONE" "--worker=$REMOTE_WORKER_INDEX" --internal-ip)
remote_command() {
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" "${cloud_args[@]}" \
    --command="bash -lc $(printf '%q' "$1")"
}
if [[ "$phase" != env ]]; then
  stage=source_archive
  git archive --format=tar.gz --output="$DEPLOY_DIR/source.tar.gz" HEAD
  git ls-files -z | xargs -0 sha256sum > "$DEPLOY_DIR/source.sha256"
  (cd "$DEPLOY_DIR"; sha256sum source.tar.gz > archive.sha256)
  stage=remote_preparation
  printf -v prepare 'set -e\ntest -x %q\nmkdir -p %q' \
    "$OLD_REPO/.venv/bin/python" "$DEPLOY_DIR"
  remote_command "$prepare"
  stage=file_transfer
  for artifact in source.tar.gz archive.sha256 source.sha256; do
    gcloud alpha compute tpus tpu-vm scp "$DEPLOY_DIR/$artifact" \
      "$TPU_NAME:$DEPLOY_DIR/$artifact" "${cloud_args[@]}"
  done
  stage=source_verification
  printf -v install \
    'set -e\ncd %q\nsha256sum -c archive.sha256\ntar -xzf source.tar.gz -C %q\ncd %q\nsha256sum -c %q\nprintf "%%s\\n" %q > .deployed_git_head' \
    "$DEPLOY_DIR" "$REPO" "$REPO" "$DEPLOY_DIR/source.sha256" "$DEPLOY_SHA"
  remote_command "$install"
fi
if [[ "$phase" != source ]]; then
  stage=remote_environment_and_cpu_checks
  expected_versions="$(env -u PYTHONPATH JAX_PLATFORMS=cpu \
    PYTHONDONTWRITEBYTECODE=1 "$python_bin" - <<'PY'
import json
import platform
from importlib.metadata import version

versions = {name: version(name) for name in ("jax", "jaxlib", "libtpu", "vllm")}
versions["python"] = platform.python_version()
print(json.dumps(versions, sort_keys=True))
PY
  )"
  printf -v header 'export REPO=%q OLD_REPO=%q VENV=%q EXPECTED_SHA=%q EXPECTED_VERSIONS=%q\n' \
    "$REPO" "$OLD_REPO" "$VENV" "$DEPLOY_SHA" "$expected_versions"
  body=$(cat <<'SH'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
test "$(cat "$REPO/.deployed_git_head")" = "$EXPECTED_SHA"
if [[ ! -e "$VENV" && ! -L "$VENV" ]]; then
  test -x "$OLD_REPO/.venv/bin/python"
  SOURCE_ENV="$(env -u PYTHONPATH JAX_PLATFORMS=cpu \
    "$OLD_REPO/.venv/bin/python" - <<'PY'
from pathlib import Path
from importlib.metadata import distribution

site = Path(distribution("jax").locate_file("")).resolve()
for candidate in (site, *site.parents):
    if (candidate / "bin/python").is_file():
        print(candidate)
        break
else:
    raise SystemExit(f"Cannot locate dependency environment from {site}")
PY
  )"
  env -u PYTHONPATH -u PYTHON_BIN REPO="$REPO" SOURCE_ENV="$SOURCE_ENV" \
    TARGET_ENV="$VENV" JAX_PLATFORMS=cpu \
    bash "$REPO/runs_xuesong/scripts/bootstrap_jason_env.sh"
fi
# Check the new shim itself, without cwd/PYTHONPATH masking an old .pth file.
cd "$REPO/runs_xuesong/cache"
env -u PYTHONPATH JAX_PLATFORMS=cpu "$VENV/bin/python" - <<'PY'
import os
import json
from pathlib import Path
import platform
from importlib.metadata import version
import tunix

source = Path(tunix.__file__).resolve()
if Path(os.environ["REPO"]).resolve() not in source.parents:
    raise SystemExit(f"Wrong Tunix source: {source}")
print(f"tunix_source={source}")
versions = {name: version(name) for name in ("jax", "jaxlib", "libtpu", "vllm")}
versions["python"] = platform.python_version()
for package, value in sorted(versions.items()):
    print(f"{package}={value}")
expected = json.loads(os.environ["EXPECTED_VERSIONS"])
if versions != expected:
    raise SystemExit(f"Worker dependency versions differ: expected {expected}, got {versions}")
PY
cd "$REPO"
env -u PYTHON_BIN PYTHONPATH="$REPO" JAX_PLATFORMS=cpu \
  "$VENV/bin/python" -m pytest -q \
  tests/cli/aime_launchers_test.py tests/models/qwen2/deepseek_qwen_config_test.py \
  tests/scripts/dual_worker_status_test.py tests/cli/grpo_main_distributed_test.py \
  tests/cli/config_test.py tests/cli/recipes/deepscaler_eval_test.py
SH
  )
  remote_command "$header$body"
fi
echo "Deployment phase '$phase' completed. No training was started."
