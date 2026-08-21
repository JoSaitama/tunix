#!/usr/bin/env bash
set -euo pipefail

# Read-only fingerprint for comparing the physical TPU VM worker that hosts
# vLLM. Run it once on worker 0 and once on worker 1 from the same checkout.

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
PYTHON_BIN="${PYTHON_BIN:-${REPO}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "RUNTIME_FINGERPRINT_VERSION=1"
echo "HOSTNAME=$(hostname)"
echo "HOSTNAME_FQDN=$(hostname -f 2>/dev/null || true)"
echo "IDENTITY=$(id)"
echo "USER=${USER:-}"
echo "LOGNAME=${LOGNAME:-}"
echo "HOME=${HOME:-}"
echo "PWD=$(pwd)"
echo "UMASK=$(umask)"
echo "UTC_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "KERNEL=$(uname -a)"
echo "PYTHON_BIN=$(realpath "${PYTHON_BIN}")"
echo "VENV_PATH=$(realpath "${REPO}/.venv" 2>/dev/null || true)"
echo "PYTHON_VERSION=$(${PYTHON_BIN} -VV 2>&1 | tr '\n' ' ')"
echo "WORKTREE_HEAD=$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || true)"
echo "WORKTREE_STATUS_BEGIN"
git -C "${REPO}" status --short 2>/dev/null || true
echo "WORKTREE_STATUS_END"

echo "RESOURCE_LIMITS_BEGIN"
ulimit -a
echo "RESOURCE_LIMITS_END"

echo "FILESYSTEM_BEGIN"
df -h / /tmp "${REPO}" 2>/dev/null || true
df -i / /tmp "${REPO}" 2>/dev/null || true
echo "FILESYSTEM_END"

echo "MEMORY_BEGIN"
free -h 2>/dev/null || true
echo "MEMORY_END"

echo "RUNTIME_ENV_BEGIN"
env | LC_ALL=C sort |
  grep -E '^(JAX|TPU|VLLM|XLA|TUNIX|OMP|PJRT|LIBTPU|XDG|HF)_|^(HOME|USER|LOGNAME|TMPDIR|PATH)=' ||
  true
echo "RUNTIME_ENV_END"

echo "PATH_PERMISSIONS_BEGIN"
namei -l "${REPO}" "${REPO}/.venv" "${PYTHON_BIN}" 2>/dev/null || true
echo "PATH_PERMISSIONS_END"

echo "ACTIVE_PROCESSES_BEGIN"
ps -eo pid,ppid,lstart,stat,nlwp,pcpu,pmem,args --sort=pid |
  grep -E 'python|vllm|grpo|libtpu|ray' |
  grep -v grep || true
echo "ACTIVE_PROCESSES_END"

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys

packages = (
    "jax",
    "jaxlib",
    "vllm",
    "libtpu",
    "tpu-info",
    "numpy",
    "flax",
    "orbax-checkpoint",
)

print("PYTHON_RUNTIME_BEGIN")
print(f"executable={sys.executable}")
print(f"version={sys.version.replace(chr(10), ' ')}")
print(f"platform={platform.platform()}")
for package in packages:
  try:
    version = importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    version = "MISSING"
  print(f"package={package} version={version}")

module_origins: list[Path] = []
for module_name in ("jax", "jaxlib", "vllm", "libtpu"):
  spec = importlib.util.find_spec(module_name)
  if spec is not None and spec.origin:
    module_origins.append(Path(spec.origin))
  print(
      f"module={module_name} origin="
      f"{None if spec is None else spec.origin}"
  )
print("PYTHON_RUNTIME_END")

candidate_roots: set[Path] = set()
for package in ("jaxlib", "vllm", "libtpu"):
  try:
    distribution = importlib.metadata.distribution(package)
  except importlib.metadata.PackageNotFoundError:
    continue
  candidate_roots.add(Path(distribution.locate_file("")))

explicit_libtpu = os.environ.get("TPU_LIBRARY_PATH") or os.environ.get(
    "LIBTPU_PATH"
)
explicit_files = [Path(explicit_libtpu)] if explicit_libtpu else []

native_files: set[Path] = {
    path for path in (*explicit_files, *module_origins) if path.suffix == ".so"
}
for root in candidate_roots:
  for pattern in (
      "jaxlib/**/*.so",
      "vllm/**/*.so",
      "libtpu/**/*.so",
      "libtpu*.so",
  ):
    native_files.update(path for path in root.glob(pattern) if path.is_file())

print("NATIVE_HASHES_BEGIN")
for path in sorted(native_files, key=str):
  digest = hashlib.sha256()
  try:
    with path.open("rb") as stream:
      for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    print(
        f"sha256={digest.hexdigest()} size={path.stat().st_size} path={path}"
    )
  except OSError as exc:
    print(f"hash_error={exc!r} path={path}")
print("NATIVE_HASHES_END")
PY
