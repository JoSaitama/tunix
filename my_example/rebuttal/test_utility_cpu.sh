#!/usr/bin/env bash
set -euo pipefail
AUDIT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$AUDIT_REPO"
AUDIT_PYTHON="${AUDIT_PYTHON:-$AUDIT_REPO/.venv_jax081/bin/python}"
env -u JAX_PLATFORM_NAME JAX_PLATFORMS=cpu \
  "$AUDIT_PYTHON" -m unittest discover \
  -s my_example/rebuttal/tests -p '*_test.py'
