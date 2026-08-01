#!/usr/bin/env bash
# Reproduce the historical TPU Python environment in Jason's checkout.

set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
SOURCE_ENV="${SOURCE_ENV:-/home/lhf_hongfu_gmail_com/tunix/.venv}"
REFERENCE_FREEZE="${REFERENCE_FREEZE:-${REPO}/pip-freeze-lhf-reference.txt}"
TARGET_ENV="${TARGET_ENV:-${REPO}/.venv}"

if [[ ! -x "${SOURCE_ENV}/bin/python" ]]; then
  echo "SOURCE_ENV Python is not executable: ${SOURCE_ENV}/bin/python" >&2
  exit 1
fi
if [[ ! -r "${REFERENCE_FREEZE}" ]]; then
  echo "Reference freeze is not readable: ${REFERENCE_FREEZE}" >&2
  exit 1
fi
if [[ -e "${TARGET_ENV}" ]]; then
  echo "TARGET_ENV already exists; inspect or move it before retrying: ${TARGET_ENV}" >&2
  exit 1
fi

source_python_version="$("${SOURCE_ENV}/bin/python" -c 'import platform; print(platform.python_version())')"
echo "Creating ${TARGET_ENV} with the historical Python ${source_python_version}."

# Debian's system Python may not include ensurepip. The historical Python can
# still create a valid environment without pip, and its pip can seed the new
# interpreter through pip's --python option.
"${SOURCE_ENV}/bin/python" -m venv --without-pip "${TARGET_ENV}"
"${SOURCE_ENV}/bin/python" -m pip --python "${TARGET_ENV}/bin/python" install \
  pip==26.0.1 setuptools wheel

filtered_freeze="$(mktemp)"
trap 'rm -f "${filtered_freeze}"' EXIT

# Install the historical dependency snapshot but never reproduce an editable
# or file-URL installation of the LHF checkout.
grep -Ev \
  '^(google-tunix|tunix)(==| @)|^-e .*tunix|(^|[[:space:]])file://' \
  "${REFERENCE_FREEZE}" > "${filtered_freeze}"

"${TARGET_ENV}/bin/python" -m pip install -r "${filtered_freeze}"
"${TARGET_ENV}/bin/python" -m pip install --no-deps -e "${REPO}"

echo "Environment created. Verifying critical versions and import ownership."
cd "${REPO}"
"${TARGET_ENV}/bin/python" - <<'PY'
import importlib.metadata
import sys

import jax
import pytest
import tunix

print(f"python={sys.version}")
print(f"jax={jax.__version__}")
print(f"jaxlib={importlib.metadata.version('jaxlib')}")
print(f"libtpu={importlib.metadata.version('libtpu')}")
print(f"vllm={importlib.metadata.version('vllm')}")
print(f"pytest={pytest.__version__}")
print(f"tunix_source={tunix.__file__}")
PY

case "$("${TARGET_ENV}/bin/python" -c 'import tunix; print(tunix.__file__)')" in
  "${REPO}"/*) ;;
  *)
    echo "Tunix does not import from Jason's checkout." >&2
    exit 1
    ;;
esac

echo "Jason-owned environment is ready: ${TARGET_ENV}"
