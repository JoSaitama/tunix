#!/usr/bin/env bash
# Create a lightweight Jason-owned environment backed by the historical TPU
# packages. This avoids copying or rebuilding the large TPU vLLM stack.

set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
SOURCE_ENV="${SOURCE_ENV:-/home/lhf_hongfu_gmail_com/tunix/.venv}"
TARGET_ENV="${TARGET_ENV:-${REPO}/.venv}"

if [[ ! -x "${SOURCE_ENV}/bin/python" ]]; then
  echo "SOURCE_ENV Python is not executable: ${SOURCE_ENV}/bin/python" >&2
  exit 1
fi
if [[ -e "${TARGET_ENV}" ]]; then
  echo "TARGET_ENV already exists; remove the failed generated environment before retrying: ${TARGET_ENV}" >&2
  exit 1
fi

source_python_version="$("${SOURCE_ENV}/bin/python" -c 'import platform; print(platform.python_version())')"
source_site_packages="$("${SOURCE_ENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"

echo "Creating lightweight ${TARGET_ENV} with Python ${source_python_version}."
"${SOURCE_ENV}/bin/python" -m venv --without-pip "${TARGET_ENV}"

target_site_packages="$("${TARGET_ENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
path_file="${target_site_packages}/aime_reproduction_paths.pth"

mkdir -p "${target_site_packages}"
printf '%s\n%s\n' "${REPO}" "${source_site_packages}" > "${path_file}"

echo "The shim reuses historical packages read-only from: ${source_site_packages}"
echo "Jason Tunix source is loaded from: ${REPO}"

cd "${REPO}"
PYTHONDONTWRITEBYTECODE=1 "${TARGET_ENV}/bin/python" - <<'PY'
import importlib.metadata
import pathlib
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

repo = pathlib.Path.cwd().resolve()
tunix_source = pathlib.Path(tunix.__file__).resolve()
if repo not in tunix_source.parents:
  raise SystemExit("Tunix does not import from Jason's checkout.")
PY

echo "Jason-owned shim environment is ready: ${TARGET_ENV}"
