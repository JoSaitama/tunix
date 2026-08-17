#!/usr/bin/env bash
set -euo pipefail

# Read-only disk inventory. This script never deletes or modifies data.

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs_xuesong}"
MODEL_ROOT="${MODEL_ROOT:-/home/lhf_hongfu_gmail_com/models}"

echo "Filesystem free space"
df -h "${REPO}" "${MODEL_ROOT}" 2>/dev/null || df -h

echo
echo "Top-level repository usage"
du -x -h -d 1 "${REPO}" 2>/dev/null | sort -h

echo
echo "Experiment storage usage"
if [[ -d "${RUNS_ROOT}" ]]; then
  du -x -h -d 2 "${RUNS_ROOT}" 2>/dev/null | sort -h | tail -80
fi

echo
echo "Model storage usage"
if [[ -d "${MODEL_ROOT}" ]]; then
  du -x -h -d 2 "${MODEL_ROOT}" 2>/dev/null | sort -h | tail -40
fi

echo
echo "Files larger than 500 MiB under experiment storage"
if [[ -d "${RUNS_ROOT}" ]]; then
  find "${RUNS_ROOT}" -xdev -type f -size +500M \
    -printf '%s\t%p\n' 2>/dev/null \
    | sort -n \
    | tail -60 \
    | awk -F '\t' '{printf "%.2f GiB\t%s\n", $1/1073741824, $2}'
fi

echo
echo "No files were deleted. Review exact paths before cleanup."
