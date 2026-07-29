#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/my_example/run_fixed_filter.sh" random group "${TUNIX_FILTER_RATIO:?set TUNIX_FILTER_RATIO}" "$@"
