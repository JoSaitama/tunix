#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Back-compat: keep the original entrypoint, but split each DBC variant into its
# own script for clarity.
#
# Variants:
#   - ./my_example/run_dbc_self_inf_batch.sh  (default)
#   - ./my_example/run_dbc_self_inf_group.sh
#   - ./my_example/run_dbc_outlier_l2.sh

exec "${SCRIPT_DIR}/run_dbc_self_inf_batch.sh" "$@"
