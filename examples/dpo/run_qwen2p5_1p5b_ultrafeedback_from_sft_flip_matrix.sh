#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="${REPO_ROOT}/examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh"

PROFILE="full"
FT_MODE="lora"
PRINT_ONLY="false"
METHODS_CSV="vanilla_dpo,random_pair_filtering,reward_based_filtering,self_inf"
DATASETS_CSV="clean,global_flip20,global_flip40"
SFT_MODEL_PATH=""
declare -a USER_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --ft-mode)
      FT_MODE="${2:-}"
      shift 2
      ;;
    --ft-mode=*)
      FT_MODE="${1#*=}"
      shift
      ;;
    --methods)
      METHODS_CSV="${2:-}"
      shift 2
      ;;
    --methods=*)
      METHODS_CSV="${1#*=}"
      shift
      ;;
    --datasets)
      DATASETS_CSV="${2:-}"
      shift 2
      ;;
    --datasets=*)
      DATASETS_CSV="${1#*=}"
      shift
      ;;
    --print-only)
      PRINT_ONLY="true"
      shift
      ;;
    *)
      if [[ -z "${SFT_MODEL_PATH}" && "$1" != *=* ]]; then
        SFT_MODEL_PATH="$1"
      else
        USER_OVERRIDES+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ "${PROFILE}" != "full" && "${PROFILE}" != "smoke" ]]; then
  echo "Unsupported --profile: ${PROFILE}" >&2
  exit 1
fi

if [[ "${FT_MODE}" != "full" && "${FT_MODE}" != "lora" ]]; then
  echo "Unsupported --ft-mode: ${FT_MODE}" >&2
  exit 1
fi

if [[ -z "${SFT_MODEL_PATH}" || ! -d "${SFT_MODEL_PATH}" ]]; then
  echo "Usage: $0 /path/to/sft_exported_model [--profile full|smoke] [--ft-mode full|lora] [--methods vanilla_dpo,random_pair_filtering,reward_based_filtering,self_inf] [--datasets clean,global_flip20,global_flip40] [--print-only] [overrides...]" >&2
  exit 1
fi

IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"

declare -a COMMANDS=()
for method in "${METHODS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    command=(
      "${LAUNCHER}"
      "${PROFILE}"
      "${method}"
      "${SFT_MODEL_PATH}"
      "--corruption-config"
      "${dataset}"
      "--ft-mode"
      "${FT_MODE}"
    )
    for override in "${USER_OVERRIDES[@]}"; do
      command+=("${override}")
    done
    COMMANDS+=("$(printf '%q ' "${command[@]}")")
  done
done

printf 'Planned %d runs\n' "${#COMMANDS[@]}"
for idx in "${!COMMANDS[@]}"; do
  printf '[%02d/%02d] %s\n' "$((idx + 1))" "${#COMMANDS[@]}" "${COMMANDS[$idx]}"
done

if [[ "${PRINT_ONLY}" == "true" ]]; then
  exit 0
fi

for command in "${COMMANDS[@]}"; do
  eval "${command}"
done
