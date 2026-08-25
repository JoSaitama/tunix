#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="/home/lhf_hongfu_gmail_com/.venvs/DPO"
LAUNCHER="${REPO_ROOT}/examples/dpo/run_qwen2p5_1p5b_ultrafeedback_from_sft.sh"
SFT_MODEL_PATH="${REPO_ROOT}/runs/sft_qwen2p5_1p5b_ultrafeedback_full_full_20260317_003657/exported_model"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LEGACY_RUN_TS="20260417_013847"
PROFILE="full"
SEEDS="0,1,2"
FAMILIES="vanilla_dpo,random_pair_filtering,reward_based_filtering,self_inf"
FILTERING_LEVELS="5,10"
METHODS=""
EXECUTE=0
NO_LEGACY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-ts)
      RUN_TS="${2:?missing value for --run-ts}"
      shift 2
      ;;
    --run-ts=*)
      RUN_TS="${1#*=}"
      shift
      ;;
    --legacy-run-ts)
      LEGACY_RUN_TS="${2:?missing value for --legacy-run-ts}"
      shift 2
      ;;
    --legacy-run-ts=*)
      LEGACY_RUN_TS="${1#*=}"
      shift
      ;;
    --no-legacy)
      NO_LEGACY=1
      shift
      ;;
    --profile)
      PROFILE="${2:?missing value for --profile}"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --seeds)
      SEEDS="${2:?missing value for --seeds}"
      shift 2
      ;;
    --seeds=*)
      SEEDS="${1#*=}"
      shift
      ;;
    --families)
      FAMILIES="${2:?missing value for --families}"
      shift 2
      ;;
    --families=*)
      FAMILIES="${1#*=}"
      shift
      ;;
    --filtering-levels)
      FILTERING_LEVELS="${2:?missing value for --filtering-levels}"
      shift 2
      ;;
    --filtering-levels=*)
      FILTERING_LEVELS="${1#*=}"
      shift
      ;;
    --methods)
      METHODS="${2:?missing value for --methods}"
      shift 2
      ;;
    --methods=*)
      METHODS="${1#*=}"
      shift
      ;;
    --sft-model-path)
      SFT_MODEL_PATH="${2:?missing value for --sft-model-path}"
      shift 2
      ;;
    --sft-model-path=*)
      SFT_MODEL_PATH="${1#*=}"
      shift
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

if [[ ! -d "${SFT_MODEL_PATH}" ]]; then
  echo "Missing shared SFT exported model: ${SFT_MODEL_PATH}" >&2
  exit 1
fi

METHOD_LIST="$(
python - "${METHODS}" "${FAMILIES}" "${FILTERING_LEVELS}" <<'PY'
import sys

methods_text, families_text, filtering_text = sys.argv[1:]
if methods_text:
  methods = [item.strip() for item in methods_text.split(",") if item.strip()]
else:
  filtering_levels = [item.strip() for item in filtering_text.split(",") if item.strip()]
  methods = []
  for family in [item.strip() for item in families_text.split(",") if item.strip()]:
    if family in ("vanilla_dpo", "self_inf"):
      methods.append(family)
    elif family == "random_pair_filtering":
      for level in filtering_levels:
        methods.append(f"random_pair_filtering_filt{level}")
    elif family == "reward_based_filtering":
      for level in filtering_levels:
        methods.append(f"reward_based_filtering_filt{level}")
    else:
      raise SystemExit(f"Unsupported family: {family}")
print(",".join(methods))
PY
)"

PLAN="$(
python - \
  "${REPO_ROOT}" \
  "${RUN_TS}" \
  "${LEGACY_RUN_TS}" \
  "${NO_LEGACY}" \
  "${SEEDS}" \
  "${METHOD_LIST}" \
  "${PROFILE}" \
  "${SFT_MODEL_PATH}" \
  "${LAUNCHER}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
run_ts = sys.argv[2]
legacy_run_ts = sys.argv[3]
no_legacy = bool(int(sys.argv[4]))
seeds = [int(item) for item in sys.argv[5].split(",") if item]
methods = [item for item in sys.argv[6].split(",") if item]
profile = sys.argv[7]
sft_model_path = sys.argv[8]
launcher = sys.argv[9]

from examples.dpo import qwen2p5_clean_main_table_lib as table_lib
from examples.dpo import qwen2p5_dpo_experiments as exp_lib

for spec in table_lib.expected_run_specs(methods=methods, seeds=seeds):
  logical_variant = spec["logical_variant"]
  seed = spec["seed"]
  seeded_run_root = Path(
      exp_lib.build_run_root(
          repo_root=str(repo_root),
          variant=logical_variant,
          corruption_config="clean",
          ft_mode="lora",
          profile=profile,
          run_ts=run_ts,
          seed=seed,
      )
  )
  seeded_model = seeded_run_root / "exported_model"
  status = "launch"
  source = "seeded"
  if seeded_model.is_dir():
    status = "present"
  elif (
      not no_legacy
      and seed == 0
      and logical_variant in table_lib.LEGACY_VARIANT_MAP
  ):
    legacy_variant = table_lib.LEGACY_VARIANT_MAP[logical_variant]
    legacy_run_root = Path(
        exp_lib.build_run_root(
            repo_root=str(repo_root),
            variant=legacy_variant,
            corruption_config="clean",
            ft_mode="lora",
            profile="full",
            run_ts=legacy_run_ts,
            seed=None,
        )
    )
    if (legacy_run_root / "exported_model").is_dir():
      status = "reuse_legacy_seed0"
      source = "legacy"
  if no_legacy:
    command = (
        f"RUN_TS={run_ts} {launcher} {profile} {logical_variant} {sft_model_path} "
        f"--corruption-config clean --ft-mode lora --run-seed {seed} "
        f"--train-shuffle-seed {seed} --curation-seed 0"
    )
  else:
    command = (
        f"RUN_TS={run_ts} {launcher} {profile} {logical_variant} {sft_model_path} "
        f"--corruption-config clean --ft-mode lora --seed {seed} --curation-seed {seed}"
    )
  print("\t".join([status, source, logical_variant, str(seed), command]))
PY
)"

echo "Clean DPO main-table matrix"
echo "  run_ts: ${RUN_TS}"
echo "  legacy_run_ts: ${LEGACY_RUN_TS}"
echo "  no_legacy: ${NO_LEGACY}"
echo "  profile: ${PROFILE}"
echo "  methods: ${METHOD_LIST}"
echo "  seeds: ${SEEDS}"
echo

LAUNCH_COUNT=0
while IFS=$'\t' read -r STATUS SOURCE METHOD SEED COMMAND; do
  [[ -z "${STATUS}" ]] && continue
  printf '[%s] %s seed=%s (%s)\n' "${STATUS}" "${METHOD}" "${SEED}" "${SOURCE}"
  printf '  %s\n' "${COMMAND}"
  if [[ "${STATUS}" == "launch" ]]; then
    LAUNCH_COUNT=$((LAUNCH_COUNT + 1))
    if [[ "${EXECUTE}" -eq 1 ]]; then
      eval "${COMMAND}"
    fi
  fi
done <<< "${PLAN}"

echo
echo "Total new launches needed: ${LAUNCH_COUNT}"
if [[ "${EXECUTE}" -eq 0 ]]; then
  echo "Dry-run only. Re-run with --execute to launch the missing runs sequentially."
fi
