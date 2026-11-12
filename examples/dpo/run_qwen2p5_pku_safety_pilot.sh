#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash $0 [--smoke] /path/to/sft/exported_model [seed ...]"
  echo "Default seed: 0. Defaults for gradient flip: MAX_FLIP_PERCENT=10 FLIP_FREQUENCY=1."
  echo "Optional environment setting: PYTHON_BIN."
  echo "TRAIN_PAIRS and MAX_STEPS default to 4096 and 16 (pilot), or 1536 and 4 (smoke)."
  echo "Requires one worker with four local v5p TPU devices."
  echo "Output: runs_xuesong/runs and runs_xuesong/logs."
  exit 0
fi
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 [--smoke] /path/to/sft/exported_model [seed ...]" >&2
  exit 2
fi

SFT_MODEL_PATH="$(cd "$1" && pwd -P)"
shift
SEEDS=(0)
if (( $# )); then
  SEEDS=("$@")
fi
seen_seeds=" "
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^(0|[1-9][0-9]*)$ || "$seen_seeds" == *" $seed "* ]]; then
    echo "Expected distinct nonnegative integer seeds, got: $seed" >&2
    exit 2
  fi
  seen_seeds+="$seed "
done
if (( SMOKE )); then
  TRAIN_PAIRS="${TRAIN_PAIRS:-1536}"
  MAX_STEPS="${MAX_STEPS:-4}"
  METHODS=(self_inf_flip)
  RUN_PREFIX=pku_safety_smoke
else
  TRAIN_PAIRS="${TRAIN_PAIRS:-4096}"
  MAX_STEPS="${MAX_STEPS:-16}"
  METHODS=(vanilla_dpo self_inf_flip reward_based_filtering_filt10)
  RUN_PREFIX=pku_safety
fi
MAX_FLIP_PERCENT="${MAX_FLIP_PERCENT:-10}"
FLIP_FREQUENCY="${FLIP_FREQUENCY:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! "$TRAIN_PAIRS" =~ ^[1-9][0-9]*$ || ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ||
      ! "$FLIP_FREQUENCY" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "Use a nonnegative integer for FLIP_FREQUENCY, and positive TRAIN_PAIRS and MAX_STEPS." >&2
  exit 2
fi
if (( TRAIN_PAIRS < MAX_STEPS * 256 )); then
  echo "TRAIN_PAIRS must be at least MAX_STEPS * 256 for a single training pass." >&2
  exit 2
fi
if ! compgen -G "$SFT_MODEL_PATH/*.safetensors" >/dev/null; then
  echo "No exported .safetensors weights in $SFT_MODEL_PATH." >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -f .env || -f my_example/.env ]]; then
  set -a
  if [[ -f .env ]]; then
    source ./.env
  else
    source my_example/.env
  fi
  set +a
fi
# Use the four chips on this VM, including when it is part of a larger slice.
export JAX_PLATFORMS=tpu
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_CHIPS=0,1,2,3 CLOUD_TPU_TASK_ID=0

# The full run needs a temporary model export for the offline evaluator.
export DPO_TIMING_ONLY=0
"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    sys.exit("Tunix requires Python 3.11 or newer.")

import jax

topology = (
    f"backend={jax.default_backend()} process_count={jax.process_count()} "
    f"device_count={jax.device_count()} local_device_count={jax.local_device_count()}"
)
if jax.default_backend() != "tpu" or (
    jax.process_count(), jax.device_count(), jax.local_device_count()
) != (1, 4, 4):
    sys.exit(f"Expected a single worker with four local TPU devices. Got {topology}.")
print(f"Local TPU ready: {topology}")
PY
mkdir -p runs_xuesong/runs runs_xuesong/logs
PILOT_ROOT="$(mktemp -d "$PWD/runs_xuesong/runs/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
PILOT_LOG_ROOT="$PWD/runs_xuesong/logs/${PILOT_ROOT##*/}"
mkdir "$PILOT_LOG_ROOT"
echo "PKU pilot output: $PILOT_ROOT"
echo "PKU pilot logs: $PILOT_LOG_ROOT"
echo "Seeds: ${SEEDS[*]}; methods: ${METHODS[*]}"

for seed in "${SEEDS[@]}"; do
  seed_dir="$PILOT_ROOT/seed$seed"
  seed_log_dir="$PILOT_LOG_ROOT/seed$seed"
  mkdir "$seed_dir" "$seed_log_dir"
  for method in "${METHODS[@]}"; do
    run_dir="$seed_dir/$method"
    train_log="$seed_log_dir/$method.train.log"
    mkdir "$run_dir"
    "$PYTHON_BIN" - "$run_dir" "$method" "$SFT_MODEL_PATH" "$seed" "$TRAIN_PAIRS" "$MAX_STEPS" \
      "$MAX_FLIP_PERCENT" "$FLIP_FREQUENCY" "$SMOKE" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

from examples.dpo import qwen2p5_dpo_experiments

(
    run_dir, method, sft_path, seed, train_pairs, max_steps,
    max_flip_percent, flip_frequency, smoke,
) = sys.argv[1:]
seed, train_pairs, max_steps = int(seed), int(train_pairs), int(max_steps)
try:
    max_flip_percent = float(max_flip_percent)
except ValueError:
    sys.exit("MAX_FLIP_PERCENT must be a number between 0 and 100.")
if not 0 <= max_flip_percent <= 100:
    sys.exit("MAX_FLIP_PERCENT must be a number between 0 and 100.")
smoke = smoke == "1"
config_path = Path(run_dir) / "launch.yaml"
qwen2p5_dpo_experiments.prepare_launch_config(
    config_path="examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml",
    output_path=str(config_path),
    run_root=run_dir,
    run_name=f"{Path(run_dir).parents[1].name}_seed{seed}_{method}",
    profile="full",
    variant=method,
    corruption_config="clean",
    sft_model_path=sft_path,
    curation_threshold="",
    curation_keep_ratio="",
    self_influence_dot_threshold="",
    dpo_seed=seed,
    curation_seed=seed,
)
cfg = OmegaConf.load(config_path)
adapter = "tunix.examples.data.pku_saferlhf_dpo:create_dataset"
cfg.train_data_module = (
    f"{adapter}(split='train', subset='train', "
    f"limit={train_pairs}, shuffle_seed={seed})"
)
eval_pairs = 16 if smoke else 128
cfg.eval_data_module = f"{adapter}(split='train', subset='eval', limit={eval_pairs})"
cfg.dpo_config.max_flip_percent = max_flip_percent
cfg.dpo_config.flip_frequency = int(flip_frequency)
cfg.training_config.max_steps = max_steps
eval_interval = max_steps if smoke else max(1, max_steps // 2)
cfg.training_config.eval_every_n_steps = eval_interval
cfg.training_config.checkpoint_root_directory = None
cfg.training_config.checkpointing_options = None
if smoke:
    cfg.exported_model_output_dir = None
cfg.training_config.metrics_logging_options.flush_every_n_steps = 1
cfg.optimizer_config.warmup_steps = max_steps // 8
cfg.optimizer_config.decay_steps = max_steps
OmegaConf.save(cfg, config_path)
PY
    echo "Training $method (seed $seed)"
    "$PYTHON_BIN" -m tunix.cli.dpo_main "$run_dir/launch.yaml" 2>&1 |
      tee "$train_log"
    if (( SMOKE )); then
      completed_steps="$(awk '/DTV flip step=/ {count++} END {print count+0}' "$train_log")"
      if (( completed_steps != MAX_STEPS )); then
        echo "Smoke expected $MAX_STEPS updates and logged $completed_steps. See $train_log." >&2
        exit 1
      fi
      echo "Clean PKU smoke completed $completed_steps updates. Log: $train_log"
      continue
    fi
    echo "Evaluating $method (seed $seed)"
    "$PYTHON_BIN" -m examples.dpo.eval_qwen2p5_pku_safety "$run_dir" "$SFT_MODEL_PATH" 2>&1 |
      tee "$seed_log_dir/$method.eval.log"
    "$PYTHON_BIN" - "$PILOT_ROOT/summary.csv" "$run_dir" "$seed" "$method" <<'PY'
import csv
import json
from pathlib import Path
import shutil
import sys

summary_path, run_dir, seed, method = sys.argv[1:]
metrics = json.loads((Path(run_dir) / "safety_metrics.json").read_text())
row = dict(method=method, seed=int(seed), **metrics)
with open(summary_path, "a", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(row))
    if output.tell() == 0:
        writer.writeheader()
    writer.writerow(row)
exported_model = Path(run_dir) / "exported_model"
shutil.rmtree(exported_model)
print(f"Saved safety metrics and removed temporary export: {exported_model}")
PY
  done
done

if (( ! SMOKE )); then
  echo "Safety pilot summary: $PILOT_ROOT/summary.csv"
fi
