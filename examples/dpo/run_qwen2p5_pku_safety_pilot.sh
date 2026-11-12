#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash $0 [--smoke | --resume RUN_DIR] /path/to/sft/exported_model [seed ...]"
  echo "Default seed: 0. Defaults for gradient flip: MAX_FLIP_PERCENT=10 FLIP_FREQUENCY=1."
  echo "Optional environment setting: PYTHON_BIN."
  echo "TRAIN_PAIRS and MAX_STEPS default to 4096 and 16 (pilot), or 1536 and 4 (smoke)."
  echo "Requires one worker with four local v5p TPU devices."
  echo "Output: runs_xuesong/runs and runs_xuesong/logs."
  exit 0
fi
SMOKE=0
RESUME_ROOT=""
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
elif [[ "${1:-}" == "--resume" && $# -ge 3 ]]; then
  RESUME_ROOT="$(cd "$2" && pwd -P)"
  shift 2
fi
if [[ $# -lt 1 || "$1" == --* ]]; then
  echo "Usage: bash $0 [--smoke | --resume RUN_DIR] /path/to/sft/exported_model [seed ...]" >&2
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
RUNS_ROOT="$(cd runs_xuesong/runs && pwd -P)"
if [[ -n "$RESUME_ROOT" ]]; then
  if [[ "$(dirname "$RESUME_ROOT")" != "$RUNS_ROOT" ||
        "${RESUME_ROOT##*/}" != pku_safety_* ||
        "${RESUME_ROOT##*/}" == pku_safety_smoke_* ]]; then
    echo "Resume expects a formal PKU run directory under runs_xuesong/runs." >&2
    exit 2
  fi
  PILOT_ROOT="$RESUME_ROOT"
else
  PILOT_ROOT="$(mktemp -d "$RUNS_ROOT/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
fi
if ! mkdir "$PILOT_ROOT/.launcher.lock"; then
  echo "Cannot lock PKU run directory: $PILOT_ROOT" >&2
  exit 1
fi
trap 'rmdir "$PILOT_ROOT/.launcher.lock"' EXIT
PILOT_LOG_ROOT="$PWD/runs_xuesong/logs/${PILOT_ROOT##*/}"
mkdir -p "$PILOT_LOG_ROOT"
echo "PKU pilot output: $PILOT_ROOT"
echo "PKU pilot logs: $PILOT_LOG_ROOT"
echo "Seeds: ${SEEDS[*]}; methods: ${METHODS[*]}"

for seed in "${SEEDS[@]}"; do
  seed_dir="$PILOT_ROOT/seed$seed"
  seed_log_dir="$PILOT_LOG_ROOT/seed$seed"
  mkdir -p "$seed_dir" "$seed_log_dir"
  for method in "${METHODS[@]}"; do
    run_dir="$seed_dir/$method"
    train_log="$seed_log_dir/$method.train.log"
    eval_log="$seed_log_dir/$method.eval.log"
    mkdir -p "$run_dir"
    train_required=1
    if [[ -f "$run_dir/safety_metrics.json" ||
          ( -s "$run_dir/exported_model/model.safetensors" &&
            ! -f "$run_dir/train.pending" &&
            ( -f "$run_dir/export.ready" || -f "$eval_log" ) ) ]]; then
      train_required=0
    fi
    "$PYTHON_BIN" - "$run_dir" "$method" "$SFT_MODEL_PATH" "$seed" "$TRAIN_PAIRS" "$MAX_STEPS" \
      "$MAX_FLIP_PERCENT" "$FLIP_FREQUENCY" "$SMOKE" "$train_required" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

from examples.dpo import qwen2p5_dpo_experiments

(
    run_dir, method, sft_path, seed, train_pairs, max_steps,
    max_flip_percent, flip_frequency, smoke, train_required,
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
adapter = "tunix.examples.data.pku_saferlhf_dpo:create_dataset"
train_data_module = (
    f"{adapter}(split='train', subset='train', "
    f"limit={train_pairs}, shuffle_seed={seed})"
)
if train_required == "0":
    cfg = OmegaConf.load(config_path)
    if (
        Path(cfg.actor_model_config.model_path).resolve() != Path(sft_path).resolve()
        or Path(cfg.reference_model_config.model_path).resolve() != Path(sft_path).resolve()
        or cfg.train_data_module != train_data_module
        or cfg.training_config.max_steps != max_steps
        or cfg.dpo_config.max_flip_percent != max_flip_percent
        or cfg.dpo_config.flip_frequency != int(flip_frequency)
    ):
        sys.exit(f"Resume settings differ from {config_path}.")
    sys.exit(0)

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
cfg.train_data_module = train_data_module
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
    if (( train_required )); then
      echo "Training $method (seed $seed)"
      if (( ! SMOKE )); then
        touch "$run_dir/train.pending"
      fi
      "$PYTHON_BIN" -m tunix.cli.dpo_main "$run_dir/launch.yaml" 2>&1 |
        tee -a "$train_log"
      if (( ! SMOKE )); then
        mv "$run_dir/train.pending" "$run_dir/export.ready"
      fi
    fi
    if (( SMOKE )); then
      completed_steps="$(awk '/DTV flip step=/ {count++} END {print count+0}' "$train_log")"
      if (( completed_steps != MAX_STEPS )); then
        echo "Smoke expected $MAX_STEPS updates and logged $completed_steps. See $train_log." >&2
        exit 1
      fi
      echo "Clean PKU smoke completed $completed_steps updates. Log: $train_log"
      continue
    fi
    if [[ -f "$run_dir/safety_metrics.json" ]]; then
      echo "Using saved evaluation for $method (seed $seed)"
    else
      echo "Evaluating $method (seed $seed)"
      "$PYTHON_BIN" -m examples.dpo.eval_qwen2p5_pku_safety "$run_dir" "$SFT_MODEL_PATH" 2>&1 |
        tee -a "$eval_log"
    fi
    "$PYTHON_BIN" - "$PILOT_ROOT/summary.csv" "$run_dir" "$seed" "$method" <<'PY'
import csv
import json
from pathlib import Path
import shutil
import sys

summary_path, run_dir, seed, method = sys.argv[1:]
metrics = json.loads((Path(run_dir) / "safety_metrics.json").read_text())
summary_path = Path(summary_path)
rows = {}
if summary_path.exists():
    with summary_path.open(newline="") as saved:
        rows = {(row["seed"], row["method"]): row for row in csv.DictReader(saved)}
row = dict(method=method, seed=seed, **metrics)
rows[seed, method] = row
temporary_summary = summary_path.with_suffix(".csv.tmp")
with temporary_summary.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(row))
    writer.writeheader()
    writer.writerows(rows.values())
temporary_summary.replace(summary_path)
exported_model = Path(run_dir) / "exported_model"
if exported_model.exists():
    shutil.rmtree(exported_model)
    print(f"Saved safety metrics and removed temporary export: {exported_model}")
PY
  done
done

if (( ! SMOKE )); then
  echo "Safety pilot summary: $PILOT_ROOT/summary.csv"
fi
