#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash $0 /path/to/sft/exported_model [seed]"
  echo "Optional environment settings: PYTHON_BIN, TRAIN_PAIRS (4096), MAX_STEPS (16)."
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash $0 /path/to/sft/exported_model [seed]" >&2
  exit 2
fi

SFT_MODEL_PATH="$(cd "$1" && pwd -P)"
SEED="${2:-0}"
TRAIN_PAIRS="${TRAIN_PAIRS:-4096}"
MAX_STEPS="${MAX_STEPS:-16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ||
      ! "$TRAIN_PAIRS" =~ ^[1-9][0-9]*$ || ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Use a nonnegative integer seed and positive TRAIN_PAIRS and MAX_STEPS." >&2
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
# Keep checkpointing and export enabled for the train-then-eval workflow.
export DPO_TIMING_ONLY=0
"$PYTHON_BIN" -c 'import sys; sys.version_info >= (3, 11) or sys.exit("Tunix requires Python 3.11 or newer.")'
mkdir -p runs
PILOT_ROOT="$(mktemp -d "$PWD/runs/pku_safety_seed${SEED}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "PKU pilot output: $PILOT_ROOT"

METHODS=(vanilla_dpo self_inf_flip10)
for method in "${METHODS[@]}"; do
  run_dir="$PILOT_ROOT/$method"
  mkdir "$run_dir"
  "$PYTHON_BIN" - "$run_dir" "$method" "$SFT_MODEL_PATH" "$SEED" "$TRAIN_PAIRS" "$MAX_STEPS" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

from examples.dpo import qwen2p5_dpo_experiments

run_dir, method, sft_path, seed, train_pairs, max_steps = sys.argv[1:]
seed, train_pairs, max_steps = int(seed), int(train_pairs), int(max_steps)
config_path = Path(run_dir) / "launch.yaml"
qwen2p5_dpo_experiments.prepare_launch_config(
    config_path="examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml",
    output_path=str(config_path),
    run_root=run_dir,
    run_name=f"{Path(run_dir).parent.name}_{method}",
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
cfg.eval_data_module = f"{adapter}(split='train', subset='eval', limit=128)"
cfg.training_config.max_steps = max_steps
eval_interval = max(1, max_steps // 2)
cfg.training_config.eval_every_n_steps = eval_interval
cfg.training_config.checkpointing_options.save_interval_steps = eval_interval
cfg.training_config.metrics_logging_options.flush_every_n_steps = 1
cfg.optimizer_config.warmup_steps = max_steps // 8
cfg.optimizer_config.decay_steps = max_steps
OmegaConf.save(cfg, config_path)
PY
  echo "Training $method (seed $SEED)"
  "$PYTHON_BIN" -m tunix.cli.dpo_main "$run_dir/launch.yaml" 2>&1 |
    tee "$run_dir/train.log"
  echo "Evaluating $method (seed $SEED)"
  "$PYTHON_BIN" -m examples.dpo.eval_qwen2p5_pku_safety "$run_dir" "$SFT_MODEL_PATH" 2>&1 |
    tee "$run_dir/eval.log"
done

"$PYTHON_BIN" - "$PILOT_ROOT" "$SEED" "${METHODS[@]}" <<'PY'
import csv
import json
from pathlib import Path
import sys

root, seed, *methods = sys.argv[1:]
root = Path(root)
rows = [
    dict(method=method, seed=int(seed), **json.loads(
        (root / method / "safety_metrics.json").read_text()
    ))
    for method in methods
]
with (root / "summary.csv").open("w", newline="") as output:
  writer = csv.DictWriter(output, fieldnames=list(rows[0]))
  writer.writeheader()
  writer.writerows(rows)
print(f"Safety pilot summary: {root / 'summary.csv'}")
PY
