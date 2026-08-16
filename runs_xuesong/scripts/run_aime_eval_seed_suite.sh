#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
VENV="${VENV:-${REPO}/.venv}"
TPU_NAME="${TPU_NAME:-node-v5p-16-ziao1}"
ZONE="${ZONE:-us-central1-a}"
MODEL_PATH="${MODEL_PATH:-/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet}"
LOO_RUN_ROOT="${LOO_RUN_ROOT:-${REPO}/runs_xuesong/runs/grpo_aime_dtv_selfinf_group_loo_policy_seed0_clean_20260805_095513}"
DTV_RUN_ROOT="${DTV_RUN_ROOT:-${REPO}/runs_xuesong/runs/grpo_aime_dtv_selfinf_group_policy_seed0_clean_20260805_101839}"
BASE_SOURCE_ROOT="${BASE_SOURCE_ROOT:-${REPO}/runs_xuesong/eval_sources/deepseek_r1_distill_qwen_1p5b}"
SUITE_ROOT="${SUITE_ROOT:-${REPO}/runs_xuesong/evals/aime2024_k16_multiseed_fsdp_tp}"
EVAL_MODELS="${EVAL_MODELS:-dtv_loo pretrain}"
TPU_RELEASE_ATTEMPTS="${TPU_RELEASE_ATTEMPTS:-30}"
TPU_RELEASE_INTERVAL_SECONDS="${TPU_RELEASE_INTERVAL_SECONDS:-10}"

SEEDS=(0 5 13 21 42)
if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
fi

mkdir -p "$SUITE_ROOT" "$BASE_SOURCE_ROOT"
STATUS_FILE="${SUITE_ROOT}/status.tsv"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf 'seed\tmodel\tstatus\texit_code\toutput_dir\n' > "$STATUS_FILE"
fi

worker0_is_free() {
  sudo -n true >/dev/null 2>&1 &&
    ! sudo -n fuser /dev/vfio/0 >/dev/null 2>&1 &&
    ! pgrep -f '[e]val_final_checkpoint_metrics|[v]llm' >/dev/null 2>&1
}

worker1_is_free() {
  local result
  result="$(gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=1 \
    --internal-ip \
    --command='if sudo -n true >/dev/null 2>&1 && ! sudo -n fuser /dev/vfio/0 >/dev/null 2>&1 && ! pgrep -f "[e]val_final_checkpoint_metrics|[v]llm" >/dev/null 2>&1; then echo WORKER1_FREE; else echo WORKER1_BUSY; fi' \
    2>/dev/null)" || return 1
  grep -q '^WORKER1_FREE$' <<< "$result"
}

clear_stale_locks() {
  if worker0_is_free; then
    sudo -n rm -f /tmp/libtpu_lockfile
  else
    return 1
  fi
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=1 \
    --internal-ip \
    --command='sudo -n rm -f /tmp/libtpu_lockfile' \
    >/dev/null 2>&1
}

wait_for_tpu_release() {
  local attempt
  for ((attempt = 1; attempt <= TPU_RELEASE_ATTEMPTS; attempt++)); do
    if worker0_is_free && worker1_is_free; then
      clear_stale_locks
      echo "TPU_RELEASED attempt=${attempt}"
      return 0
    fi
    echo "WAITING_FOR_TPU_RELEASE attempt=${attempt}/${TPU_RELEASE_ATTEMPTS}"
    sleep "$TPU_RELEASE_INTERVAL_SECONDS"
  done
  echo "ERROR: TPU workers did not become free within the bounded wait." >&2
  return 1
}

completed_output() {
  local output_dir="$1"
  [[ -f "${output_dir}/logs/launcher.status" ]] &&
    grep -qx 'EXIT_CODE=0' "${output_dir}/logs/launcher.status" &&
    [[ -f "${output_dir}/samples.jsonl" ]] &&
    [[ "$(wc -l < "${output_dir}/samples.jsonl")" -eq 480 ]] &&
    [[ -f "${output_dir}/summary.json" ]] &&
    [[ -f "${output_dir}/.primary_done" ]]
}

run_eval() {
  local seed="$1"
  local label="$2"
  local checkpoint_source="$3"
  local checkpoint_step="$4"
  local run_root="$5"
  local output_dir="${SUITE_ROOT}/seed${seed}/${label}"
  local nohup_log="${SUITE_ROOT}/seed${seed}/${label}.launcher.log"

  mkdir -p "${SUITE_ROOT}/seed${seed}"
  if completed_output "$output_dir"; then
    echo "SKIP_COMPLETED seed=${seed} model=${label} output=${output_dir}"
    printf '%s\t%s\tskipped_complete\t0\t%s\n' \
      "$seed" "$label" "$output_dir" >> "$STATUS_FILE"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "ERROR: incomplete output already exists: ${output_dir}" >&2
    return 1
  fi

  wait_for_tpu_release
  echo "START seed=${seed} model=${label} output=${output_dir}"
  printf '%s\t%s\tstarted\t-\t%s\n' \
    "$seed" "$label" "$output_dir" >> "$STATUS_FILE"

  set +e
  env \
    REPO="$REPO" \
    VENV="$VENV" \
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    EVAL_DATA_PATH="$EVAL_DATA_PATH" \
    MODEL_PATH="$MODEL_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    "${REPO}/runs_xuesong/scripts/run_aime_final_eval.sh" \
      --run-root "$run_root" \
      --output-dir "$output_dir" \
      --checkpoint-source "$checkpoint_source" \
      --checkpoint-step "$checkpoint_step" \
      --eval-seed "$seed" \
      --num-samples 16 \
      --max-generation-steps 8192 \
      --problem-batch-size 16 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 38400 \
    > "$nohup_log" 2>&1
  local status=$?
  set -e

  if [[ "$status" -eq 0 ]] && completed_output "$output_dir"; then
    echo "COMPLETE seed=${seed} model=${label} output=${output_dir}"
    printf '%s\t%s\tcomplete\t0\t%s\n' \
      "$seed" "$label" "$output_dir" >> "$STATUS_FILE"
  else
    echo "FAILED seed=${seed} model=${label} exit_code=${status}" >&2
    printf '%s\t%s\tfailed\t%s\t%s\n' \
      "$seed" "$label" "$status" "$output_dir" >> "$STATUS_FILE"
    return 1
  fi

  wait_for_tpu_release
}

read -r -a SELECTED_MODELS <<< "$EVAL_MODELS"
if [[ "${#SELECTED_MODELS[@]}" -eq 0 ]]; then
  echo "ERROR: EVAL_MODELS must contain at least one model label." >&2
  exit 2
fi

for seed in "${SEEDS[@]}"; do
  for model in "${SELECTED_MODELS[@]}"; do
    case "$model" in
      dtv)
        run_eval "$seed" dtv checkpoint 314 "$DTV_RUN_ROOT"
        ;;
      dtv_loo)
        run_eval "$seed" dtv_loo checkpoint 314 "$LOO_RUN_ROOT"
        ;;
      pretrain)
        run_eval "$seed" pretrain base_model 0 "$BASE_SOURCE_ROOT"
        ;;
      *)
        echo "ERROR: unsupported EVAL_MODELS entry: $model" >&2
        exit 2
        ;;
    esac
  done
done

echo "EVAL_SEED_SUITE_COMPLETE models=${SELECTED_MODELS[*]} seeds=${SEEDS[*]}"
echo "STATUS_FILE=${STATUS_FILE}"
