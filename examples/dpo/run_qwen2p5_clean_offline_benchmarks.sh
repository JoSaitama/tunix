#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/lhf_hongfu_gmail_com/tunix}"
DPO_VENV="${DPO_VENV:-/home/lhf_hongfu_gmail_com/.venvs/DPO}"
OFFLINE_EVAL_VENV="${OFFLINE_EVAL_VENV:-/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OFFLINE}"
RUN_TS_DEFAULT="${RUN_TS:-20260417_013847}"
METHODS_DEFAULT="${METHODS:-vanilla_dpo random_pair_filtering reward_based_filtering self_inf}"
BENCHMARKS_DEFAULT="${BENCHMARKS:-ifeval ifbench rewardbench_v1 rewardbench_v2 livebench_if xstest harmbench wildbench}"
RUN_TS=""
METHODS_OVERRIDE=()
BENCHMARKS_OVERRIDE=()
QUESTION_LIMIT="${QUESTION_LIMIT:-}"
REWARDBENCH_EVAL_LIMIT="${REWARDBENCH_EVAL_LIMIT:-}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
REWARDBENCH_BATCH_SIZE="${REWARDBENCH_BATCH_SIZE:-8}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
SAFETY_JUDGE_BATCH_SIZE="${SAFETY_JUDGE_BATCH_SIZE:-8}"
WILDBENCH_SUBMIT="${WILDBENCH_SUBMIT:-0}"
FORCE="${FORCE:-0}"

if [[ $# -gt 0 ]] && [[ "${1}" != --* ]]; then
  RUN_TS="$1"
  shift
fi

while [[ $# -gt 0 ]] && [[ "${1}" != --* ]]; do
  METHODS_OVERRIDE+=("$1")
  shift
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-ts)
      RUN_TS="$2"
      shift 2
      ;;
    --methods)
      METHODS_OVERRIDE=()
      shift
      while [[ $# -gt 0 ]] && [[ "${1}" != --* ]]; do
        METHODS_OVERRIDE+=("$1")
        shift
      done
      ;;
    --benchmarks)
      BENCHMARKS_OVERRIDE=()
      shift
      while [[ $# -gt 0 ]] && [[ "${1}" != --* ]]; do
        BENCHMARKS_OVERRIDE+=("$1")
        shift
      done
      ;;
    --question-limit)
      QUESTION_LIMIT="$2"
      shift 2
      ;;
    --rewardbench-eval-limit)
      REWARDBENCH_EVAL_LIMIT="$2"
      shift 2
      ;;
    --generation-batch-size)
      GENERATION_BATCH_SIZE="$2"
      shift 2
      ;;
    --rewardbench-batch-size)
      REWARDBENCH_BATCH_SIZE="$2"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --safety-judge-batch-size)
      SAFETY_JUDGE_BATCH_SIZE="$2"
      shift 2
      ;;
    --wildbench-submit)
      WILDBENCH_SUBMIT="1"
      shift
      ;;
    --force)
      FORCE="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

RUN_TS="${RUN_TS:-${RUN_TS_DEFAULT}}"
if [[ ${#METHODS_OVERRIDE[@]} -gt 0 ]]; then
  METHODS="${METHODS_OVERRIDE[*]}"
else
  METHODS="${METHODS_DEFAULT}"
fi
if [[ ${#BENCHMARKS_OVERRIDE[@]} -gt 0 ]]; then
  BENCHMARKS="${BENCHMARKS_OVERRIDE[*]}"
else
  BENCHMARKS="${BENCHMARKS_DEFAULT}"
fi

cd "${REPO_ROOT}"

bash "${REPO_ROOT}/examples/dpo/setup_qwen2p5_clean_offline_eval_env.sh"

# shellcheck disable=SC1090
source "${DPO_VENV}/bin/activate"

ifeval_args=()
rewardbench_args=()
summary_args=()

if [[ -n "${QUESTION_LIMIT}" ]]; then
  ifeval_args+=(--question-limit "${QUESTION_LIMIT}")
fi
if [[ -n "${REWARDBENCH_EVAL_LIMIT}" ]]; then
  rewardbench_args+=(--eval-limit "${REWARDBENCH_EVAL_LIMIT}")
fi
if [[ "${FORCE}" == "1" ]]; then
  ifeval_args+=(--force)
  rewardbench_args+=(--force)
fi

if [[ " ${BENCHMARKS} " == *" ifeval "* ]]; then
  python examples/dpo/eval_qwen2p5_clean_benchmarks.py \
    --run-ts "${RUN_TS}" \
    --methods ${METHODS} \
    --benchmarks ifeval \
    --offline-eval-venv "${OFFLINE_EVAL_VENV}" \
    --generation-batch-size "${GENERATION_BATCH_SIZE}" \
    "${ifeval_args[@]}"
fi

if [[ " ${BENCHMARKS} " == *" ifbench "* ]]; then
  python examples/dpo/eval_qwen2p5_clean_benchmarks.py \
    --run-ts "${RUN_TS}" \
    --methods ${METHODS} \
    --benchmarks ifbench \
    --offline-eval-venv "${OFFLINE_EVAL_VENV}" \
    --generation-batch-size "${GENERATION_BATCH_SIZE}" \
    "${ifeval_args[@]}"
fi

if [[ " ${BENCHMARKS} " == *" rewardbench_v1 "* ]]; then
  python examples/dpo/eval_qwen2p5_rewardbench_v1.py \
    --run-ts "${RUN_TS}" \
    --methods ${METHODS} \
    --batch-size "${REWARDBENCH_BATCH_SIZE}" \
    "${rewardbench_args[@]}"
fi

if [[ " ${BENCHMARKS} " == *" rewardbench_v2 "* ]]; then
  python examples/dpo/eval_qwen2p5_rewardbench_v2.py \
    --run-ts "${RUN_TS}" \
    --methods ${METHODS} \
    --batch-size "${REWARDBENCH_BATCH_SIZE}" \
    "${rewardbench_args[@]}"
fi

if [[ " ${BENCHMARKS} " == *" livebench_if "* ]]; then
  python examples/dpo/eval_qwen2p5_livebench_instruction_following.py \
    --run-ts "${RUN_TS}" \
    --methods ${METHODS} \
    --offline-eval-venv "${OFFLINE_EVAL_VENV}" \
    --generation-batch-size "${GENERATION_BATCH_SIZE}" \
    "${ifeval_args[@]}"
fi

if [[ " ${BENCHMARKS} " == *" xstest "* ]] || [[ " ${BENCHMARKS} " == *" harmbench "* ]]; then
  safety_benchmarks=()
  if [[ " ${BENCHMARKS} " == *" xstest "* ]]; then
    safety_benchmarks+=(xstest)
  fi
  if [[ " ${BENCHMARKS} " == *" harmbench "* ]]; then
    safety_benchmarks+=(harmbench)
  fi
  python examples/dpo/eval_qwen2p5_safety_benchmarks.py \
    --run-ts "${RUN_TS}" \
    --methods ${METHODS} \
    --benchmarks "${safety_benchmarks[@]}" \
    --offline-eval-venv "${OFFLINE_EVAL_VENV}" \
    --generation-batch-size "${GENERATION_BATCH_SIZE}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-batch-size "${SAFETY_JUDGE_BATCH_SIZE}" \
    "${ifeval_args[@]}"
fi

if [[ " ${BENCHMARKS} " == *" wildbench "* ]]; then
  wildbench_args=()
  if [[ "${WILDBENCH_SUBMIT}" == "1" ]]; then
    wildbench_args+=(--submit-batch)
  fi
  python examples/dpo/eval_qwen2p5_wildbench.py \
    --run-ts "${RUN_TS}" \
    --methods ${METHODS} \
    --offline-eval-venv "${OFFLINE_EVAL_VENV}" \
    --judge-model "${JUDGE_MODEL}" \
    "${wildbench_args[@]}" \
    "${ifeval_args[@]}"
fi

python examples/dpo/summarize_qwen2p5_clean_tables.py \
  --run-ts "${RUN_TS}" \
  --columns \
    clean_val_acc_auc \
    clean_test_acc \
    rewardbench_overall \
    ifeval_prompt_strict \
    mt_bench_avg_score \
    alpacaeval2_lc_win_rate \
  "${summary_args[@]}"

echo
echo "Offline clean benchmark pipeline completed for run_ts=${RUN_TS}"
