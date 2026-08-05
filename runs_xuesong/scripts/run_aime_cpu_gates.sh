#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/jason_chia925_gmail_com/Project/tunix}"
VENV="${VENV:-${REPO}/.venv}"
PYTHON="${VENV}/bin/python"

cd "$REPO"
test -x "$PYTHON"

run_gate() {
  local name="$1"
  shift
  echo "=== CPU GATE: ${name} ==="
  timeout --signal=TERM --kill-after=10s 600s \
    env JAX_PLATFORMS=cpu \
    "$PYTHON" -m pytest -q -x "$@"
}

run_gate data_and_eval \
  tests/cli/recipes/deepscaler_data_test.py \
  tests/cli/recipes/deepscaler_eval_test.py

run_gate weighted_accumulation_and_lr \
  tests/sft/weighted_gradient_accumulation_test.py

run_gate policy_selection_and_gradients \
  tests/rl/memory_bounded_curation_test.py \
  tests/rl/self_inf_loo_trainer_test.py \
  tests/rl/self_inf_trainer_test.py

run_gate trainer_checkpointing \
  tests/sft/peft_trainer_test.py \
  -k 'gradient_accumulation or periodic_checkpoint or checkpointing or injected_params'

run_gate rl_routing \
  tests/rl/rl_cluster_test.py \
  tests/cli/grpo_main_test.py \
  -k 'not sglang_jax'

run_gate agentic_loss_and_checkpointing \
  tests/rl/agentic/agentic_grpo_learner_test.py \
  tests/rl/agentic/agentic_rl_learner_test.py \
  -k 'grpo_loss_fn or checkpoint or restore or train'

echo "ALL_AIME_CPU_GATES_PASSED"
