# 2026-10-03: 1.5B restore smoke and evaluation teardown

The two-step baseline completed with both worker programs reporting status 0.
The restore evaluator restored actor checkpoint 2, synchronized its weights to
vLLM and generated exactly four records (two questions, two sample slots).
`summary.json`, `samples.jsonl` and `.primary_done` were written. All four
responses reached the 512-token integration cap; these are not quality results.
This verifies actor restoration, not resumed optimizer/scheduler training.

Worker 1 reported `HOST=...-w-1 STATUS=0`. Worker 0 retained scheduler child
PIDs 815261 and 815274, both with PPID 1 and start times matching the vLLM
two-worker scheduler startup. The evaluator parent had exited. Its hard
`os._exit(0)` bypassed the sampler's registered atexit stop callback. Forked
scheduler children inherited output descriptors and held the local tee pipe
open, leaving the launcher waiting without `logs/launcher.status`.

The fix affects only the standalone evaluator and a new lifecycle helper:

- Stop the sampler before the existing global completion barrier and hard exit.
- Bound sampler.stop to 60 seconds. Reap only multiprocessing children created
  during this sampler's lifetime, excluding preexisting children. After graceful
  shutdown, join remaining children, then terminate/kill them if necessary.
- On evaluation/cleanup failure, release the secondary once, print the original
  traceback and hard-exit nonzero. Never retry a failed completion barrier.
- Record sampler shutdown, child cleanup, barrier and hard-exit phases plus PIDs.

Training launch/communication, native JAX ranks, vLLM driver/sampler, weight
transfer, checkpoint implementation, LOO scoring and decoding settings are
unchanged. Sixteen protected training files remain identical to c168b7f4.
The existing uncommitted develop.md changes were not edited or staged.

Local verification: 14 lifecycle tests, including a real fork/pipe reproduction
and a real cleanup/EOF regression, two historical status tests, nine deployment
tests and five launcher tests passed (30 total). The launcher tests required
execution outside the local sandbox because process substitution accesses
/dev/fd. Dependency-backed pytest and TPU integration still run on the server.

## 1. Reclaim only the two confirmed orphan workers, on worker 0

This validates ownership, PPID and the exact evaluator/output directory before
sending TERM. It preserves checkpoints/results and does not scan or signal
other TPU jobs. Run in the diagnostic terminal, leaving the original launcher
terminal open so its return status can be observed.

```bash
export REPO=/home/jason_chia925_gmail_com/Project_7B/tunix
export EVAL_DIR="$REPO/runs_xuesong/runs/grpo_aime_1p5b_smoke_baseline_seed0_clean_20261003_023616_780668/eval/restore_smoke_20261003_030850"

JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$REPO/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import signal

targets = []
for pid in (815261, 815274):
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        print(f"PID {pid} already exited")
        continue
    args = proc.joinpath("cmdline").read_bytes().decode().split("\0")
    status = proc.joinpath("status").read_text().splitlines()
    ppid = next(line.split()[1] for line in status if line.startswith("PPid:"))
    if (proc.stat().st_uid != os.getuid() or ppid != "1"
        or os.environ["EVAL_DIR"] not in args
        or not any(arg.endswith("/eval_final_checkpoint_metrics.py") for arg in args)):
        raise SystemExit(f"Refusing to signal PID {pid}: identity no longer matches")
    targets.append(pid)
for pid in targets:
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent TERM to confirmed evaluator orphan {pid}")
    except ProcessLookupError:
        print(f"PID {pid} already exited")
PY

sleep 2
ps -o pid,ppid,stat,args -p 809719,815261,815274 || true
sudo fuser -v /dev/vfio/0 || true
gcloud alpha compute tpus tpu-vm ssh node-v5p-16-ziao1 \
  --zone=us-central1-a --worker=1 \
  --command='sudo fuser -v /dev/vfio/0 || true'
```

If either orphan remains, inspect its state before proceeding. Do not signal
additional PIDs. An empty fuser result means no device owner was shown at that
instant. Wait for the old launcher terminal to return before deployment/retest.
Any zero status after this manual cleanup is not proof of automatic teardown;
the fresh smoke below is the required gate.

## 2. Commit and push from the Mac

The fix is initially an uncommitted local change. Stage only these four files;
leave the preexisting develop.md notes outside this commit.

```bash
cd '/Users/jason/Documents/1. Education/9. Codex/DTV_GRPO_AIME/tunix'
test "$(git branch --show-current)" = codex/aime-deepseek-7b
git add \
  examples/deepscaler/eval_final_checkpoint_metrics.py \
  examples/deepscaler/eval_sampler_lifecycle.py \
  tests/cli/recipes/aime_eval_lifecycle_test.py \
  runs_xuesong/AIME_EVAL_TEARDOWN.md
git diff --cached --check
git diff --cached --stat
git commit -m "Reap standalone evaluation sampler workers before hard exit"
git push origin codex/aime-deepseek-7b
git rev-parse HEAD
```

## 3. Pull on worker 0, then redeploy worker 1

Both workers must be idle. Worker 1 uses an archived source deployment, so do
not run git pull there. The existing deployment helper discovers/validates the
target and retains the existing shim dependencies.

```bash
export REPO=/home/jason_chia925_gmail_com/Project_7B/tunix
export OLD_REPO=/home/jason_chia925_gmail_com/Project/tunix
export VENV="$REPO/.venv"
export TPU_NAME=node-v5p-16-ziao1 ZONE=us-central1-a REMOTE_WORKER_INDEX=1
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
unset PYTHON_BIN

(
  set -euo pipefail
  cd "$REPO"
  test "$(git branch --show-current)" = codex/aime-deepseek-7b
  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short
    echo 'Update stopped: the new checkout has local changes.' >&2
    exit 1
  fi
  git pull --ff-only origin codex/aime-deepseek-7b
  JAX_PLATFORMS=cpu "$VENV/bin/python" -m pytest -q \
    tests/cli/recipes/aime_eval_lifecycle_test.py \
    tests/cli/recipes/deepscaler_eval_test.py
  bash "$REPO/runs_xuesong/scripts/deploy_aime_7b_worker.sh" --phase all
)
```

## 4. Rerun only checkpoint-2 restore/generation, not training

Use a fresh output directory so old success artifacts cannot satisfy this gate.
This repeats the original TP=2, DP=2, async-vLLM smoke settings.

```bash
export SMOKE_RUN="$REPO/runs_xuesong/runs/grpo_aime_1p5b_smoke_baseline_seed0_clean_20261003_023616_780668"
export MODEL_PATH=/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b
export TOKENIZER_PATH="$MODEL_PATH"
export EVAL_DATA_PATH=/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet
export EVAL_DIR="$SMOKE_RUN/eval/restore_smoke_teardown_$(date -u +%Y%m%d_%H%M%S)_$$"
source "$REPO/runs_xuesong/cache/deploy/$(git -C "$REPO" rev-parse HEAD)/target.env"
unset JAX_PLATFORMS PYTHON_BIN

(
  set -euo pipefail
  cd "$REPO"
  bash "$REPO/runs_xuesong/scripts/run_aime_final_eval.sh" \
    --run-root "$SMOKE_RUN" --checkpoint-step 2 \
    --model-config deepseek_r1_distill_qwen_1p5b \
    --protocol-name smoke_1p5b_restore --output-dir "$EVAL_DIR" \
    --limit 2 --num-samples 2 --max-generation-steps 512 \
    --problem-batch-size 2 --max-num-seqs 4 \
    --tensor-parallel-size 2 --data-parallel-size 2 \
    --vllm-hbm-utilization 0.4 --vllm-server-mode --vllm-async-scheduling
  cat "$EVAL_DIR/logs/launcher.status"
  test "$(wc -l < "$EVAL_DIR/samples.jsonl")" -eq 4
)
```

Acceptance: launcher returns automatically; LOCAL_STATUS=0, REMOTE_STATUS=0,
EXIT_CODE=0, SAMPLE_COUNT=4; local log shows sampler_stop_complete,
children_reaped remaining=0, barrier_complete and hard_exit exit_code=0.
Confirm the same evaluator has no remaining processes/device owners before
starting the next TPU job. A shutdown timeout is a failed gate, even if result
files are present. Full 1.5B optimizer resume and 7B full/LOO gates remain separate.
