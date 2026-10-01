# 7B review and deployment

This procedure retains the established dual-worker communication implementation.
It does not rebuild the historical TPU Python/vLLM environment. Execute server
commands manually on the directly connected worker while both TPU workers are
idle; do not update an active training checkout.

## Review evidence

Base commit: `c168b7f4` on `for_GRPO_vLLM_aime`.
Development branch: `codex/aime-deepseek-7b`.

Sixteen protected runtime files were compared byte-for-byte with the base:

- `runs_xuesong/scripts/run_official_like_dual_worker.sh`
- `runs_xuesong/scripts/dual_worker_status.sh`
- `examples/deepscaler/run_deepscaler_disagg_v5p16_1epoch.sh`
- `tunix/cli/grpo_main_distributed.py`, `tunix/cli/grpo_main.py`, `tunix/cli/config.py`
- `tunix/cli/base_agentic_config.yaml`
- `tunix/rl/rollout/vllm_rollout.py`
- `tunix/generate/vllm_sampler.py`, `tunix/generate/vllm_async_driver.py`
- `tunix/rl/reshard.py`, `tunix/rl/rl_cluster.py`
- `tunix/rl/agentic/agentic_grpo_learner.py`
- `tunix/rl/agentic/agentic_rl_learner.py`
- `tunix/rl/self_inf_loo_policy_trainer.py`, `tunix/sft/peft_trainer.py`

Consequently, the extension preserves native JAX rank discovery, rank-ordered
host mapping, two-endpoint validation, remote command/environment propagation,
the RPC port, distributed rollout owner selection, weight-sync barriers, KV
allocation reuse, queue/update behavior and program-status precedence over SSH
status. The evaluation command builder now selects an explicit architecture;
its SSH dispatch, result collection, sample-cardinality and exit-status checks
are unchanged.

Five launcher boundary tests and the two historical remote-status tests passed
locally. Production-schema inheritance and dependency-backed distributed tests
remain server checks. These are source/CPU checks; 7B changes memory/compile and
transfer costs and still requires real TPU gates.

All shell examples were syntax-checked. The exact remote installation command
formatter was also exercised against a temporary local archive: archive and
source checksums passed, and the deployed commit marker matched. This checks
command construction and extraction, not real gcloud connectivity.

## Push from the Mac

These commands stage only the extension and its tests/documentation. The
pre-existing uncommitted `develop.md` notes are left intact outside this commit.

```bash
cd '/Users/jason/Documents/1. Education/9. Codex/DTV_GRPO_AIME/tunix'
git branch --show-current
git add \
  tunix/models/qwen2/model.py \
  examples/deepscaler/eval_final_checkpoint_metrics.py \
  runs_xuesong/scripts/run_aime_seeded_full.sh \
  runs_xuesong/scripts/run_aime_final_eval.sh \
  runs_xuesong/scripts/run_aime_7b_full.sh \
  runs_xuesong/scripts/run_aime_7b_eval.sh \
  tests/models/automodel_test.py \
  tests/models/qwen2/config_test.py \
  tests/cli/config_test.py \
  tests/cli/aime_launchers_test.py \
  runs_xuesong/AIME_7B.md \
  runs_xuesong/AIME_7B_DEPLOY.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add DeepSeek 7B AIME profiles preserving dual-worker runtime"
git push -u origin codex/aime-deepseek-7b
git rev-parse HEAD
```

## Pull on the directly connected server worker

Use the actual TPU name and zone. No fixed physical-IP or service-account name
is assumed. Worker 1 remains a source deployment directory, not a Git checkout.
First inspect device owners/processes on both workers. The commands below do
not stop processes or remove locks.

```bash
export REPO=/home/jason_chia925_gmail_com/Project/tunix
export TPU_NAME='<actual TPU name>'
export ZONE='<actual zone>'
export REMOTE_WORKER_INDEX=1
cd "$REPO"
sudo fuser -v /dev/vfio/0
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --worker="$REMOTE_WORKER_INDEX" --internal-ip \
  --command='hostname; sudo fuser -v /dev/vfio/0'
```

An empty `fuser` result normally has a nonzero status. Confirm both workers are
idle before entering this fail-fast update block. If the server tree is dirty,
preserve/inspect those changes before continuing; do not reset it.

```bash
(
  set -euo pipefail
  cd "$REPO"
  test -z "$(git status --porcelain)"
  BRANCH=codex/aime-deepseek-7b
  # Explicit refspec also works for the historical single-branch clone.
  git fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git switch "$BRANCH"
  else
    git switch --track -c "$BRANCH" "origin/$BRANCH"
  fi
  git pull --ff-only origin "$BRANCH"
  git rev-parse HEAD
)
```

Run the focused CPU checks on this worker before deploying to worker 1:

```bash
(
  set -euo pipefail
  cd "$REPO"
  JAX_PLATFORMS=cpu "$REPO/.venv/bin/python" -m pytest -q \
    tests/cli/aime_launchers_test.py tests/models/qwen2/config_test.py \
    tests/scripts/dual_worker_status_test.py tests/cli/grpo_main_distributed_test.py \
    tests/cli/config_test.py tests/cli/recipes/deepscaler_eval_test.py
)
```

These checks do not allocate a real 7B model or start TPU training. Do not run
the whole `vllm_sampler_test.py` merely to check cache reuse: its shared setup
downloads a gated Llama checkpoint. The unchanged KV implementation and real
weight-sync logs can be audited without introducing that download.

## Deploy the exact committed source to worker 1

Run only after the CPU checks pass. A persistent archive can be retransmitted
from byte zero; extraction happens only after its SHA-256 matches. This avoids
the historical `git archive | gcloud ssh ... tar` retry corruption. If a copy
fails, this block stops before extraction; retain the files for a bounded retry
or a resumable transfer using the actual SSH endpoint.

The existing remote repository directory and `.venv` must already have the
working ownership/compatibility layout established for 1.5B. This procedure
does not recreate symlinks or copy models/data.

```bash
(
  set -euo pipefail
  cd "$REPO"
  test -z "$(git status --porcelain)"
  DEPLOY_SHA="$(git rev-parse HEAD)"
  DEPLOY_DIR="/tmp/tunix-7b-deploy-${DEPLOY_SHA}"
  mkdir -p "$DEPLOY_DIR"
  git archive --format=tar.gz --output="$DEPLOY_DIR/source.tar.gz" HEAD
  git ls-files -z | xargs -0 sha256sum > "$DEPLOY_DIR/source.sha256"
  (
    cd "$DEPLOY_DIR"
    sha256sum source.tar.gz > archive.sha256
  )

  printf -v PREPARE_COMMAND 'set -e\ntest -d %q\nmkdir -p %q' \
    "$REPO" "$DEPLOY_DIR"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --worker="$REMOTE_WORKER_INDEX" --internal-ip \
    --command="bash -lc $(printf '%q' "$PREPARE_COMMAND")"

  for artifact in source.tar.gz archive.sha256 source.sha256; do
    gcloud alpha compute tpus tpu-vm scp \
      "$DEPLOY_DIR/$artifact" "$TPU_NAME:$DEPLOY_DIR/$artifact" \
      --zone="$ZONE" --worker="$REMOTE_WORKER_INDEX" --internal-ip
  done

  printf -v INSTALL_COMMAND \
    'set -e\ncd %q\nsha256sum -c archive.sha256\ntar -xzf source.tar.gz -C %q\ncd %q\nsha256sum -c %q\nprintf "%%s\\n" %q > .deployed_git_head' \
    "$DEPLOY_DIR" "$REPO" "$REPO" "$DEPLOY_DIR/source.sha256" "$DEPLOY_SHA"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --worker="$REMOTE_WORKER_INDEX" --internal-ip \
    --command="bash -lc $(printf '%q' "$INSTALL_COMMAND")"
)
```

Do not launch until every checksum is `OK`, the remote interpreter is usable,
and its source import resolves to the deployed repository. Then run the same
focused CPU checks on worker 1 with `cd "$REPO"` and `JAX_PLATFORMS=cpu` before
the unchanged 1.5B regression gate and the 7B integration gates.

## Post-gate efficiency analysis

Keep RPC/rank/teardown behavior fixed for the first successful 7B run. Collect
compilation time separately from steady steps, actor compute/scoring time,
rollout time, weight-sync phase durations, queue waiting, checkpoint time and
HBM on both workers. Summed overlapping durations are not the global wall time.

The historical staged 1.5B DTV gate spent about 97% of its step in actor
microbatches (`develop.md`, 2026-08-02), so do not assume network communication
is the primary bottleneck. Measure the current exact-vmap 7B path first. Based
on those measurements, consider rollout concurrency/batching, JAX cache clear
and recompilation behavior, synchronization transfer/resharding, and exact
LOO gradient memory/compute. Change one factor at a time and preserve the
same sampling/update and checkpoint-restore contract.
