# 7B review and deployment

This procedure retains the established dual-worker communication implementation.
It does not rebuild the historical TPU Python/vLLM environment. The 7B checkout,
shim environment, runs, logs and caches live under
`/home/jason_chia925_gmail_com/Project_7B/tunix` on both workers. The existing
`Project/tunix` checkout is read only throughout this procedure. Dependency
packages are reused read only; large model/data assets may be reused by explicit
paths. This is source/output isolation, not exclusive use of the shared TPU.
Execute server commands manually. Both workers must be idle for real TPU gates.

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

All shell examples and the generated remote setup body were syntax-checked.
Temporary local fixtures also verified cloning while preserving a dirty old
checkout, archive/source checksums and the deployed commit marker. A fixture
with an old shim pointing at a physical dependency environment verified that
the new shim imports the new source without changing the old path file. The
fixture used minimal stand-in packages; it does not validate actual TPU/JAX
dependencies or real gcloud connectivity.

## Push from the Mac

The 7B implementation was already committed as `01785bc`. These commands stage
only the revised independent-directory documentation. The pre-existing
uncommitted `develop.md` notes are left intact outside this commit.

```bash
cd '/Users/jason/Documents/1. Education/9. Codex/DTV_GRPO_AIME/tunix'
git branch --show-current
git add \
  runs_xuesong/AIME_7B.md \
  runs_xuesong/AIME_7B_DEPLOY.md
git diff --cached --check
git diff --cached --stat
git commit -m "Document isolated Project_7B deployment and environments"
git push -u origin codex/aime-deepseek-7b
git rev-parse HEAD
```

## Create a separate checkout on the directly connected server worker

Use the actual TPU name and zone. No fixed physical-IP or service-account name
is assumed. Worker 1 remains a source deployment directory, not a Git checkout.
First inspect device owners/processes on both workers. The commands below do
not stop processes or remove locks.

```bash
export OLD_REPO=/home/jason_chia925_gmail_com/Project/tunix
export REPO=/home/jason_chia925_gmail_com/Project_7B/tunix
export VENV="$REPO/.venv"
export TPU_NAME='<actual TPU name>'
export ZONE='<actual zone>'
export REMOTE_WORKER_INDEX=1
sudo fuser -v /dev/vfio/0
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --worker="$REMOTE_WORKER_INDEX" --internal-ip \
  --command='hostname; sudo fuser -v /dev/vfio/0'
```

An empty `fuser` result normally has a nonzero status. The earlier update block
used `test -z "$(git status --porcelain)"` with `set -e`; a dirty checkout exits
silently before fetch. Inspect it read only with `git -C "$OLD_REPO" status
--short`. No stash, reset, branch switch or pull in the old checkout is needed.

Clone the pushed branch into the new directory. This reuses the old remote URL
and authentication scheme, while obtaining committed source directly from the
remote. It refuses to overwrite any existing destination.

```bash
(
  set -euo pipefail
  if [[ -e "$REPO" || -L "$REPO" ]]; then
    echo "Destination already exists; inspect it before continuing: $REPO" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$REPO")"
  ORIGIN_URL="$(git -C "$OLD_REPO" remote get-url origin)"
  git clone --single-branch --branch codex/aime-deepseek-7b \
    "$ORIGIN_URL" "$REPO"
  git -C "$REPO" branch --show-current
  git -C "$REPO" rev-parse HEAD
)
```

For later updates, use the new checkout only. This dirty-tree guard includes an
explicit explanation rather than silently returning to the prompt:

```bash
(
  set -euo pipefail
  cd "$REPO"
  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short
    echo "New checkout has local changes; update stopped: $REPO" >&2
    exit 1
  fi
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

## Create a new shim environment on worker 0

Do not link `Project_7B/tunix/.venv` to the old shim: its `.pth` may pin imports
to `Project/tunix`. Discover the underlying dependency environment from the
old working interpreter's JAX package metadata, then run the existing bootstrap
with an explicit new target. The bootstrap writes only the new lightweight
environment and reuses dependency packages read only; it does not invoke pip.

```bash
(
  set -euo pipefail
  test -x "$OLD_REPO/.venv/bin/python"
  SOURCE_ENV="$(env -u PYTHONPATH JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    "$OLD_REPO/.venv/bin/python" - <<'PY'
from pathlib import Path
from importlib.metadata import distribution

site = Path(distribution("jax").locate_file("")).resolve()
for candidate in (site, *site.parents):
    if (candidate / "bin/python").is_file():
        print(candidate)
        break
else:
    raise SystemExit(f"Cannot locate dependency environment from {site}")
PY
  )"
  env -u PYTHONPATH -u PYTHON_BIN \
    REPO="$REPO" SOURCE_ENV="$SOURCE_ENV" TARGET_ENV="$VENV" \
    JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    bash "$REPO/runs_xuesong/scripts/bootstrap_jason_env.sh"
)
```

If `.venv` already exists, bootstrap stops. Inspect it rather than deleting or
replacing it. A usable new environment must report `tunix_source` inside
`Project_7B/tunix`. For subsequent commands, clear a stale `PYTHON_BIN` and pin
`PYTHONPATH` to the new source:

```bash
unset PYTHON_BIN
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1
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

The preparation command creates only the new remote repository/cache
directories. It does not replace the old repository, its environment or any
existing model/data links. The new remote `.venv` is created in the next step.

```bash
(
  set -euo pipefail
  cd "$REPO"
  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short
    echo "Deployment stopped: new checkout has local changes." >&2
    exit 1
  fi
  DEPLOY_SHA="$(git rev-parse HEAD)"
  DEPLOY_DIR="${REPO}/runs_xuesong/cache/deploy/${DEPLOY_SHA}"
  mkdir -p "$DEPLOY_DIR"
  git archive --format=tar.gz --output="$DEPLOY_DIR/source.tar.gz" HEAD
  git ls-files -z | xargs -0 sha256sum > "$DEPLOY_DIR/source.sha256"
  (
    cd "$DEPLOY_DIR"
    sha256sum source.tar.gz > archive.sha256
  )

  printf -v PREPARE_COMMAND 'set -e\nmkdir -p %q\nmkdir -p %q' \
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

## Create the new environment on worker 1

After every archive/source checksum is `OK`, use the following from worker 0.
The dependency discovery runs on worker 1 itself because its physical
environment path can differ from worker 0. Nothing is written to the old tree.

```bash
(
  set -euo pipefail
  printf -v REMOTE_HEADER 'export OLD_REPO=%q\nexport REPO=%q\nexport VENV=%q\n' \
    "$OLD_REPO" "$REPO" "$VENV"
  REMOTE_BODY=$(cat <<'SH'
set -euo pipefail
test -x "$OLD_REPO/.venv/bin/python"
SOURCE_ENV="$(env -u PYTHONPATH JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  "$OLD_REPO/.venv/bin/python" - <<'PY'
from pathlib import Path
from importlib.metadata import distribution

site = Path(distribution("jax").locate_file("")).resolve()
for candidate in (site, *site.parents):
    if (candidate / "bin/python").is_file():
        print(candidate)
        break
else:
    raise SystemExit(f"Cannot locate dependency environment from {site}")
PY
)"
env -u PYTHONPATH -u PYTHON_BIN \
  REPO="$REPO" SOURCE_ENV="$SOURCE_ENV" TARGET_ENV="$VENV" \
  JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  bash "$REPO/runs_xuesong/scripts/bootstrap_jason_env.sh"
cd "$REPO"
env -u PYTHON_BIN PYTHONPATH="$REPO" JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -m pytest -q \
  tests/cli/aime_launchers_test.py tests/models/qwen2/config_test.py \
  tests/scripts/dual_worker_status_test.py tests/cli/grpo_main_distributed_test.py \
  tests/cli/config_test.py tests/cli/recipes/deepscaler_eval_test.py
SH
  )
  REMOTE_SETUP="${REMOTE_HEADER}${REMOTE_BODY}"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --worker="$REMOTE_WORKER_INDEX" --internal-ip \
    --command="bash -lc $(printf '%q' "$REMOTE_SETUP")"
)
```

Then run the unchanged 1.5B regression gate from the new checkout, followed by
7B integration gates. `REPO` and `VENV` must remain the new paths when invoking
either launcher; otherwise their backward-compatible defaults use the old
checkout. Existing model/data files can stay at their readable common paths;
new 7B model files can live under `$REPO/runs_xuesong/cache/models/` when they
are provisioned on both workers. Do not copy old run/log/checkpoint directories
as part of creating the new checkout. The two projects use the same TPU devices
and default RPC port, so train them sequentially.

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
