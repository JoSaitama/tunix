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
locally. Worker 0 subsequently passed all 66 focused CPU checks, including
production-schema inheritance and dependency-backed distributed tests. Worker
1's checks remain pending until deployment. These are source/CPU checks; 7B
changes memory/compile and transfer costs and still requires real TPU gates.

All shell examples and the generated remote setup body were syntax-checked.
Temporary local fixtures also verified cloning while preserving a dirty old
checkout, archive/source checksums and the deployed commit marker. A fixture
with an old shim pointing at a physical dependency environment verified that
the new shim imports the new source without changing the old path file. The
fixture used minimal stand-in packages; it does not validate actual TPU/JAX
dependencies or real gcloud connectivity.

## Push from the Mac

The 7B implementation was committed as `01785bc`; worker 0's focused CPU gate
passed all 66 tests at `ebf8d19`. These commands stage only the new deployment
helper, its tests and updated instructions. The pre-existing uncommitted
`develop.md` notes are left intact outside this commit.

```bash
cd '/Users/jason/Documents/1. Education/9. Codex/DTV_GRPO_AIME/tunix'
git branch --show-current
git add \
  runs_xuesong/scripts/deploy_aime_7b_worker.sh \
  tests/scripts/aime_7b_deploy_test.py \
  runs_xuesong/AIME_7B.md \
  runs_xuesong/AIME_7B_DEPLOY.md
git diff --cached --check
git diff --cached --stat
git commit -m "Automate 7B worker deployment with validated TPU discovery"
git push -u origin codex/aime-deepseek-7b
git rev-parse HEAD
```

## Create a separate checkout on the directly connected server worker

Worker 1 remains a source deployment directory, not a Git checkout. TPU name,
project and zone are discovered later by the deployment helper; clone and CPU
checks do not require these variables. No fixed physical-IP or service-account
name is assumed.

```bash
export OLD_REPO=/home/jason_chia925_gmail_com/Project/tunix
export REPO=/home/jason_chia925_gmail_com/Project_7B/tunix
export VENV="$REPO/.venv"
export REMOTE_WORKER_INDEX=1
```

The earlier update block used `test -z "$(git status --porcelain)"` with `set -e`; a dirty checkout exits
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
    tests/cli/aime_launchers_test.py tests/models/qwen2/deepseek_qwen_config_test.py \
    tests/scripts/dual_worker_status_test.py tests/cli/grpo_main_distributed_test.py \
    tests/cli/config_test.py tests/cli/recipes/deepscaler_eval_test.py
)
```

These checks do not allocate a real 7B model or start TPU training. Do not run
the whole `vllm_sampler_test.py` merely to check cache reuse: its shared setup
downloads a gated Llama checkpoint. The unchanged KV implementation and real
weight-sync logs can be audited without introducing that download.

The first server collection exposed a name collision between the new Qwen
`config_test.py` and the existing CLI `config_test.py`: these test directories
are not Python packages, so pytest's default import mode used the same module
name. The new file is now `deepseek_qwen_config_test.py`. This changes only test
collection and its documented paths. For an older checkout, adding
`--import-mode=importlib` to its original pytest command avoids the collision
without changing files. Cache/environment deletion is not needed for this
diagnosed collision. The corrected dependency-backed checks still need to run
on the server before deployment.

## Deploy source and create the environment on worker 1

The original manual blocks depended on previously exported `TPU_NAME` and
`ZONE`; omitting that setup produced `unbound variable` before contacting the
remote worker. Use the committed helper instead of pasting long remote shell
blocks. It detects missing project/zone from VM metadata and selects the TPU
whose endpoints match the current host's IPs. Explicit overrides are accepted
but must still match this host and exactly two distinct worker endpoints. A
wrong or ambiguous match stops before SSH.

After the worker-0 CPU checks pass, run these from worker 0:

```bash
export REPO=/home/jason_chia925_gmail_com/Project_7B/tunix
export OLD_REPO=/home/jason_chia925_gmail_com/Project/tunix
export VENV="$REPO/.venv"
export REMOTE_WORKER_INDEX=1
cd "$REPO"
# Clear unset/stale/placeholder settings and let discovery use this VM.
unset TPU_NAME ZONE TPU_PROJECT
bash runs_xuesong/scripts/deploy_aime_7b_worker.sh --dry-run
```

The dry run queries only VM metadata and the TPU list, saves discovery output
under the new repository cache, and prints the exact name/zone/project and
remote IP. It does not execute SSH or copy files. A successful dry run validates
that the selected worker is not the current host. Continue with:

```bash
(
  set -euo pipefail
  cd "$REPO"
  bash runs_xuesong/scripts/deploy_aime_7b_worker.sh
  source "$REPO/runs_xuesong/cache/deploy/$(git rev-parse HEAD)/target.env"
)
```

The subshell above limits imported settings to that block. For later training
or standalone remote commands, load them into the current shell explicitly:

```bash
source "$REPO/runs_xuesong/cache/deploy/$(git -C "$REPO" rev-parse HEAD)/target.env"
```

The saved settings include `CLOUDSDK_CORE_PROJECT` so existing launchers use
the same discovered project without changing persistent gcloud configuration.
If metadata is unavailable, set the actual `ZONE` and `TPU_PROJECT` and retry.
If the account cannot list nodes, the helper reports the gcloud error and
stops; it does not guess a resource from the VM hostname.

The helper preserves the old repository and performs these steps in order:

1. Reject a dirty new checkout; archive the exact committed source.
2. Check the old remote Python is usable and create only the new directories.
3. Copy the persistent archive and checksum manifests, stopping on any error.
4. Verify archive SHA-256 before extraction, then all source checksums and write
   the deployed commit marker. A failed copy never reaches extraction.
5. Create the remote shim from worker 1's own dependency environment if absent.
   An existing shim is validated rather than replaced. Verify imports from a
   neutral directory with `PYTHONPATH` cleared so it cannot mask an old `.pth`.
6. Compare Python, JAX, jaxlib, libtpu and vLLM versions against worker 0, then
   run the 66 focused CPU tests on worker 1 with `JAX_PLATFORMS=cpu`.

No package install, model/data copy or training is performed. The helper has
`--phase source` and `--phase env` for separately retrying those stages. The
latter requires the remote commit marker to match the local current commit.
Failed archive transfer files remain available in the new repository cache.

Nine local tests passed using fake metadata/gcloud calls, covering missing
variables, wrong/ambiguous targets, selecting the local worker, dirty source,
transfer failure, operation ordering and generated shell syntax. They do not
establish actual cloud connectivity or TPU fit. The real CPU evidence remains
worker 0's 66 passing tests; worker 1 must run its checks during deployment.

Before TPU smoke, inspect device owners on both workers (these commands do not
stop processes or remove locks):

```bash
sudo fuser -v /dev/vfio/0
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$TPU_PROJECT" --zone="$ZONE" --worker="$REMOTE_WORKER_INDEX" --internal-ip \
  --command='hostname; sudo fuser -v /dev/vfio/0'
```

An empty `fuser` result normally has a nonzero status. Both workers must be idle
for real TPU gates. Then run the 1.5B regression gate from the new checkout,
followed by 7B integration gates. Keep `REPO` and `VENV` set to the new paths;
otherwise backward-compatible launcher defaults use the old checkout. Existing
model/data files can remain at their readable common paths; new 7B weights can
live under `$REPO/runs_xuesong/cache/models/` on each worker. The two projects
share TPU devices and the default RPC port, so train them sequentially.

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
