# Dual-Worker TPU Reproduction Start Guide

This document is the operational handoff for preparing and launching the AIME
GRPO reproduction on a two-worker TPU VM. All commands run on TPU VMs. The Mac
checkout is used only to edit, commit, and push source code.

Do not poll processes or logs in a loop. Use the finite checks in this guide.
Do not run a single-worker `jax.devices()` test on a multi-host TPU slice.

## 1. Topology and invariants

The known working topology for `node-v5p-16-ziao1` was:

```text
direct/login host: worker 0
remote host:       worker 1
REMOTE_WORKER_INDEX=1
```

The planned topology for the new ziao2 node is reversed:

```text
direct/login host: worker 1
remote host:       worker 0
REMOTE_WORKER_INDEX=0
```

The wrapper may be launched from either worker. `REMOTE_WORKER_INDEX` must be
the other worker. Do not reverse `TUNIX_PROCESS_HOSTS`: the wrapper obtains the
canonical endpoint order from the TPU API, and JAX process indices must retain
that order.

Use the same absolute repository, environment, model, dataset, run, log, and
cache paths on both workers. The paths may be symlinks to different physical
locations on each worker. Models and datasets do not need to be copied when a
readable local copy already exists on each worker.

Generated output layout:

```text
/home/jason_chia925_gmail_com/Project/tunix/runs_xuesong/runs/<run-name>
/home/jason_chia925_gmail_com/Project/tunix/runs_xuesong/logs/<run-name>
/home/jason_chia925_gmail_com/Project/tunix/runs_xuesong/cache/<run-name>
```

## 2. Set node-specific values

On the directly logged-in worker, set the real TPU name and zone. Confirm the
name instead of copying the example literally.

```bash
export TPU_NAME='<ziao2 TPU VM name>'
export ZONE='<ziao2 zone>'
export BRANCH='for_GRPO_vLLM'
export REPO='/home/jason_chia925_gmail_com/Project/tunix'
export REMOTE_WORKER_INDEX=0
```

For the reversed ziao2 topology, the current host must be worker 1:

```bash
hostname
whoami

gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
  --zone="$ZONE" \
  --format='yaml(name,state,health,acceleratorType,networkEndpoints)'
```

Expected properties:

```text
hostname ends in -w-1
state: READY
exactly two network endpoints
acceleratorType: v5p-16
```

Confirm that the direct worker can reach worker 0:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker=0 \
  --internal-ip \
  --command='hostname; whoami; id; sudo -n true && echo SUDO_OK'
```

Do not continue unless the output host ends in `-w-0` and passwordless sudo is
available on the remote deployment account.

## 3. Clone the canonical repository on the direct worker

The directly logged-in worker is the only Git working tree and launch host for
this node. Clone once:

```bash
sudo install -d -m 0755 \
  -o "$(id -un)" \
  -g "$(id -gn)" \
  /home/jason_chia925_gmail_com \
  /home/jason_chia925_gmail_com/Project

git clone \
  --branch "$BRANCH" \
  --single-branch \
  git@github.com:JoSaitama/tunix.git \
  "$REPO"

cd "$REPO"
git status --short --branch
git rev-parse HEAD
```

If the repository already exists, do not clone over it:

```bash
cd "$REPO"
git status --short
git pull --ff-only origin "$BRANCH"
git rev-parse HEAD
```

`git status --short` must be empty before deployment or launch. Never run
`git reset --hard` to resolve a dirty tree.

## 4. Discover existing environments, models, and datasets

Run these finite searches separately on both workers. `find` is used because
`rg` may not be installed.

```bash
echo '=== Python environments ==='
find /home -maxdepth 5 -type f -name pyvenv.cfg -print 2>/dev/null | sort

echo '=== candidate models ==='
find /home /tmp -maxdepth 6 -type f \
  -name model.safetensors -printf '%s %p\n' 2>/dev/null | sort -n

echo '=== candidate datasets ==='
find /home -maxdepth 6 -type f \
  \( -name deepscaler_train.json -o -name aime_eval.parquet \) \
  -print 2>/dev/null | sort
```

Run the same command on worker 0 from the direct worker:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker="$REMOTE_WORKER_INDEX" \
  --internal-ip \
  --command="
    find /home -maxdepth 5 -type f -name pyvenv.cfg -print 2>/dev/null | sort
    find /home /tmp -maxdepth 6 -type f -name model.safetensors \
      -printf '%s %p\\n' 2>/dev/null | sort -n
    find /home -maxdepth 6 -type f \
      \( -name deepscaler_train.json -o -name aime_eval.parquet \) \
      -print 2>/dev/null | sort
  "
```

Each worker requires one readable TPU environment, model, training dataset,
and evaluation dataset. Stop if an asset is absent. Do not download CUDA vLLM
or rebuild the historical TPU vLLM environment.

Known historical physical paths were:

```text
worker-0 style:
  /home/lhf_hongfu_gmail_com/tunix/.venv
  /home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b
  /home/lhf_hongfu_gmail_com/tunix-hf-data

worker-1 style:
  /home/eve/tunix/.venv
  /home/eve/models/deepseek-r1-distill-qwen-1.5b
  /home/eve/tunix-hf-data
```

## 5. Create compatible paths on each worker

The wrapper uses these common paths:

```text
environment: /home/jason_chia925_gmail_com/Project/tunix/.venv
model:       /home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b
datasets:    /home/lhf_hongfu_gmail_com/tunix-hf-data
```

On each worker, create symlinks to that worker's discovered physical assets.
Replace the three `PHYSICAL_*` values independently on each worker.

```bash
PHYSICAL_ENV='<existing TPU .venv on this worker>'
PHYSICAL_MODELS='<directory containing deepseek-r1-distill-qwen-1.5b>'
PHYSICAL_DATA='<directory containing both dataset files>'

test -x "$PHYSICAL_ENV/bin/python"
test -r "$PHYSICAL_MODELS/deepseek-r1-distill-qwen-1.5b/model.safetensors"
test -r "$PHYSICAL_DATA/deepscaler_train.json"
test -r "$PHYSICAL_DATA/aime_eval.parquet"

sudo install -d -m 0755 \
  -o "$(id -un)" \
  -g "$(id -gn)" \
  /home/jason_chia925_gmail_com \
  /home/jason_chia925_gmail_com/Project \
  "$REPO" \
  /home/jason_chia925_gmail_com/tunix-runs \
  /home/jason_chia925_gmail_com/tunix-cache

ln -sfn "$PHYSICAL_ENV" "$REPO/.venv"

sudo install -d -m 0755 /home/lhf_hongfu_gmail_com
sudo ln -sfn "$PHYSICAL_MODELS" /home/lhf_hongfu_gmail_com/models
sudo ln -sfn "$PHYSICAL_DATA" /home/lhf_hongfu_gmail_com/tunix-hf-data
```

Before using `ln -sfn`, inspect any pre-existing destination. Do not replace a
real directory with a symlink. If a destination is a real directory containing
valid assets, use it directly.

Validate on each worker:

```bash
test -x "$REPO/.venv/bin/python" &&
test -r /home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b/model.safetensors &&
test -r /home/lhf_hongfu_gmail_com/tunix-hf-data/deepscaler_train.json &&
test -r /home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet &&
echo ASSETS_READY
```

## 6. Deploy the same committed source to remote worker 0

Create the remote compatibility repository as the remote SSH account:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker="$REMOTE_WORKER_INDEX" \
  --internal-ip \
  --command='
    REMOTE_USER="$(id -un)"
    REMOTE_GROUP="$(id -gn)"
    sudo install -d -m 0755 -o "$REMOTE_USER" -g "$REMOTE_GROUP" \
      /home/jason_chia925_gmail_com \
      /home/jason_chia925_gmail_com/Project \
      /home/jason_chia925_gmail_com/Project/tunix
  '
```

Deploy tracked files from the direct worker. The remote directory is a source
snapshot, not a second Git working tree:

```bash
cd "$REPO"
git status --short
git rev-parse HEAD

git archive --format=tar HEAD |
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker="$REMOTE_WORKER_INDEX" \
  --internal-ip \
  --command='tar -xf - -C /home/jason_chia925_gmail_com/Project/tunix'
```

Create the remote `.venv`, model, and dataset compatibility links using the
worker-0 physical paths discovered in section 4. Do not copy model weights if
worker 0 already has them.

Compare source checksums:

```bash
cd "$REPO"
sha256sum \
  runs_xuesong/scripts/run_official_like_dual_worker.sh \
  runs_xuesong/scripts/run_aime_total_loss_reproduction.sh

gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker="$REMOTE_WORKER_INDEX" \
  --internal-ip \
  --command='
    cd /home/jason_chia925_gmail_com/Project/tunix
    sha256sum \
      runs_xuesong/scripts/run_official_like_dual_worker.sh \
      runs_xuesong/scripts/run_aime_total_loss_reproduction.sh
  '
```

The two pairs of hashes must match.

## 7. Disk, journal, Docker, and TPU-agent checks

Run locally and remotely. A smoke needs enough space for caches and a
checkpoint. Prefer at least 15 GiB free; long experiments need more.

```bash
df -h /
df -ih /
sudo du -xhd1 /var /tmp /home 2>/dev/null | sort -h
sudo du -xhd2 /var/log 2>/dev/null | sort -h | tail -40
sudo journalctl --disk-usage 2>&1 || true
sudo find /var/log /tmp -xdev -type f -size +500M \
  -printf '%s %p\n' 2>/dev/null | sort -n
```

`/var/log/lastlog` is commonly sparse. `find -printf %s` reports its logical
size; use `du -h /var/log/lastlog` for actual disk blocks. Do not delete it
based only on its logical size.

Inspect the TPU health agent without changing it:

```bash
sudo docker ps --no-trunc --filter name=healthagent
sudo docker inspect \
  --format='name={{.Name}} pid={{.State.Pid}} status={{.State.Status}} started={{.State.StartedAt}} memory_limit={{.HostConfig.Memory}}' \
  healthagent 2>/dev/null || true
sudo docker stats --no-stream \
  --format='{{.Name}} memory={{.MemUsage}} cpu={{.CPUPerc}}' \
  healthagent 2>/dev/null || true
sudo journalctl -k --since '10 minutes ago' --no-pager |
  grep -E 'healthAgent invoked oom-killer|Out of memory' |
  tail -30 || true
```

If current healthAgent OOM messages exist and memory is at its 512 MiB limit,
restart it once:

```bash
sudo docker restart healthagent
sudo docker stats --no-stream \
  --format='{{.Name}} memory={{.MemUsage}} cpu={{.CPUPerc}}' healthagent
sudo journalctl -k --since '1 minute ago' --no-pager |
  grep -E 'healthAgent invoked oom-killer|Out of memory' |
  tail -20 || true
```

A successful recovery has a new container PID, low memory use, and no OOM after
the restart time. Do not repeatedly restart the agent.

Only when system logs are consuming material disk space, stop the OOM source
first and then reclaim historical logs. Truncation permanently removes the old
contents but preserves the active files:

```bash
sudo systemctl stop rsyslog
sudo truncate -s 0 /var/log/syslog /var/log/kern.log
sudo systemctl start rsyslog

sudo journalctl --rotate
sudo journalctl --vacuum-size=200M
sudo systemctl restart systemd-journald

sudo journalctl --disk-usage
df -h /
```

Do not recursively delete `/var/log`, `/tmp`, a home directory, an environment,
or a model directory. Inspect duplicate `/tmp/models` weights with `lsof`
before deciding whether they are removable.

## 8. Confirm that both TPU workers are idle

On the direct worker:

```bash
pgrep -a -x python || true
pgrep -a -x python3 || true
pgrep -a -f 'grpo_main|deepscaler|vllm' || true
```

On remote worker 0:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker="$REMOTE_WORKER_INDEX" \
  --internal-ip \
  --command="
    pgrep -a -x python || true
    pgrep -a -x python3 || true
    pgrep -a -f 'grpo_main|deepscaler|vllm' || true
    df -h /
  "
```

The local `gcloud.py ... ssh` process is only the SSH client and does not own
the TPU. Do not launch if an old JAX, vLLM, GRPO, pytest, or training process is
present. Do not run a one-worker JAX TPU availability test.

## 9. Baseline one-batch smoke from direct worker 1

This is the reversed ziao2 launch. It runs locally on worker 1 and asks the
wrapper to start worker 0 remotely.

```bash
cd "$REPO"

RUN_NAME="grpo_aime_baseline_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$REPO/runs_xuesong/runs/$RUN_NAME"
LOG_ROOT="$REPO/runs_xuesong/logs/$RUN_NAME"
CACHE_ROOT="$REPO/runs_xuesong/cache/$RUN_NAME"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CACHE_ROOT"

nohup env \
  METHOD=baseline \
  NUM_BATCHES=1 \
  MAX_STEPS_OVERRIDE=1 \
  CHECKPOINT_INTERVAL=1 \
  RUN_NAME="$RUN_NAME" \
  RUN_ROOT="$RUN_ROOT" \
  LOG_ROOT="$LOG_ROOT" \
  CACHE_ROOT="$CACHE_ROOT" \
  REPO="$REPO" \
  VENV="$REPO/.venv" \
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  REMOTE_WORKER_INDEX="$REMOTE_WORKER_INDEX" \
  MODEL_PATH=/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b \
  TOKENIZER_PATH=/home/lhf_hongfu_gmail_com/models/deepseek-r1-distill-qwen-1.5b \
  TRAIN_DATA_PATH=/home/lhf_hongfu_gmail_com/tunix-hf-data/deepscaler_train.json \
  EVAL_DATA_PATH=/home/lhf_hongfu_gmail_com/tunix-hf-data/aime_eval.parquet \
  TUNIX_DISABLE_TRAJECTORY_LOGGING=1 \
  bash runs_xuesong/scripts/run_aime_total_loss_reproduction.sh \
  > "$LOG_ROOT/launcher.out" 2>&1 < /dev/null &

SMOKE_PID=$!
echo "$SMOKE_PID" > "$LOG_ROOT/launcher.pid"

echo "RUN_NAME=$RUN_NAME"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "LAUNCHER_PID=$SMOKE_PID"
```

The uppercase `NUM_BATCHES=1` is propagated by the wrapper as lowercase
`num_batches=1` to both worker processes before the recipe computes its
training schedule.

## 10. Finite smoke inspection and acceptance

Do not use `tail -F` or a polling loop. Inspect once after allowing the first
model initialization and compilation to proceed:

```bash
ps -fp "$(cat "$LOG_ROOT/launcher.pid")" || true
tail -n 120 "$LOG_ROOT/launcher.out"
tail -n 160 "$LOG_ROOT/workers/local.log"
cat "$LOG_ROOT/workers/remote.status" 2>/dev/null || true
find "$RUN_ROOT/checkpoints" -maxdepth 4 -type f \
  -printf '%s %p\n' 2>/dev/null | sort
```

Inspect the remote worker log once:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --worker="$REMOTE_WORKER_INDEX" \
  --internal-ip \
  --command="tail -n 160 '$LOG_ROOT/workers/remote.log'"
```

The smoke passes only when:

```text
LOCAL_STATUS=0
REMOTE_STATUS=0
one rollout completes
one optimizer step completes
a checkpoint exists under RUN_ROOT/checkpoints
```

Transient `lost its connection to the rollout owner; retrying once after
remote restart` warnings are not automatically fatal. In the known ziao1 run,
vLLM recovered, accepted the retried requests, and drained `Running` requests
from 1024 to zero. Judge the final statuses and subsequent progress, not the
warning alone.

TPU vLLM may also report that CUDA `vllm._C` or Triton is unavailable. These
messages are expected on the TPU build unless followed by a fatal exception.

## 11. DTV smoke commands after baseline passes

Use a new run name for each method. Keep all other settings matched.

```bash
METHOD=dtv_batch_total_loss
```

or:

```bash
METHOD=dtv_group_total_loss
```

Substitute the method in the section 9 `nohup env` command. Do not run the
three methods concurrently on the same TPU slice.

## 12. Source update workflow

All source edits are made on the Mac checkout. Codex edits locally but does not
push. The operator runs:

```bash
git status --short
git diff --check
git add <explicit files>
git commit -m '<message>'
git push origin for_GRPO_vLLM
```

On the direct TPU worker, only while no experiment is running:

```bash
cd "$REPO"
git status --short
git pull --ff-only origin for_GRPO_vLLM
git rev-parse HEAD
```

Then redeploy the committed snapshot to the other worker with `git archive` as
shown in section 6. Never edit the remote snapshot independently.
