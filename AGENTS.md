# Project Intent (Debug Context)

## Summary
- We added Dynamic Batch Curation for GRPO training to filter samples with abnormally large per-sample gradient L2 norms for stability.
- Training completes, but the post-train checkpoint restore crashes.
- Training can appear to "skip" when existing checkpoints already reach `max_steps`.

## What Was Added
- `tunix/rl/robust_trainer.py`: `RobustTrainer` with `jax.vmap` per-sample gradients and filtering.
- `tunix/rl/rl_cluster.py`: runtime selection of `RobustTrainer` via `use_dynamic_batch_curation`.
- `my_example/config.py` and `my_example/train.py`: expose config flags.
- `my_example/run_grpo_gemma.sh`: CLI flags `--use-dynamic-batch-curation` and `--curation-threshold`.

## Current Bug
- Crash during checkpoint restore after training finishes:
  `wandb.errors.errors.Error: You must call wandb.init() before wandb.log()`
- Root cause: `orbax.checkpoint` calls `jax.monitoring.record_scalar`, which routes into metrax/WandB listeners. WandB context is not active (disabled or ended), so restore fails.
- Prior attempt: `jax.monitoring.clear_event_listeners()` before restore. It did not fully prevent re-registered listeners.

## Do Not Touch
- Do not change `tunix/rl/robust_trainer.py` logic.
- Do not change `RLTrainingConfig` or CLI argument structure.
- Baseline command works:
  `./my_example/run_grpo_gemma.sh --num-test-batches 1 --metrics-log-dir /tmp/content/tmp/tensorboard/grpo_$(date +%Y%m%d_%H%M%S) --checkpoint-root /tmp/content/ckpts_run2_$(date +%Y%m%d_%H%M%S)`并且其相关的内容不允许修改，只能增加分支。

## Goal
- 使得 Dynamic batch curation command work:
  `./my_example/run_grpo_gemma.sh --use-dynamic-batch-curation --curation-threshold 3.0 --num-test-batches 1 --metrics-log-dir /tmp/content/tmp/tensorboard/grpo_$(date +%Y%m%d_%H%M%S) --checkpoint-root /tmp/content/ckpts_run2_$(date +%Y%m%d_%H%M%S)`

## Acceptance
- Baseline command works:
  `./my_example/run_grpo_gemma.sh --num-test-batches 1 --metrics-log-dir /tmp/content/tmp/tensorboard/grpo_$(date +%Y%m%d_%H%M%S) --checkpoint-root /tmp/content/ckpts_run2_$(date +%Y%m%d_%H%M%S)`并且其相关的内容不允许修改
- Dynamic batch curation command also works:

  `./my_example/run_grpo_gemma.sh --use-dynamic-batch-curation --curation-threshold 3.0 --num-test-batches 1 --metrics-log-dir /tmp/content/tmp/tensorboard/grpo_$(date +%Y%m%d_%H%M%S) --checkpoint-root /tmp/content/ckpts_run2_$(date +%Y%m%d_%H%M%S)`
