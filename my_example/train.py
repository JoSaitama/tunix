from __future__ import annotations

import os
from typing import Tuple

import optax
from orbax import checkpoint as ocp
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from .config import GrpoGenerationConfig, TrainingConfig
from .rewards import (
    check_answer,
    check_numbers,
    format_integrity_reward,
    numeric_relative_reward,
    match_format_approximately,
    match_format_exactly,
)


def build_optimizer(training: TrainingConfig, max_steps: int):
    warmup_steps = int(training.warmup_fraction * max_steps)
    optimizer = optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=training.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=0.0,
        ),
        b1=training.b1,
        b2=training.b2,
        weight_decay=training.weight_decay,
    )
    if training.max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm=training.max_grad_norm),
            optimizer,
        )
    return optimizer


def build_cluster_config(
    mesh,
    grpo: GrpoGenerationConfig,
    eval_cfg,
    training: TrainingConfig,
    optimizer,
    max_steps: int,
    train_micro_batch_size: int,
    eos_tokens: list[int],
    use_wandb: bool,
):
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=training.save_interval_steps,
        max_to_keep=training.max_to_keep,
    )
    backend_factories = None
    if not use_wandb:
        backend_factories = [
            lambda: metrics_logger.TensorboardBackend(
                log_dir=training.metrics_log_dir,
                flush_every_n_steps=training.metrics_flush_every_n_steps,
            )
        ]

    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=training.metrics_log_dir,
        flush_every_n_steps=training.metrics_flush_every_n_steps,
        backend_factories=backend_factories,
    )

    train_rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=grpo.total_generation_steps,
        max_prompt_length=grpo.max_prompt_length,
        kv_cache_size=grpo.max_prompt_length + grpo.total_generation_steps + 256,
        temperature=grpo.temperature,
        top_p=grpo.top_p,
        top_k=grpo.top_k,
        eos_tokens=eos_tokens,
    )

    eval_total_steps = (
        eval_cfg.total_generation_steps
        if eval_cfg.total_generation_steps is not None
        else grpo.total_generation_steps
    )
    eval_rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=eval_total_steps,
        max_prompt_length=eval_cfg.max_prompt_length,
        kv_cache_size=eval_cfg.max_prompt_length + eval_total_steps + 256,
        temperature=eval_cfg.temperature,
        top_p=eval_cfg.top_p,
        top_k=eval_cfg.top_k,
        eos_tokens=eos_tokens,
    )

    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=training.eval_every_n_steps,
            max_steps=max_steps,
            mini_batch_size=train_micro_batch_size,
            train_micro_batch_size=train_micro_batch_size,
            metrics_logging_options=metrics_logging_options,
            checkpoint_root_directory=training.checkpoint_root_directory,
            checkpointing_options=checkpointing_options,
            use_dynamic_batch_curation=training.use_dynamic_batch_curation,
            curation_threshold=training.curation_threshold,
        ),
        rollout_config={
            rl_cluster_lib.Mode.TRAIN: train_rollout_config,
            rl_cluster_lib.Mode.EVAL: eval_rollout_config,
        },
    )
    return cluster_config


def build_grpo_config(grpo: GrpoGenerationConfig) -> GRPOConfig:
    return GRPOConfig(
        num_generations=grpo.num_generations,
        num_iterations=grpo.num_iterations,
        beta=grpo.beta,
        epsilon=grpo.epsilon,
    )


def build_trainer(rl_cluster, grpo: GrpoGenerationConfig) -> GRPOLearner:
    grpo_config = build_grpo_config(grpo)
    use_accuracy_reward_mode = (
        os.environ.get("TUNIX_REWARD_MODE", "").lower() == "accuracy"
    )
    if use_accuracy_reward_mode:
        reward_fns = [
            format_integrity_reward,
            numeric_relative_reward,
            check_numbers,
        ]
    else:
        reward_fns = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ]
    return GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[
            *reward_fns,
        ],
        algo_config=grpo_config,
    )
