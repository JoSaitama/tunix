"""Opt-in prompt-group reward rank reversal for online GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Sequence

import numpy as np
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOLearner


NOISE_FRACTION_ENV = "TUNIX_REWARD_RANK_NOISE_FRACTION"
NOISE_SEED_ENV = "TUNIX_REWARD_RANK_NOISE_SEED"
NOISE_SCHEMA = "grpo-reward-rank-reversal-v1"


@dataclass(frozen=True)
class RewardRankNoiseConfig:
    """Configuration for deterministic prompt-group reward corruption."""

    fraction: float
    seed: int

    @property
    def enabled(self) -> bool:
        return self.fraction > 0.0


@dataclass(frozen=True)
class RewardRankNoiseAudit:
    """Host-side audit values for one reward corruption call."""

    selected_groups: np.ndarray
    effective_groups: np.ndarray
    changed_completions: np.ndarray
    clean_rewards: np.ndarray
    corrupted_rewards: np.ndarray


def config_from_env() -> RewardRankNoiseConfig:
    """Parses opt-in noise settings without changing the existing CLI."""
    raw_fraction = os.environ.get(NOISE_FRACTION_ENV, "").strip()
    if not raw_fraction:
        return RewardRankNoiseConfig(fraction=0.0, seed=0)

    try:
        fraction = float(raw_fraction)
    except ValueError as exc:
        raise ValueError(
            f"{NOISE_FRACTION_ENV} must be a float in [0, 1]; "
            f"got {raw_fraction!r}."
        ) from exc
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{NOISE_FRACTION_ENV} must be in [0, 1]; got {fraction}."
        )

    raw_seed = os.environ.get(NOISE_SEED_ENV, "0").strip()
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError(
            f"{NOISE_SEED_ENV} must be a nonnegative integer; "
            f"got {raw_seed!r}."
        ) from exc
    if seed < 0:
        raise ValueError(
            f"{NOISE_SEED_ENV} must be nonnegative; got {seed}."
        )

    return RewardRankNoiseConfig(fraction=fraction, seed=seed)


def stable_prompt_score(prompt: str, seed: int) -> float:
    """Returns a deterministic value in [0, 1) for prompt selection."""
    payload = f"{NOISE_SCHEMA}\0{seed}\0{prompt}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    numerator = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return numerator / float(1 << 64)


def selected_prompt(prompt: str, config: RewardRankNoiseConfig) -> bool:
    """Selects a prompt group using a stable approximate-rate threshold."""
    return stable_prompt_score(prompt, config.seed) < config.fraction


def reverse_reward_ranks(rewards: np.ndarray) -> np.ndarray:
    """Reassigns ascending reward values to descending-ranked samples."""
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"rewards must be one-dimensional; got shape={values.shape}."
        )
    order = np.argsort(values, kind="stable")
    reversed_values = values.copy()
    reversed_values[order] = values[order[::-1]]
    return reversed_values


def apply_reward_rank_noise(
    rewards: np.ndarray,
    prompts: Sequence[str],
    num_generations: int,
    config: RewardRankNoiseConfig,
) -> RewardRankNoiseAudit:
    """Reverses rewards for deterministically selected contiguous groups."""
    clean = np.asarray(rewards, dtype=np.float64)
    prompt_values = [str(prompt) for prompt in prompts]
    if clean.ndim != 1:
        raise ValueError(
            f"rewards must be one-dimensional; got shape={clean.shape}."
        )
    if len(prompt_values) != clean.shape[0]:
        raise ValueError(
            "prompts and rewards must have the same length; "
            f"got {len(prompt_values)} and {clean.shape[0]}."
        )
    if num_generations <= 1:
        raise ValueError(
            f"num_generations must be greater than 1; got {num_generations}."
        )
    if clean.shape[0] % num_generations != 0:
        raise ValueError(
            "reward count must be divisible by num_generations; "
            f"got {clean.shape[0]} and {num_generations}."
        )

    num_groups = clean.shape[0] // num_generations
    grouped_rewards = clean.reshape(num_groups, num_generations)
    corrupted = grouped_rewards.copy()
    selected_groups = np.zeros(num_groups, dtype=np.bool_)

    for group_index in range(num_groups):
        start = group_index * num_generations
        stop = start + num_generations
        group_prompts = prompt_values[start:stop]
        if any(prompt != group_prompts[0] for prompt in group_prompts[1:]):
            raise ValueError(
                "GRPO reward groups must contain repeated copies of one "
                f"prompt; group_index={group_index}."
            )
        if selected_prompt(group_prompts[0], config):
            selected_groups[group_index] = True
            corrupted[group_index] = reverse_reward_ranks(
                grouped_rewards[group_index]
            )

    corrupted_flat = corrupted.reshape(clean.shape)
    changed_completions = ~np.isclose(
        corrupted_flat, clean, equal_nan=True
    )
    effective_groups = changed_completions.reshape(
        num_groups, num_generations
    ).any(axis=1)
    return RewardRankNoiseAudit(
        selected_groups=selected_groups,
        effective_groups=effective_groups,
        changed_completions=changed_completions,
        clean_rewards=clean,
        corrupted_rewards=corrupted_flat,
    )


class RewardRankNoiseGRPOLearner(GRPOLearner):
    """GRPO learner that corrupts selected TRAIN reward rankings only."""

    def __init__(self, *args, noise_config: RewardRankNoiseConfig, **kwargs):
        if not noise_config.enabled:
            raise ValueError(
                "RewardRankNoiseGRPOLearner requires a positive noise fraction."
            )
        super().__init__(*args, **kwargs)
        self.noise_config = noise_config

    def _buffer_noise_metrics(
        self,
        audit: RewardRankNoiseAudit,
        mode: rl_cluster_lib.Mode,
        step: int | None,
    ) -> None:
        selected_fraction = float(np.mean(audit.selected_groups))
        effective_fraction = float(np.mean(audit.effective_groups))
        changed_fraction = float(np.mean(audit.changed_completions))
        mean_abs_delta = float(
            np.mean(np.abs(audit.corrupted_rewards - audit.clean_rewards))
        )
        metrics = {
            "noise/configured_fraction": (
                self.noise_config.fraction,
                np.mean,
            ),
            "noise/selected_group_fraction": (
                selected_fraction,
                np.mean,
            ),
            "noise/effective_group_fraction": (
                effective_fraction,
                np.mean,
            ),
            "noise/changed_completion_fraction": (
                changed_fraction,
                np.mean,
            ),
            "noise/reward_assignment_abs_delta": (
                mean_abs_delta,
                np.mean,
            ),
        }
        if step is None:
            self.rl_cluster.buffer_metrics(metrics, mode=mode)
        else:
            self.rl_cluster.buffer_metrics_async(
                metrics, mode=mode, step=step
            )

        for clean_reward, corrupted_reward in zip(
            audit.clean_rewards, audit.corrupted_rewards
        ):
            reward_metrics = {
                "rewards/clean_sum": (clean_reward, np.mean),
                "rewards/corrupted_sum": (corrupted_reward, np.mean),
            }
            if step is None:
                self.rl_cluster.buffer_metrics(reward_metrics, mode=mode)
            else:
                self.rl_cluster.buffer_metrics_async(
                    reward_metrics, mode=mode, step=step
                )

    def _compute_rewards(
        self,
        prompts,
        completions,
        mode: rl_cluster_lib.Mode,
        step: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        clean_rewards = super()._compute_rewards(
            prompts=prompts,
            completions=completions,
            mode=mode,
            step=step,
            **kwargs,
        )
        if mode != rl_cluster_lib.Mode.TRAIN:
            return clean_rewards

        audit = apply_reward_rank_noise(
            rewards=clean_rewards,
            prompts=prompts,
            num_generations=self.algo_config.num_generations,
            config=self.noise_config,
        )
        self._buffer_noise_metrics(audit, mode, step)
        return audit.corrupted_rewards
