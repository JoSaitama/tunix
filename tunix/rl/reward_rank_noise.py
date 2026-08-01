"""Opt-in prompt-group reward rank reversal for online GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Sequence

import numpy as np


NOISE_FRACTION_ENV = "TUNIX_REWARD_RANK_NOISE_FRACTION"
NOISE_SEED_ENV = "TUNIX_REWARD_RANK_NOISE_SEED"
NOISE_SCHEMA = "grpo-reward-rank-reversal-v1"


@dataclass(frozen=True)
class RewardRankNoiseConfig:
  fraction: float
  seed: int

  @property
  def enabled(self) -> bool:
    return self.fraction > 0.0


@dataclass(frozen=True)
class RewardRankNoiseAudit:
  selected_groups: np.ndarray
  effective_groups: np.ndarray
  changed_completions: np.ndarray
  clean_rewards: np.ndarray
  corrupted_rewards: np.ndarray


def config_from_env() -> RewardRankNoiseConfig:
  raw_fraction = os.environ.get(NOISE_FRACTION_ENV, "").strip()
  if not raw_fraction:
    return RewardRankNoiseConfig(fraction=0.0, seed=0)
  try:
    fraction = float(raw_fraction)
  except ValueError as exc:
    raise ValueError(
        f"{NOISE_FRACTION_ENV} must be a float in [0, 1]; got {raw_fraction!r}."
    ) from exc
  if not 0.0 <= fraction <= 1.0:
    raise ValueError(f"{NOISE_FRACTION_ENV} must be in [0, 1]; got {fraction}.")
  raw_seed = os.environ.get(NOISE_SEED_ENV, "0").strip()
  try:
    seed = int(raw_seed)
  except ValueError as exc:
    raise ValueError(
        f"{NOISE_SEED_ENV} must be a nonnegative integer; got {raw_seed!r}."
    ) from exc
  if seed < 0:
    raise ValueError(f"{NOISE_SEED_ENV} must be nonnegative; got {seed}.")
  return RewardRankNoiseConfig(fraction=fraction, seed=seed)


def stable_prompt_score(prompt: str, seed: int) -> float:
  payload = f"{NOISE_SCHEMA}\0{seed}\0{prompt}".encode("utf-8")
  digest = hashlib.sha256(payload).digest()
  return int.from_bytes(digest[:8], "big") / float(1 << 64)


def selected_prompt(prompt: str, config: RewardRankNoiseConfig) -> bool:
  return stable_prompt_score(prompt, config.seed) < config.fraction


def reverse_reward_ranks(rewards: np.ndarray) -> np.ndarray:
  values = np.asarray(rewards, dtype=np.float64)
  if values.ndim != 1:
    raise ValueError(f"rewards must be one-dimensional; got {values.shape}.")
  order = np.argsort(values, kind="stable")
  result = values.copy()
  result[order] = values[order[::-1]]
  return result


def apply_reward_rank_noise(
    rewards: np.ndarray,
    prompts: Sequence[str],
    num_generations: int,
    config: RewardRankNoiseConfig,
) -> RewardRankNoiseAudit:
  clean = np.asarray(rewards, dtype=np.float64)
  prompt_values = [str(prompt) for prompt in prompts]
  if clean.ndim != 1 or len(prompt_values) != clean.shape[0]:
    raise ValueError("prompts and one-dimensional rewards must have equal length.")
  if num_generations <= 1 or clean.shape[0] % num_generations:
    raise ValueError("reward count must be divisible by num_generations > 1.")
  num_groups = clean.shape[0] // num_generations
  grouped = clean.reshape(num_groups, num_generations)
  corrupted = grouped.copy()
  selected = np.zeros(num_groups, dtype=np.bool_)
  for index in range(num_groups):
    start = index * num_generations
    group_prompts = prompt_values[start : start + num_generations]
    if any(prompt != group_prompts[0] for prompt in group_prompts[1:]):
      raise ValueError(f"reward group {index} does not repeat one prompt.")
    if selected_prompt(group_prompts[0], config):
      selected[index] = True
      corrupted[index] = reverse_reward_ranks(grouped[index])
  corrupted_flat = corrupted.reshape(clean.shape)
  changed = ~np.isclose(corrupted_flat, clean, equal_nan=True)
  effective = changed.reshape(num_groups, num_generations).any(axis=1)
  return RewardRankNoiseAudit(
      selected, effective, changed, clean, corrupted_flat
  )
