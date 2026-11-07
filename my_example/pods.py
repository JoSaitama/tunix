# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""PODS max-variance rollout selection for the frozen GSM8K GRPO pipeline.

Algorithm adapted from Yixuan Even Xu, Yash Savani, Fei Fang and J. Zico Kolter,
Not All Rollouts are Useful (arXiv:2504.13818), Apache-2.0 implementation:
https://github.com/YixuanEvenXu/pods/tree/187f43b648f69385103029c2dd0934521e17bb71
See trainer/maxvargrpo_trainer.py and open-r1/src/open_r1/grpo_trainer.py.

Only selection uses NumPy/host computation. Actor updates remain on JAX/TPU.
Generation, rewards (including rank noise), and reference/old-policy logps reuse
the original learner. Those logps are computed on all n rollouts, then gathered;
only the actor update operates on the smaller m-rollout batch. No speedup claim
is made for reference inference. Heavy framework imports are lazy for host tests.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

UPSTREAM_COMMIT = "187f43b648f69385103029c2dd0934521e17bb71"


def validate_keep_count(n: int, m: int) -> None:
    if not 2 <= m <= n:
        raise ValueError(f"PODS requires 2 <= keep_count <= num_generations; got m={m}, n={n}")


def select_max_variance_indices(rewards, keep_count: int) -> np.ndarray:
    """Return [prompts, m] local indices; never select across prompt groups.

    Enumerate top-k + bottom-(m-k) subsets, as in the author's implementation.
    Stable sorting resolves tied rewards by input order. Equal variances retain
    the first candidate (strict > comparison). m=n preserves original order.
    """
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("PODS rewards must be a finite [prompts, generations] array")
    n = values.shape[1]
    validate_keep_count(n, keep_count)
    if keep_count == n:
        return np.tile(np.arange(n), (len(values), 1))
    order = np.argsort(-values, axis=1, kind="stable")
    selected = order[:, :keep_count].copy()
    best = np.take_along_axis(values, selected, axis=1).var(axis=1, ddof=1)
    for k in range(keep_count):
        trial = np.concatenate((order[:, :k], order[:, k - keep_count:]), axis=1)
        variance = np.take_along_axis(values, trial, axis=1).var(axis=1, ddof=1)
        better = variance > best
        selected[better] = trial[better]
        best[better] = variance[better]
    return selected


def subset_advantages(rewards) -> np.ndarray:
    """Same sample standard deviation and epsilon as frozen Tunix GRPO."""
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite grouped rewards with at least two rollouts")
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, ddof=1, keepdims=True) + 1e-4
    )


def selected_micro_batch_sizes(sizes, n: int, m: int) -> list[int]:
    validate_keep_count(n, m)
    if any(size <= 0 or size % n for size in sizes):
        raise ValueError("PODS micro-batches must contain complete prompt groups")
    return [size // n * m for size in sizes]


class PODSMixin:
    """Opt-in preparation layer over ordinary or reward-rank-noise GRPO."""

    def __init__(self, *args, pods_keep_count: int, pods_artifact_dir=None, **kwargs):
        # Validate before constructing the trainer or opening any artifacts.
        config = kwargs["algo_config"]
        validate_keep_count(config.num_generations, pods_keep_count)
        cluster = kwargs["rl_cluster"]
        if cluster.cluster_config.training_config.use_dynamic_batch_curation:
            raise ValueError("PODS cannot be combined with DBC/DTV/random/reward gradient filtering")
        self.pods_keep_count = pods_keep_count
        self._pods_capture = False
        self._pods_observed = None
        self._pods_chunks = self._pods_generated = self._pods_selected = 0
        self._pods_dir = Path(pods_artifact_dir) if pods_artifact_dir else None
        super().__init__(*args, **kwargs)
        if self._pods_dir:
            self._pods_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "method": "pods", "upstream_commit": UPSTREAM_COMMIT,
                "num_generations": config.num_generations,
                "keep_count": pods_keep_count,
                "retained_fraction": pods_keep_count / config.num_generations,
                "beta": config.beta,
                "max_steps": cluster.cluster_config.training_config.max_steps,
                "experiment_seed": os.environ.get("TUNIX_EXPERIMENT_SEED"),
                "mismatch_fraction": os.environ.get("TUNIX_REWARD_RANK_NOISE_FRACTION", "0"),
                "mismatch_seed": os.environ.get("TUNIX_REWARD_RANK_NOISE_SEED", "0"),
                "argv": sys.argv,
                "adaptations": [
                    "frozen dense reward, model, LoRA, optimizer, KL and rollout budget",
                    "rank noise applied on all n rewards before selection",
                    "subset advantage: sample std (ddof=1), epsilon=1e-4",
                    "stable input-order reward ties; m=n identity order",
                    "reference/old logps on n rollouts; actor update on m",
                ],
            }
            (self._pods_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        print(f"PODS enabled: n={config.num_generations}, m={pods_keep_count}, "
              f"retain={pods_keep_count / config.num_generations:.1%}")

    def _compute_rewards(self, prompts, completions, mode, **kwargs):
        # super() includes RewardRankNoiseGRPOLearner in mismatch runs.
        rewards = super()._compute_rewards(
            prompts=prompts, completions=completions, mode=mode, **kwargs
        )
        if self._pods_capture:
            self._pods_observed = (np.asarray(rewards).copy(), list(prompts))
        return rewards

    def _generate_and_compute_advantage(self, training_input, mode):
        import jax
        from tunix.rl import rl_cluster as cluster_lib

        if mode != cluster_lib.Mode.TRAIN:
            return super()._generate_and_compute_advantage(training_input, mode)
        self._pods_capture = True
        self._pods_observed = None
        try:
            full = super()._generate_and_compute_advantage(training_input, mode)
            if self._pods_observed is None:
                raise RuntimeError("PODS did not capture the training rewards")
            rewards, prompts = self._pods_observed
        finally:
            self._pods_capture = False
            self._pods_observed = None
        n, m = self.algo_config.num_generations, self.pods_keep_count
        if len(rewards) != len(prompts) or len(rewards) % n:
            raise ValueError("PODS reward/prompt group size mismatch")
        grouped = rewards.reshape(-1, n)
        for group in range(len(grouped)):
            if len(set(prompts[group*n:(group+1)*n])) != 1:
                raise ValueError("PODS received a group containing different prompts")
        local = select_max_variance_indices(grouped, m)
        flat = (local + np.arange(len(grouped))[:, None] * n).reshape(-1)
        chosen = np.take_along_axis(grouped, local, axis=1)
        advantages = subset_advantages(chosen)
        if m == n:
            result = full
        else:
            result = jax.tree.map(lambda x: x[flat], full).replace(
                advantages=jax.device_put(advantages.reshape(-1))
            )
        self._record_pods(grouped, chosen, local, advantages, prompts)
        self.rl_cluster.buffer_metrics({
            "pods/generated_completions": (len(rewards), np.sum),
            "pods/selected_completions": (len(flat), np.sum),
            "pods/kept_fraction": (m / n, np.mean),
            "pods/selected_reward_variance": (chosen.var(axis=1).mean(), np.mean),
        }, mode=mode)
        return result

    def _process_accumulated_batches(self, micro_batches, micro_batch_sizes, mode):
        from tunix.rl import rl_cluster as cluster_lib, utils as rl_utils

        if mode != cluster_lib.Mode.TRAIN:
            return super()._process_accumulated_batches(micro_batches, micro_batch_sizes, mode)
        if not micro_batches:
            return []
        sizes = selected_micro_batch_sizes(
            micro_batch_sizes, self.algo_config.num_generations, self.pods_keep_count
        )
        merged = rl_utils.merge_micro_batches(micro_batches)
        if sum(micro_batch_sizes) != len(merged["prompts"]):
            raise ValueError("PODS accumulated micro-batch sizes do not match input")
        combined = self._generate_and_compute_advantage(merged, mode)
        if len(combined.advantages) != sum(sizes):
            raise ValueError("PODS selected batch size does not match prompt_count * m")
        produced, offset = [], 0
        for size in sizes:
            produced.append(rl_utils.get_batch_slice(combined, slice(offset, offset + size)))
            offset += size
        return produced

    def _record_pods(self, rewards, chosen, local, advantages, prompts):
        self._pods_chunks += 1
        self._pods_generated += rewards.size
        self._pods_selected += chosen.size
        if not self._pods_dir:
            return
        n = rewards.shape[1]
        noise = getattr(self, "noise_config", None)
        with (self._pods_dir / "pods_selection.jsonl").open("a", encoding="utf-8") as stream:
            for i, indices in enumerate(local):
                prompt = str(prompts[i*n])
                corrupted = False
                if noise is not None:
                    from .reward_rank_noise import selected_prompt
                    corrupted = selected_prompt(prompt, noise)
                record = {
                    "chunk": self._pods_chunks,
                    "update_step": int(self.rl_cluster.global_steps),
                    "prompt_id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    "group_in_chunk": i, "n": n, "m": self.pods_keep_count,
                    "mismatch_selected": corrupted,
                    "observed_rewards": rewards[i].tolist(),
                    "selected_indices": indices.tolist(),
                    "selected_rewards": chosen[i].tolist(),
                    "selected_advantages": advantages[i].tolist(),
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        summary = {
            "method": "pods", "chunks": self._pods_chunks,
            "generated_completions": self._pods_generated,
            "selected_completions": self._pods_selected,
            "retained_fraction": self._pods_selected / self._pods_generated,
            "num_generations": n, "keep_count": self.pods_keep_count,
            "counts_scope": "current process session; selection JSONL appends on resume",
        }
        temporary = self._pods_dir / "pods_summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self._pods_dir / "pods_summary.json")


def pods_learner_class(base):
    """Preserve the factory-selected clean or mismatch reward implementation."""
    class PODSLearner(PODSMixin, base):
        pass

    return PODSLearner
