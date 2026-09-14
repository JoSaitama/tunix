"""Lazy curricula that run selection against the current policy weights."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from .artifacts import SelectionArtifactWriter, prompt_id
from .data_utils import Example, batch_examples, cyclic_batches
from .scoring import (
    cosine_scores,
    learnability,
    learnalign_scores,
    stable_top_indices,
)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Dependency-free average ranks, sufficient for selector diagnostics."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        return float("nan")
    x = _rankdata(left)
    y = _rankdata(right)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


class _BaseCurriculum:
    def __init__(
        self,
        *,
        training_examples: Sequence[Example],
        train_batch_size: int,
        total_update_steps: int,
        estimator,
        writer: SelectionArtifactWriter,
        rl_cluster,
    ):
        if not training_examples:
            raise ValueError("training_examples must not be empty")
        self.examples = list(training_examples)
        self.train_batch_size = train_batch_size
        self.total_update_steps = total_update_steps
        self.estimator = estimator
        self.writer = writer
        self.rl_cluster = rl_cluster

    def _log(self, values: dict[str, float], step: int) -> None:
        for name, value in values.items():
            if not np.isfinite(value):
                continue
            self.rl_cluster.log_scalar_immediately(
                f"alignment/{name}", value, step=step
            )


class LearnAlignCurriculum(_BaseCurriculum):
    """Paper-style warmup followed by one static LearnAlign ranking."""

    def __init__(
        self,
        *,
        warmup_prompts: int,
        estimation_rollouts: int,
        selection_ratio: int,
        warmup_counts_toward_budget: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.warmup_prompts = min(warmup_prompts, len(self.examples))
        self.estimation_rollouts = estimation_rollouts
        self.selection_ratio = selection_ratio
        self.warmup_counts_toward_budget = warmup_counts_toward_budget
        self.warmup_steps = math.ceil(self.warmup_prompts / self.train_batch_size)
        if self.warmup_prompts % self.train_batch_size:
            raise ValueError("learnalign_warmup_prompts must divide train batch size")

    @property
    def effective_max_steps(self) -> int:
        if self.warmup_counts_toward_budget:
            return self.total_update_steps
        return self.total_update_steps + self.warmup_steps

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        warmup = self.examples[: self.warmup_prompts]
        yield from cyclic_batches(
            warmup,
            batch_size=self.train_batch_size,
            num_batches=self.warmup_steps,
        )

        estimate = self.estimator.estimate(
            self.examples,
            num_rollouts=self.estimation_rollouts,
            apply_mismatch=True,
        )
        values = learnability(estimate.selector_rewards)
        scores = learnalign_scores(estimate.features, values)
        selected_count = max(
            self.train_batch_size,
            len(self.examples) // self.selection_ratio,
        )
        selected_count -= selected_count % self.train_batch_size
        selected_indices = stable_top_indices(scores, selected_count)
        selected = [self.examples[index] for index in selected_indices]
        selected_set = set(int(index) for index in selected_indices)
        gradient_prompt_batch_size = int(
            getattr(
                self.estimator,
                "grouped_prompt_subbatch_size",
                self.train_batch_size,
            )
        )

        self.writer.write_records(
            {
                "method": "learnalign",
                "source_index": index,
                "prompt_id": prompt_id(str(example["prompts"])),
                "clean_binary_outcomes": [
                    int(value) for value in estimate.clean_binary_outcomes[index]
                ],
                "selector_rewards": [
                    int(value) for value in estimate.selector_rewards[index]
                ],
                "selector_advantages": [
                    float(value) for value in estimate.selector_advantages[index]
                ],
                "mismatch_selected": bool(estimate.mismatch_selected[index]),
                "mismatch_effective": bool(estimate.mismatch_effective[index]),
                "clean_success_rate": float(
                    estimate.clean_binary_outcomes[index].mean()
                ),
                "selector_success_rate": float(
                    estimate.selector_rewards[index].mean()
                ),
                "learnability": float(values[index]),
                "score": float(scores[index]),
                "selected": index in selected_set,
            }
            for index, example in enumerate(self.examples)
        )
        self.writer.write_summary(
            {
                "method": "learnalign",
                "paper_equations": ["V=p(1-p)", "Eq. 8 row-mean"],
                "candidate_prompts": len(self.examples),
                "selected_prompts": selected_count,
                "selection_ratio": self.selection_ratio,
                "estimation_rollouts": self.estimation_rollouts,
                "warmup_prompts": self.warmup_prompts,
                "warmup_update_steps": self.warmup_steps,
                "warmup_counts_toward_budget": self.warmup_counts_toward_budget,
                "effective_max_steps": self.effective_max_steps,
                "selector_reward": (
                    "binary_exact_correctness_with_optional_rank_reversal"
                ),
                "selector_mismatch_scope": "all_training_candidates",
                "training_reward": "frozen_dense_reward_with_optional_rank_mismatch",
                "projection": "deterministic_sparse_jl_feature_hash",
                "gradient_feature_prompt_batch_size": gradient_prompt_batch_size,
                "gradient_feature_completion_batch_size": (
                    gradient_prompt_batch_size * self.estimation_rollouts
                ),
                "gradient_feature_calls_per_rollout_chunk": math.ceil(
                    self.train_batch_size / gradient_prompt_batch_size
                ),
                "gradient_aggregation": "rollout_mean_loss_before_gradient",
            }
        )
        self._log(
            {
                "learnalign_selected_fraction": selected_count / len(self.examples),
                "learnalign_mean_learnability": float(values.mean()),
                "learnalign_nonzero_gradient_fraction": float(
                    np.mean(np.linalg.norm(estimate.features, axis=1) > 0)
                ),
                "learnalign_selector_mismatch_selected_fraction": float(
                    estimate.mismatch_selected.mean()
                ),
                "learnalign_selector_mismatch_effective_fraction": float(
                    estimate.mismatch_effective.mean()
                ),
            },
            step=self.warmup_steps,
        )

        selected_steps = self.total_update_steps
        if self.warmup_counts_toward_budget:
            selected_steps -= self.warmup_steps
        yield from cyclic_batches(
            selected,
            batch_size=self.train_batch_size,
            num_batches=max(0, selected_steps),
        )


class GradAlignCurriculum(_BaseCurriculum):
    """Online GradAlign rounds with a clean held-out validation direction."""

    def __init__(
        self,
        *,
        validation_examples: Sequence[Example],
        candidate_rollouts: int,
        validation_rollouts: int,
        selection_ratio: int,
        selection_interval: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not validation_examples:
            raise ValueError("GradAlign requires a non-empty validation set")
        self.validation_examples = list(validation_examples)
        self.candidate_rollouts = candidate_rollouts
        self.validation_rollouts = validation_rollouts
        self.selection_ratio = selection_ratio
        self.selection_interval = selection_interval

    @property
    def effective_max_steps(self) -> int:
        return self.total_update_steps

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        source_cursor = 0
        update_step = 0
        previous_scores: dict[str, float] = {}
        previous_selected: set[str] = set()
        summaries: list[dict[str, Any]] = []

        while update_step < self.total_update_steps:
            round_steps = min(
                self.selection_interval,
                self.total_update_steps - update_step,
            )
            selected_count = round_steps * self.train_batch_size
            candidate_count = selected_count * self.selection_ratio
            candidates = [
                self.examples[(source_cursor + offset) % len(self.examples)]
                for offset in range(candidate_count)
            ]
            source_indices = [
                (source_cursor + offset) % len(self.examples)
                for offset in range(candidate_count)
            ]
            source_cursor = (source_cursor + candidate_count) % len(self.examples)

            validation_estimate = self.estimator.estimate(
                self.validation_examples,
                num_rollouts=self.validation_rollouts,
                apply_mismatch=False,
            )
            validation_gradient = validation_estimate.features.mean(axis=0)
            candidate_estimate = self.estimator.estimate(
                candidates,
                num_rollouts=self.candidate_rollouts,
                apply_mismatch=True,
            )
            scores = cosine_scores(candidate_estimate.features, validation_gradient)
            selected_local = stable_top_indices(scores, selected_count)
            selected = [candidates[index] for index in selected_local]
            selected_local_set = set(int(index) for index in selected_local)

            ids = [prompt_id(str(example["prompts"])) for example in candidates]
            current_scores = dict(zip(ids, (float(score) for score in scores)))
            overlap = sorted(set(previous_scores) & set(current_scores))
            spearman = (
                _spearman(
                    np.asarray([previous_scores[key] for key in overlap]),
                    np.asarray([current_scores[key] for key in overlap]),
                )
                if overlap
                else float("nan")
            )
            current_selected = {ids[index] for index in selected_local}
            jaccard = (
                _jaccard(previous_selected, current_selected)
                if previous_selected
                else float("nan")
            )
            round_index = len(summaries)
            summary = {
                "round": round_index,
                "start_update_step": update_step,
                "candidate_prompts": candidate_count,
                "selected_prompts": selected_count,
                "mean_candidate_score": float(np.mean(scores)),
                "validation_success_rate": float(
                    validation_estimate.clean_binary_outcomes.mean()
                ),
                "candidate_clean_success_rate": float(
                    candidate_estimate.clean_binary_outcomes.mean()
                ),
                "candidate_selector_success_rate": float(
                    candidate_estimate.selector_rewards.mean()
                ),
                "candidate_mismatch_selected_fraction": float(
                    candidate_estimate.mismatch_selected.mean()
                ),
                "candidate_mismatch_effective_fraction": float(
                    candidate_estimate.mismatch_effective.mean()
                ),
                "overlap_score_spearman": _finite_or_none(spearman),
                "selected_set_jaccard": _finite_or_none(jaccard),
            }
            summaries.append(summary)
            self.writer.write_records(
                {
                    "method": "gradalign",
                    "record_type": "validation",
                    "round": round_index,
                    "start_update_step": update_step,
                    "prompt_id": prompt_id(str(example["prompts"])),
                    "clean_binary_outcomes": [
                        int(value)
                        for value in validation_estimate.clean_binary_outcomes[index]
                    ],
                    "selector_rewards": [
                        int(value)
                        for value in validation_estimate.selector_rewards[index]
                    ],
                    "selector_advantages": [
                        float(value)
                        for value in validation_estimate.selector_advantages[index]
                    ],
                    "mismatch_selected": False,
                    "mismatch_effective": False,
                    "clean_success_rate": float(
                        validation_estimate.clean_binary_outcomes[index].mean()
                    ),
                    "selector_success_rate": float(
                        validation_estimate.selector_rewards[index].mean()
                    ),
                }
                for index, example in enumerate(self.validation_examples)
            )
            self.writer.write_records(
                {
                    "method": "gradalign",
                    "record_type": "candidate",
                    "round": round_index,
                    "start_update_step": update_step,
                    "source_index": source_indices[index],
                    "prompt_id": ids[index],
                    "clean_binary_outcomes": [
                        int(value)
                        for value in candidate_estimate.clean_binary_outcomes[index]
                    ],
                    "selector_rewards": [
                        int(value)
                        for value in candidate_estimate.selector_rewards[index]
                    ],
                    "selector_advantages": [
                        float(value)
                        for value in candidate_estimate.selector_advantages[index]
                    ],
                    "mismatch_selected": bool(
                        candidate_estimate.mismatch_selected[index]
                    ),
                    "mismatch_effective": bool(
                        candidate_estimate.mismatch_effective[index]
                    ),
                    "clean_success_rate": float(
                        candidate_estimate.clean_binary_outcomes[index].mean()
                    ),
                    "selector_success_rate": float(
                        candidate_estimate.selector_rewards[index].mean()
                    ),
                    "cosine_score": float(scores[index]),
                    "selected": index in selected_local_set,
                }
                for index in range(candidate_count)
            )
            self._log(
                {
                    "gradalign_selected_fraction": selected_count / candidate_count,
                    "gradalign_mean_cosine": float(np.mean(scores)),
                    "gradalign_validation_success_rate": float(
                        validation_estimate.clean_binary_outcomes.mean()
                    ),
                    "gradalign_candidate_clean_success_rate": float(
                        candidate_estimate.clean_binary_outcomes.mean()
                    ),
                    "gradalign_candidate_selector_success_rate": float(
                        candidate_estimate.selector_rewards.mean()
                    ),
                    "gradalign_selector_mismatch_selected_fraction": float(
                        candidate_estimate.mismatch_selected.mean()
                    ),
                    "gradalign_selector_mismatch_effective_fraction": float(
                        candidate_estimate.mismatch_effective.mean()
                    ),
                    "gradalign_overlap_score_spearman": spearman,
                    "gradalign_selected_set_jaccard": jaccard,
                },
                step=update_step,
            )
            self.writer.write_summary(
                {
                    "method": "gradalign",
                    "paper_score": "cos(g_candidate, mean(g_validation))",
                    "selection_ratio": self.selection_ratio,
                    "selection_interval": self.selection_interval,
                    "candidate_rollouts": self.candidate_rollouts,
                    "validation_rollouts": self.validation_rollouts,
                    "validation_prompts": len(self.validation_examples),
                    "effective_max_steps": self.effective_max_steps,
                    "candidate_selector_reward": (
                        "binary_exact_correctness_with_optional_rank_reversal"
                    ),
                    "validation_selector_reward": (
                        "clean_binary_exact_correctness"
                    ),
                    "selector_mismatch_scope": "training_candidates_only",
                    "training_reward": (
                        "frozen_dense_reward_with_optional_rank_mismatch"
                    ),
                    "projection": "deterministic_sparse_jl_feature_hash",
                    "rounds": summaries,
                }
            )

            yield from cyclic_batches(
                selected,
                batch_size=self.train_batch_size,
                num_batches=round_steps,
            )
            update_step += round_steps
            previous_scores = current_scores
            previous_selected = current_selected
