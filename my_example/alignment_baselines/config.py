"""Configuration owned by the alignment baselines.

The existing ``my_example.config`` parser is deliberately left untouched.  This
module consumes only baseline-specific flags and forwards the remaining flags
to the frozen GSM8K GRPO parser.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class AlignmentBaselineConfig:
    method: str
    selection_ratio: int = 4
    projection_dim: int = 4096
    selection_micro_batch_size: int = 4
    learnalign_warmup_prompts: int = 300
    learnalign_estimation_rollouts: int = 8
    learnalign_warmup_counts_toward_budget: bool = True
    gradalign_validation_prompts: int = 30
    gradalign_candidate_rollouts: int = 4
    gradalign_validation_rollouts: int = 4
    gradalign_selection_interval: int = 10
    artifact_dir: str = "./runs_xuesong/alignment_baselines"

    def validate(self, *, train_batch_size: int) -> None:
        if self.method not in {"learnalign", "gradalign"}:
            raise ValueError(f"Unsupported alignment method: {self.method!r}")
        for name in (
            "selection_ratio",
            "projection_dim",
            "selection_micro_batch_size",
            "learnalign_warmup_prompts",
            "learnalign_estimation_rollouts",
            "gradalign_validation_prompts",
            "gradalign_candidate_rollouts",
            "gradalign_validation_rollouts",
            "gradalign_selection_interval",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "learnalign_estimation_rollouts",
            "gradalign_candidate_rollouts",
            "gradalign_validation_rollouts",
        ):
            if int(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be greater than one for GRPO")
        if self.selection_ratio <= 1:
            raise ValueError("selection_ratio must be greater than one")
        if self.selection_micro_batch_size % train_batch_size != 0:
            raise ValueError(
                "selection_micro_batch_size must be divisible by the frozen "
                f"training batch size ({train_batch_size})"
            )
        if self.learnalign_warmup_prompts % train_batch_size != 0:
            raise ValueError(
                "learnalign_warmup_prompts must be divisible by the frozen "
                f"training batch size ({train_batch_size})"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_alignment_args(
    argv: Sequence[str] | None = None,
) -> tuple[AlignmentBaselineConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--alignment-method",
        required=True,
        choices=("learnalign", "gradalign"),
    )
    parser.add_argument("--selection-ratio", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--selection-micro-batch-size", type=int, default=4)
    parser.add_argument("--learnalign-warmup-prompts", type=int, default=300)
    parser.add_argument("--learnalign-estimation-rollouts", type=int, default=8)
    parser.add_argument(
        "--learnalign-warmup-counts-toward-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gradalign-validation-prompts", type=int, default=30)
    parser.add_argument("--gradalign-candidate-rollouts", type=int, default=4)
    parser.add_argument("--gradalign-validation-rollouts", type=int, default=4)
    parser.add_argument("--gradalign-selection-interval", type=int, default=10)
    parser.add_argument(
        "--alignment-artifact-dir",
        default="./runs_xuesong/alignment_baselines",
    )
    args, remaining = parser.parse_known_args(argv)
    return (
        AlignmentBaselineConfig(
            method=args.alignment_method,
            selection_ratio=args.selection_ratio,
            projection_dim=args.projection_dim,
            selection_micro_batch_size=args.selection_micro_batch_size,
            learnalign_warmup_prompts=args.learnalign_warmup_prompts,
            learnalign_estimation_rollouts=args.learnalign_estimation_rollouts,
            learnalign_warmup_counts_toward_budget=(
                args.learnalign_warmup_counts_toward_budget
            ),
            gradalign_validation_prompts=args.gradalign_validation_prompts,
            gradalign_candidate_rollouts=args.gradalign_candidate_rollouts,
            gradalign_validation_rollouts=args.gradalign_validation_rollouts,
            gradalign_selection_interval=args.gradalign_selection_interval,
            artifact_dir=args.alignment_artifact_dir,
        ),
        remaining,
    )
