#!/usr/bin/env python3
"""Plot GRPO GSM8K DTV self/cross diagnostics from five LOO JSONL logs.

The input records are produced by Group Policy-DTV-LOO training.  They retain
enough gradient inner-product statistics to reconstruct the corresponding
ordinary Policy-DTV score without recomputing gradients:

  self_term  = raw_self / G
  cross_term = raw_cross_sum / G
  dtv_score  = self_term + cross_term
  loo_score  = raw_cross_sum / (G - 1)

All decision plots use the raw zero-score rules.  The minimum-retention cap and
its final mask are intentionally not part of these theoretical decision-region
figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


COLOR_SELF = "#1F5AA6"
COLOR_SELF_FILL = "#BDD7F7"
COLOR_DTV = "#F28E2B"
COLOR_DTV_FILL = "#FAD7A0"
COLOR_LOO = "#D62728"
COLOR_LOO_FILL = "#F5B5B5"
COLOR_DTV_DROP = "#017340"
COLOR_DARK_GRAY = "#4D4D4D"
COLOR_GREEN = "#2CA02C"
COLOR_YELLOW = "#E5AE00"

LINE_W = 2.0
MAIN_FIGSIZE = (5.9, 5.2)
PAPER_AXES_POSITION = [0.18, 0.20, 0.78, 0.76]
COVERAGE_AXES_POSITION = [0.18, 0.20, 0.66, 0.76]
LABEL_FS = 24
TICK_FS = 22
TITLE_FS = 22
LEGEND_FS = 19

ARRAY_FIELDS = (
    "loo_raw_self",
    "loo_raw_cross_sum",
    "loo_standard_self_term",
    "loo_standard_cross_term",
    "loo_standard_score",
    "loo_scores",
    "loo_group_indices",
    "loo_generation_indices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create GSM8K Group Policy-DTV/LOO self-protection figures from "
            "five selection JSONL files."
        )
    )
    parser.add_argument(
        "--selection-files",
        nargs=5,
        required=True,
        metavar=("SEED0", "SEED1", "SEED2", "SEED3", "SEED4"),
        help="Five Group Policy-DTV-LOO selection JSONL paths.",
    )
    parser.add_argument(
        "--seed-labels",
        nargs=5,
        default=None,
        metavar=("LABEL0", "LABEL1", "LABEL2", "LABEL3", "LABEL4"),
        help="Optional labels corresponding to --selection-files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--title",
        default="GSM8K | Group Policy-DTV-Loo | 5 seeds",
    )
    parser.add_argument("--sample-limit", type=int, default=50_000)
    parser.add_argument(
        "--scatter-quantile",
        type=float,
        default=0.995,
        help=(
            "Deprecated compatibility option; accepted but ignored. Use "
            "--decision-x-limits and --decision-y-limits."
        ),
    )
    parser.add_argument(
        "--score-y-limits",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help=(
            "Deprecated shared limits for both mean/std figures. Prefer the "
            "two figure-specific options below."
        ),
    )
    parser.add_argument(
        "--decomposition-y-limits",
        nargs=2,
        type=float,
        default=(-6.0, 35.0),
        metavar=("YMIN", "YMAX"),
        help="Explicit Figure 01 limits; overrides automatic quantile limits.",
    )
    parser.add_argument(
        "--decomposition-y-ticks",
        nargs="+",
        type=float,
        default=(-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
        help="Explicit Figure 01 y ticks.",
    )
    parser.add_argument(
        "--decomposition-lower-quantile",
        type=float,
        default=0.005,
        help="Automatic lower display quantile (default: 0.005).",
    )
    parser.add_argument(
        "--decomposition-upper-quantile",
        type=float,
        default=0.95,
        help="Automatic upper display quantile (default: 0.95).",
    )
    parser.add_argument(
        "--decomposition-band-mode",
        choices=("std", "sem", "none"),
        default="std",
        help="Figure 01 uncertainty band (default: std).",
    )
    parser.add_argument(
        "--completion-central-coverage",
        type=float,
        default=0.99,
        help=(
            "Global joint central coverage for the completion-level "
            "DTV/Self/Cross trend mask (default: 0.99)."
        ),
    )
    parser.add_argument(
        "--coverage-bin-size",
        type=int,
        default=50,
        help="Temporal bin width for Figure 01 retained-coverage markers.",
    )
    parser.add_argument(
        "--coverage-y-limits",
        nargs=2,
        type=float,
        default=(0.94, 1.00),
        metavar=("YMIN", "YMAX"),
        help="Right-axis retained-coverage limits.",
    )
    parser.add_argument(
        "--coverage-y-ticks",
        nargs="+",
        type=float,
        default=(0.94, 0.96, 0.98, 1.00),
        help="Right-axis retained-coverage ticks, expressed as fractions.",
    )
    parser.add_argument(
        "--coverage-alpha",
        type=float,
        default=0.75,
        help="Opacity of retained-coverage markers (default: 0.75).",
    )
    parser.add_argument(
        "--coverage-marker-max",
        type=float,
        default=0.99,
        help=(
            "Plot retained-coverage triangles only below this fraction "
            "(default: 0.99)."
        ),
    )
    parser.add_argument(
        "--show-coverage-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show/hide Figure 01 retained-coverage markers and right axis.",
    )
    parser.add_argument(
        "--conflict-central-coverage",
        type=float,
        default=0.99,
        help="Joint central coverage for active conflict completions.",
    )
    parser.add_argument(
        "--conflict-y-limits",
        nargs=2,
        type=float,
        default=(-10.0, 40.0),
        metavar=("YMIN", "YMAX"),
        help="Explicit Figure 04 limits; default uses conflict quantiles.",
    )
    parser.add_argument(
        "--conflict-y-ticks",
        nargs="+",
        type=float,
        default=(-10.0, 0.0, 10.0, 20.0, 30.0),
        help="Explicit Figure 04 y ticks.",
    )
    parser.add_argument(
        "--conflict-hide-above-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide conflict mean points above the displayed upper limit.",
    )
    parser.add_argument(
        "--conflict-overflow-bin-size",
        type=int,
        default=50,
        help="Step-bin width for conflict upper-overflow markers.",
    )
    parser.add_argument(
        "--conflict-overflow-marker-threshold",
        type=float,
        default=0.0,
        help="Minimum hidden-point fraction required to draw a yellow marker.",
    )
    parser.add_argument(
        "--conflict-band-mode",
        choices=("std", "sem", "none"),
        default="std",
        help="Figure 04 uncertainty band (default: std).",
    )
    parser.add_argument(
        "--decision-x-limits",
        nargs=2,
        type=float,
        default=(-250.0, 450.0),
        metavar=("XMIN", "XMAX"),
        help="Decision-region x-axis display limits (default: -250 450).",
    )
    parser.add_argument(
        "--decision-y-limits",
        nargs=2,
        type=float,
        default=(0.0, 1100.0),
        metavar=("YMIN", "YMAX"),
        help="Decision-region y-axis display limits (default: 0 1100).",
    )
    parser.add_argument(
        "--drop-bin-size",
        type=int,
        default=20,
        help="Number of training steps averaged into each drop-ratio bar.",
    )
    parser.add_argument(
        "--drop-y-limits",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Explicit drop-ratio limits; default chooses a non-clipping upper bound.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Centered moving-average width for plotted mean/std curves only; "
            "1 keeps the original per-step curves (default: 1)."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=0,
        help="Deterministic seed used only when limiting scatter points.",
    )
    args = parser.parse_args()
    if args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    if not 0.5 < args.scatter_quantile < 1.0:
        parser.error("--scatter-quantile must be in (0.5, 1.0)")
    if args.drop_bin_size <= 0:
        parser.error("--drop-bin-size must be positive")
    if args.coverage_bin_size <= 0:
        parser.error("--coverage-bin-size must be positive")
    if not 0.0 <= args.coverage_alpha <= 1.0:
        parser.error("--coverage-alpha must be in [0, 1]")
    if not 0.0 <= args.coverage_marker_max <= 1.0:
        parser.error("--coverage-marker-max must be in [0, 1]")
    if args.conflict_overflow_bin_size <= 0:
        parser.error("--conflict-overflow-bin-size must be positive")
    if not 0.0 <= args.conflict_overflow_marker_threshold <= 1.0:
        parser.error("--conflict-overflow-marker-threshold must be in [0, 1]")
    if not 0.0 < args.completion_central_coverage <= 1.0:
        parser.error("--completion-central-coverage must be in (0, 1]")
    if not 0.0 < args.conflict_central_coverage <= 1.0:
        parser.error("--conflict-central-coverage must be in (0, 1]")
    if not 0.0 <= args.decomposition_lower_quantile < 0.5:
        parser.error("--decomposition-lower-quantile must be in [0, 0.5)")
    if not 0.5 < args.decomposition_upper_quantile <= 1.0:
        parser.error("--decomposition-upper-quantile must be in (0.5, 1]")
    if args.decomposition_lower_quantile >= args.decomposition_upper_quantile:
        parser.error("decomposition lower quantile must be below upper quantile")
    if args.smooth_window <= 0:
        parser.error("--smooth-window must be positive")
    for option in (
        "score_y_limits",
        "decomposition_y_limits",
        "conflict_y_limits",
        "decision_x_limits",
        "decision_y_limits",
        "drop_y_limits",
        "coverage_y_limits",
    ):
        limits = getattr(args, option)
        if limits is None:
            continue
        if limits[0] >= limits[1]:
            parser.error(f"--{option.replace('_', '-')} requires MIN < MAX")
    return args


def configure_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": TICK_FS,
            "axes.labelsize": LABEL_FS,
            "axes.titlesize": TITLE_FS,
            "xtick.labelsize": TICK_FS,
            "ytick.labelsize": TICK_FS,
            "legend.fontsize": LEGEND_FS,
            "axes.linewidth": 1.0,
            "figure.dpi": 300,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8,
        labelsize=TICK_FS,
    )
    ax.xaxis.labelpad = 10
    ax.grid(True, alpha=0.20, linewidth=0.7)
    ax.set_box_aspect(0.90)
    ax.set_position(PAPER_AXES_POSITION)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def savefig(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[SAVE] {output}")


def _as_float_array(record: dict[str, Any], field: str, size: int) -> np.ndarray:
    value = record.get(field)
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(
            f"field {field!r} must be a list of length {size}; "
            f"got {type(value).__name__} length="
            f"{len(value) if isinstance(value, list) else 'n/a'}"
        )
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"field {field!r} contains NaN or Inf")
    return array


def _validate_identity(
    observed: np.ndarray,
    expected: np.ndarray,
    name: str,
    path: Path,
    line_number: int,
) -> None:
    if not np.allclose(observed, expected, rtol=2e-6, atol=1e-4):
        error = float(np.max(np.abs(observed - expected)))
        raise ValueError(
            f"{path}:{line_number}: inconsistent {name}; max error={error}"
        )


def load_selection_file(path: Path, seed_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_steps: set[int] = set()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error

            step = int(record.get("train_step", -1))
            if step <= 0 or step in seen_steps:
                raise ValueError(
                    f"{path}:{line_number}: invalid or duplicate train_step={step}"
                )
            seen_steps.add(step)

            if record.get("scope") != "group":
                raise ValueError(
                    f"{path}:{line_number}: expected scope='group'; "
                    f"got {record.get('scope')!r}"
                )
            if record.get("score_objective") != "policy":
                raise ValueError(
                    f"{path}:{line_number}: expected score_objective='policy'; "
                    f"got {record.get('score_objective')!r}"
                )

            group_size = int(record.get("num_generations", 0))
            if group_size != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected num_generations=4; "
                    f"got {group_size}"
                )
            batch_size = 16
            arrays = {
                field: _as_float_array(record, field, batch_size)
                for field in ARRAY_FIELDS
            }

            expected_groups = np.repeat(np.arange(4), group_size)
            expected_generations = np.tile(np.arange(group_size), 4)
            if not np.array_equal(
                arrays["loo_group_indices"].astype(np.int64), expected_groups
            ):
                raise ValueError(f"{path}:{line_number}: invalid 4x4 group indices")
            if not np.array_equal(
                arrays["loo_generation_indices"].astype(np.int64),
                expected_generations,
            ):
                raise ValueError(
                    f"{path}:{line_number}: invalid 4x4 generation indices"
                )

            raw_self = arrays["loo_raw_self"]
            raw_cross = arrays["loo_raw_cross_sum"]
            self_term = arrays["loo_standard_self_term"]
            cross_term = arrays["loo_standard_cross_term"]
            dtv_score = arrays["loo_standard_score"]
            loo_score = arrays["loo_scores"]

            _validate_identity(
                self_term,
                raw_self / group_size,
                "self_term = raw_self / G",
                path,
                line_number,
            )
            _validate_identity(
                cross_term,
                raw_cross / group_size,
                "cross_term = raw_cross_sum / G",
                path,
                line_number,
            )
            _validate_identity(
                dtv_score,
                (raw_self + raw_cross) / group_size,
                "dtv_score = (raw_self + raw_cross_sum) / G",
                path,
                line_number,
            )
            _validate_identity(
                loo_score,
                raw_cross / (group_size - 1),
                "loo_score = raw_cross_sum / (G - 1)",
                path,
                line_number,
            )

            for index in range(batch_size):
                rows.append(
                    {
                        "seed": seed_label,
                        "source_file": str(path),
                        "step": step,
                        "group_index": int(arrays["loo_group_indices"][index]),
                        "generation_index": int(
                            arrays["loo_generation_indices"][index]
                        ),
                        "self_term": float(self_term[index]),
                        "cross_term": float(cross_term[index]),
                        "dtv_score": float(dtv_score[index]),
                        "loo_score": float(loo_score[index]),
                        "active_gradient": bool(raw_self[index] > 0.0),
                        "kept_dtv": bool(dtv_score[index] >= 0.0),
                        "kept_loo": bool(loo_score[index] >= 0.0),
                    }
                )

    if not rows:
        raise ValueError(f"{path}: no JSONL records found")
    steps = sorted(seen_steps)
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError(f"{path}: train steps are not contiguous")

    print(
        f"[LOAD] seed={seed_label} file={path} "
        f"steps={len(steps)} range={steps[0]}-{steps[-1]} "
        f"samples={len(rows)}"
    )
    return pd.DataFrame(rows)


def validate_seed_alignment(samples: pd.DataFrame) -> None:
    step_sets = {
        str(seed): tuple(sorted(group["step"].unique()))
        for seed, group in samples.groupby("seed", sort=False)
    }
    reference_seed, reference_steps = next(iter(step_sets.items()))
    for seed, steps in step_sets.items():
        if steps != reference_steps:
            raise ValueError(
                "all five inputs must have identical train-step coverage; "
                f"seed {seed!r} differs from seed {reference_seed!r}"
            )

    counts = samples.groupby(["seed", "step"]).size()
    invalid = counts[counts != 16]
    if not invalid.empty:
        raise ValueError(
            "every seed/step must contain exactly 16 completion units; "
            f"first invalid entries: {invalid.head().to_dict()}"
        )


def build_per_seed_step(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        raise ValueError("no completion samples available for aggregation")
    per_step = (
        samples.groupby(["seed", "step"], as_index=False, sort=True)
        .agg(
            completion_count=("dtv_score", "size"),
            active_policy_gradient_count=("active_gradient", "sum"),
            dtv_mean=("dtv_score", "mean"),
            cross_mean=("cross_term", "mean"),
            self_mean=("self_term", "mean"),
            dtv_drop_ratio=("kept_dtv", lambda values: 1.0 - values.mean()),
            loo_drop_ratio=("kept_loo", lambda values: 1.0 - values.mean()),
            self_protected_count=(
                "dtv_score",
                lambda _: 0,
            ),
        )
    )

    conflicts = samples[ samples["kept_dtv"] & ~samples["kept_loo"] ]
    conflict_counts = conflicts.groupby(["seed", "step"]).size()
    index = pd.MultiIndex.from_frame(per_step[["seed", "step"]])
    per_step["self_protected_count"] = (
        conflict_counts.reindex(index, fill_value=0).to_numpy(dtype=np.int64)
    )
    per_step["self_protected_conflict_ratio"] = (
        per_step["self_protected_count"] / per_step["completion_count"]
    )
    return per_step


def aggregate_over_seeds(per_seed_step: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_seed_step.groupby("step", as_index=False, sort=True)
        .agg(
            contributing_seed_count=("seed", "nunique"),
            completion_count=("completion_count", "sum"),
            active_policy_gradient_count=("active_policy_gradient_count", "sum"),
            dtv_mean=("dtv_mean", "mean"),
            dtv_std=("dtv_mean", "std"),
            cross_mean=("cross_mean", "mean"),
            cross_std=("cross_mean", "std"),
            self_mean=("self_mean", "mean"),
            self_std=("self_mean", "std"),
            dtv_drop_ratio=("dtv_drop_ratio", "mean"),
            dtv_drop_ratio_std=("dtv_drop_ratio", "std"),
            loo_drop_ratio=("loo_drop_ratio", "mean"),
            loo_drop_ratio_std=("loo_drop_ratio", "std"),
            self_protected_conflict_ratio=(
                "self_protected_conflict_ratio",
                "mean",
            ),
            self_protected_conflict_ratio_std=(
                "self_protected_conflict_ratio",
                "std",
            ),
        )
        .sort_values("step")
    )
    std_columns = [column for column in grouped if column.endswith("_std")]
    grouped[std_columns] = grouped[std_columns].fillna(0.0)
    return grouped


def build_conflict_summary(
    samples: pd.DataFrame,
    coverage: float,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], float]:
    conflicts = samples[
        samples["active_gradient"] & samples["kept_dtv"] & ~samples["kept_loo"]
    ].copy()
    if conflicts.empty:
        raise ValueError("no self-protected conflict samples were found")
    bounds = completion_quantile_bounds(conflicts, coverage)
    conflicts = apply_threshold_faithful_trend_mask(conflicts, bounds)
    retained_fraction = float(conflicts["trend_inlier"].mean())
    conflicts = conflicts[conflicts["trend_inlier"]].copy()

    per_seed_step = (
        conflicts.groupby(["seed", "step"], as_index=False, sort=True)
        .agg(
            self_case_mean=("self_term", "mean"),
            cross_case_mean=("cross_term", "mean"),
            conflict_count=("self_term", "size"),
        )
    )
    # Missing seed/step cases stay missing. They are not imputed as zero.
    grouped = (
        per_seed_step.groupby("step", as_index=False, sort=True)
        .agg(
            self_mean=("self_case_mean", "mean"),
            self_std=("self_case_mean", "std"),
            cross_mean=("cross_case_mean", "mean"),
            cross_std=("cross_case_mean", "std"),
            seeds_with_conflicts=("seed", "nunique"),
            conflict_count=("conflict_count", "sum"),
        )
        .sort_values("step")
    )
    grouped[["self_std", "cross_std"]] = grouped[
        ["self_std", "cross_std"]
    ].fillna(0.0)
    return grouped, bounds, retained_fraction


def completion_quantile_bounds(
    samples: pd.DataFrame,
    coverage: float,
) -> dict[str, tuple[float, float]]:
    tail = (1.0 - coverage) / 2.0
    columns = {
        "dtv": "dtv_score",
        "self": "self_term",
        "cross": "cross_term",
    }
    return {
        component: (
            float(samples[column].quantile(tail)),
            float(samples[column].quantile(1.0 - tail)),
        )
        for component, column in columns.items()
    }


def apply_threshold_faithful_trend_mask(
    samples: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Apply one common central-coverage mask to complete score tuples."""
    result = samples.copy()
    columns = {"dtv": "dtv_score", "self": "self_term", "cross": "cross_term"}
    inlier_columns = []
    for component, column in columns.items():
        lower, upper = bounds[component]
        lower_name = f"trend_{component}_lower_outlier"
        upper_name = f"trend_{component}_upper_outlier"
        result[lower_name] = result[column] < lower
        result[upper_name] = result[column] > upper
        inlier_columns.extend([lower_name, upper_name])
    result["trend_inlier"] = ~result[inlier_columns].any(axis=1)
    return result


def build_coverage_summary(
    samples: pd.DataFrame,
    bin_size: int,
) -> pd.DataFrame:
    work = samples[["step", "trend_inlier"]].copy()
    work["coverage_bin"] = (work["step"].astype(int) - 1) // bin_size
    return (
        work.groupby("coverage_bin", as_index=False, sort=True)
        .agg(
            step_start=("step", "min"),
            step_end=("step", "max"),
            step_center=("step", "mean"),
            completion_count=("trend_inlier", "size"),
            retained_count=("trend_inlier", "sum"),
            retained_coverage=("trend_inlier", "mean"),
        )
    )


def _format_percent(fraction: float) -> str:
    value = 100.0 * fraction
    return f"{value:.1f}".rstrip("0").rstrip(".") + "%"


def _draw_mean_std(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    line_color: str,
    fill_color: str,
    label: str,
    zorder: int,
    y_limits: tuple[float, float] | None = None,
    band_mode: str = "std",
    sample_count: np.ndarray | None = None,
) -> None:
    # Input values have already been selected using the documented trend mask.
    uncertainty = std.copy()
    if band_mode == "sem":
        if sample_count is None:
            raise ValueError("SEM band requires sample counts")
        uncertainty = uncertainty / np.sqrt(np.maximum(sample_count, 1.0))

    if band_mode != "none":
        band_lower = mean - uncertainty
        band_upper = mean + uncertainty
        ax.fill_between(
            x,
            band_lower,
            band_upper,
            color=fill_color,
            alpha=0.35,
            linewidth=0,
            zorder=zorder - 1,
        )
    ax.plot(
        x,
        mean,
        color=line_color,
        linewidth=LINE_W,
        label=label,
        zorder=zorder,
    )


def _nice_quantile_limits(
    summary: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    columns = ("dtv_mean", "self_mean", "cross_mean")
    lower = min(float(summary[column].quantile(lower_quantile)) for column in columns)
    upper = max(float(summary[column].quantile(upper_quantile)) for column in columns)
    magnitude = max(abs(lower), abs(upper), 1.0)
    unit = 10.0 ** (math.floor(math.log10(magnitude)) - 1)
    padded_lower = math.floor(lower / unit) * unit
    padded_upper = math.ceil((1.05 * upper) / unit) * unit
    return float(padded_lower), float(padded_upper)


def _nice_conflict_limits(summary: pd.DataFrame) -> tuple[float, float]:
    columns = ("self_mean", "cross_mean")
    lower = min(float(summary[column].quantile(0.005)) for column in columns)
    upper = max(float(summary[column].quantile(0.95)) for column in columns)
    magnitude = max(abs(lower), abs(upper), 1.0)
    unit = 10.0 ** (math.floor(math.log10(magnitude)) - 1)
    return (
        float(math.floor(lower / unit) * unit),
        float(math.ceil((1.05 * upper) / unit) * unit),
    )


def smooth_for_display(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    window: int,
) -> pd.DataFrame:
    """Return centered rolling curves without changing raw exported statistics."""
    if window == 1:
        return frame
    smoothed = frame.copy()
    smoothed[list(columns)] = frame[list(columns)].rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean()
    return smoothed


def plot_score_decomposition(
    summary: pd.DataFrame,
    output_dir: Path,
    y_limits: tuple[float, float],
    smooth_window: int,
    band_mode: str,
    y_ticks: tuple[float, ...],
    coverage: pd.DataFrame,
    show_coverage_markers: bool,
    coverage_y_limits: tuple[float, float],
    coverage_y_ticks: tuple[float, ...],
    coverage_alpha: float,
    coverage_marker_max: float,
) -> None:
    summary = smooth_for_display(
        summary,
        ("dtv_mean", "dtv_std", "cross_mean", "cross_std", "self_mean", "self_std"),
        smooth_window,
    )
    x = summary["step"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    _draw_mean_std(
        ax,
        x,
        summary["dtv_mean"].to_numpy(),
        summary["dtv_std"].to_numpy(),
        COLOR_DTV,
        COLOR_DTV_FILL,
        "DTV",
        5,
        y_limits,
        band_mode,
        summary["contributing_seed_count"].to_numpy(dtype=np.float64),
    )
    _draw_mean_std(
        ax,
        x,
        summary["cross_mean"].to_numpy(),
        summary["cross_std"].to_numpy(),
        COLOR_LOO,
        COLOR_LOO_FILL,
        "Cross-term",
        6,
        y_limits,
        band_mode,
        summary["contributing_seed_count"].to_numpy(dtype=np.float64),
    )
    _draw_mean_std(
        ax,
        x,
        summary["self_mean"].to_numpy(),
        summary["self_std"].to_numpy(),
        COLOR_SELF,
        COLOR_SELF_FILL,
        "Self-term",
        4,
        y_limits,
        band_mode,
        summary["contributing_seed_count"].to_numpy(dtype=np.float64),
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Training step", fontsize=LABEL_FS)
    ax.set_ylabel("Score mean", labelpad=8, fontsize=LABEL_FS)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    if show_coverage_markers:
        ax.set_position(COVERAGE_AXES_POSITION)
        ax_right = ax.twinx()
        visible_coverage = coverage[
            coverage["retained_coverage"] < coverage_marker_max
        ]
        coverage_handle = ax_right.scatter(
            visible_coverage["step_center"],
            visible_coverage["retained_coverage"],
            marker="v",
            s=42,
            color=COLOR_GREEN,
            alpha=coverage_alpha,
            edgecolors="none",
            zorder=15,
            label="Retained coverage",
        )
        ax_right.set_ylim(*coverage_y_limits)
        ax_right.set_yticks(coverage_y_ticks)
        ax_right.set_yticklabels([f"{100.0 * value:g}%" for value in coverage_y_ticks])
        ax_right.set_ylabel(
            "Retained coverage (%)",
            color="black",
            fontsize=max(16, LABEL_FS - 5),
            labelpad=5,
        )
        ax_right.tick_params(
            axis="y",
            colors="black",
            direction="out",
            length=3,
            width=0.8,
            labelsize=max(15, TICK_FS - 5),
        )
        for side in ("left", "top", "bottom"):
            ax_right.spines[side].set_visible(False)
        ax_right.spines["right"].set_color("black")
        ax_right.spines["right"].set_linewidth(0.9)
        ax_right.set_box_aspect(0.90)
        ax_right.set_position(COVERAGE_AXES_POSITION)
        handles.append(coverage_handle)
        labels.append("Retained coverage")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FS,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.3,
        labelspacing=0.35,
        borderpad=0.0,
    )
    savefig(fig, output_dir / "01_score_decomposition_over_steps.png")


def plot_drop_ratio_story(
    summary: pd.DataFrame,
    output_dir: Path,
    bin_size: int,
    y_limits: tuple[float, float] | None,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    binned = summary.assign(
        step_bin=((summary["step"].astype(int) - 1) // bin_size)
    ).groupby("step_bin", as_index=False).agg(
        step=("step", "mean"),
        dtv_drop_ratio=("dtv_drop_ratio", "mean"),
        loo_drop_ratio=("loo_drop_ratio", "mean"),
    )
    x = binned["step"].to_numpy(dtype=np.float64)
    dtv_drop = binned["dtv_drop_ratio"].to_numpy(dtype=np.float64)
    loo_drop = binned["loo_drop_ratio"].to_numpy(dtype=np.float64)
    additional = np.clip(loo_drop - dtv_drop, 0.0, None)
    bar_width = max(0.82, 0.82 * bin_size)

    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    ax.bar(
        x,
        dtv_drop,
        width=bar_width,
        color=COLOR_DTV_DROP,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.20,
        label="DTV drop ratio",
    )
    ax.bar(
        x,
        additional,
        bottom=dtv_drop,
        width=bar_width,
        color=COLOR_LOO,
        alpha=0.62,
        edgecolor="white",
        linewidth=0.20,
        label="Additional DTV-Loo drop ratio",
    )
    ax.set_xlabel("Training step", fontsize=LABEL_FS)
    ax.set_ylabel("Drop ratio", labelpad=8, fontsize=LABEL_FS)
    if y_limits is None:
        observed = float(np.max(loo_drop))
        unit = 0.05
        y_limits = (0.0, math.ceil(observed / unit) * unit + unit)
    ax.set_ylim(*y_limits)
    style_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        frameon=False,
        fontsize=LEGEND_FS,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.35,
        borderpad=0.0,
    )
    savefig(fig, output_dir / "02_drop_ratio_story.png")
    binned.to_csv(output_dir / "grpo_dtv_drop_ratio_binned.csv", index=False)
    return binned, y_limits


def classify_decisions(samples: pd.DataFrame) -> pd.Series:
    kept_dtv = samples["kept_dtv"].to_numpy(dtype=bool)
    kept_loo = samples["kept_loo"].to_numpy(dtype=bool)
    labels = np.full(len(samples), "Other", dtype=object)
    labels[kept_dtv & kept_loo] = "Keep by both"
    labels[kept_dtv & ~kept_loo] = "DTV keeps / LOO drops"
    labels[~kept_dtv & ~kept_loo] = "Drop by both"
    labels[~kept_dtv & kept_loo] = "DTV drops / LOO keeps"
    return pd.Series(labels, index=samples.index)


def plot_decision_regions(
    samples: pd.DataFrame,
    output_dir: Path,
    sample_limit: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    sampling_seed: int,
) -> dict[str, float | int]:
    samples = samples.copy()
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    visible = samples[
        samples["cross_term"].between(x_min, x_max)
        & samples["self_term"].between(y_min, y_max)
    ].copy()
    omitted = len(samples) - len(visible)
    if len(visible) > sample_limit:
        visible = visible.sample(n=sample_limit, random_state=sampling_seed)
    visible["decision"] = classify_decisions(visible)

    order = (
        "Keep by both",
        "DTV keeps / LOO drops",
        "Drop by both",
        "DTV drops / LOO keeps",
    )
    label_map = {
        "Keep by both": "Both keep",
        "DTV keeps / LOO drops": "DTV keeps only",
        "Drop by both": "Both drop",
        "DTV drops / LOO keeps": "DTV-Loo keeps only",
    }
    color_map = {
        "Keep by both": COLOR_GREEN,
        "DTV keeps / LOO drops": COLOR_LOO,
        "Drop by both": COLOR_DARK_GRAY,
        "DTV drops / LOO keeps": "#9467BD",
    }

    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    for label in order:
        subset = visible[visible["decision"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["cross_term"],
            subset["self_term"],
            s=9,
            alpha=0.85,
            color=color_map[label],
            label=label_map[label],
            edgecolors="none",
            rasterized=True,
        )

    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.50)
    boundary_x = np.linspace(x_min, min(0.0, x_max), 256)
    boundary_y = -boundary_x
    boundary_visible = (boundary_y >= y_min) & (boundary_y <= y_max)
    ax.plot(
        boundary_x[boundary_visible],
        boundary_y[boundary_visible],
        color=COLOR_DTV,
        linewidth=2.0,
        linestyle="--",
        label="DTV boundary",
    )
    ax.axvline(
        0.0,
        color=COLOR_LOO,
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
        label="DTV-Loo boundary",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    x_tick_start = math.ceil(x_min / 200.0) * 200.0
    y_tick_start = math.ceil(y_min / 200.0) * 200.0
    ax.set_xticks(np.arange(x_tick_start, x_max, 200.0))
    ax.set_yticks(np.arange(y_tick_start, y_max, 200.0))
    ax.set_xlabel("Cross-term score", fontsize=LABEL_FS)
    ax.set_ylabel("Self-term score", labelpad=8, fontsize=LABEL_FS)
    style_axes(ax)
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.01, 1.01),
        ncol=1,
        frameon=False,
        fontsize=LEGEND_FS,
        markerscale=1.5,
        scatterpoints=1,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.35,
        borderpad=0.0,
    )
    for handle in getattr(
        legend,
        "legend_handles",
        getattr(legend, "legendHandles", []),
    ):
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([30])
            handle.set_alpha(0.95)
    savefig(fig, output_dir / "03_decision_region_scatter.png")
    print(
        "[SCATTER] "
        f"full={len(samples)} within_display_window={len(samples) - omitted} "
        f"plotted={len(visible)} omitted_as_axis_outliers={omitted} "
        f"xlim=({x_min:.6g}, {x_max:.6g}) "
        f"ylim=({y_min:.6g}, {y_max:.6g})"
    )
    return {
        "scatter_full_count": len(samples),
        "scatter_display_window_count": len(samples) - omitted,
        "scatter_plotted_count": len(visible),
        "scatter_axis_outlier_count": omitted,
        "scatter_x_min": x_min,
        "scatter_x_max": x_max,
        "scatter_y_min": y_min,
        "scatter_y_max": y_max,
    }


def plot_conflict_means(
    conflicts: pd.DataFrame,
    output_dir: Path,
    y_limits: tuple[float, float],
    smooth_window: int,
    band_mode: str,
    y_ticks: tuple[float, ...],
    hide_above_limit: bool,
    overflow_bin_size: int,
    overflow_marker_threshold: float,
) -> pd.DataFrame:
    conflicts = smooth_for_display(
        conflicts,
        ("self_mean", "self_std", "cross_mean", "cross_std"),
        smooth_window,
    )
    upper_limit = y_limits[1]
    overflow_work = conflicts[["step", "self_mean"]].copy()
    overflow_work["step_bin"] = (
        (overflow_work["step"].astype(int) - 1) // overflow_bin_size
    )
    overflow_work["above_upper_limit"] = (
        overflow_work["self_mean"] > upper_limit
    )
    overflow = (
        overflow_work.groupby("step_bin", as_index=False, sort=True)
        .agg(
            step_start=("step", "min"),
            step_end=("step", "max"),
            step_center=("step", "mean"),
            available_step_count=("above_upper_limit", "size"),
            above_upper_count=("above_upper_limit", "sum"),
            above_upper_fraction=("above_upper_limit", "mean"),
        )
    )
    plotted = conflicts.copy()
    if hide_above_limit:
        for prefix in ("self", "cross"):
            hidden = plotted[f"{prefix}_mean"] > upper_limit
            plotted.loc[hidden, [f"{prefix}_mean", f"{prefix}_std"]] = np.nan

    x = plotted["step"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    suffix = {
        "std": " mean ± 1 std",
        "sem": " mean ± 1 SEM",
        "none": " mean",
    }[band_mode]
    _draw_mean_std(
        ax,
        x,
        plotted["self_mean"].to_numpy(),
        plotted["self_std"].to_numpy(),
        COLOR_SELF,
        COLOR_SELF_FILL,
        "Self-term" + suffix,
        5,
        y_limits,
        band_mode,
        plotted["seeds_with_conflicts"].to_numpy(dtype=np.float64),
    )
    _draw_mean_std(
        ax,
        x,
        plotted["cross_mean"].to_numpy(),
        plotted["cross_std"].to_numpy(),
        COLOR_LOO,
        COLOR_LOO_FILL,
        "Cross-term" + suffix,
        6,
        y_limits,
        band_mode,
        plotted["seeds_with_conflicts"].to_numpy(dtype=np.float64),
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Training step", fontsize=LABEL_FS)
    ax.set_ylabel("Conflicted case score", labelpad=8, fontsize=LABEL_FS)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)
    style_axes(ax)
    marker_rows = overflow[
        overflow["above_upper_fraction"] > overflow_marker_threshold
    ]
    marker_handle = None
    if not marker_rows.empty:
        marker_y = upper_limit - 0.025 * (y_limits[1] - y_limits[0])
        marker_handle = ax.scatter(
            marker_rows["step_center"],
            np.full(len(marker_rows), marker_y),
            marker="^",
            s=48,
            color=COLOR_YELLOW,
            edgecolors="none",
            zorder=20,
            label=f"Self-term mean > {upper_limit:g}",
        )
        for row in marker_rows.itertuples(index=False):
            ax.annotate(
                _format_percent(float(row.above_upper_fraction)),
                (float(row.step_center), marker_y),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=max(10, TICK_FS - 10),
                color=COLOR_YELLOW,
                zorder=21,
            )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=1,
        frameon=False,
        fontsize=LEGEND_FS,
    )
    savefig(fig, output_dir / "04_self_protected_conflict_means.png")
    overflow.to_csv(
        output_dir / "grpo_dtv_conflict_upper_overflow_by_step_bin.csv",
        index=False,
    )
    return overflow


def plot_log_log_self_protection(
    samples: pd.DataFrame,
    output_dir: Path,
    sample_limit: int,
    sampling_seed: int,
) -> dict[str, float | int]:
    conflicts = samples[
        samples["active_gradient"] & (samples["cross_term"] < 0.0)
    ].copy()
    if conflicts.empty:
        raise ValueError("no active-gradient samples with negative Cross-term")
    conflicts["abs_cross_term"] = -conflicts["cross_term"]
    conflicts["self_to_abs_cross_ratio"] = (
        conflicts["self_term"] / conflicts["abs_cross_term"]
    )
    conflicts["decision"] = np.where(
        conflicts["dtv_score"] >= 0.0,
        "DTV keeps only",
        "Both drop",
    )
    plotted = conflicts
    if len(plotted) > sample_limit:
        plotted = plotted.sample(n=sample_limit, random_state=sampling_seed)

    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    for label, color in (
        ("DTV keeps only", COLOR_LOO),
        ("Both drop", COLOR_DARK_GRAY),
    ):
        subset = plotted[plotted["decision"] == label]
        ax.scatter(
            subset["abs_cross_term"],
            subset["self_term"],
            s=8,
            alpha=0.38,
            color=color,
            label=label,
            edgecolors="none",
            rasterized=True,
        )
    all_values = np.concatenate(
        [
            conflicts["abs_cross_term"].to_numpy(dtype=np.float64),
            conflicts["self_term"].to_numpy(dtype=np.float64),
        ]
    )
    boundary_min = 10.0 ** math.floor(math.log10(float(np.min(all_values))))
    boundary_max = 10.0 ** math.ceil(math.log10(float(np.max(all_values))))
    boundary = np.geomspace(boundary_min, boundary_max, 256)
    ax.plot(
        boundary,
        boundary,
        color=COLOR_DTV,
        linestyle="--",
        linewidth=2.0,
        label=r"DTV boundary: $S=|C|$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(boundary_min, boundary_max)
    ax.set_ylim(boundary_min, boundary_max)
    ax.set_xlabel("Absolute Cross-term magnitude", fontsize=LABEL_FS)
    ax.set_ylabel("Self-term score", labelpad=8, fontsize=LABEL_FS)
    style_axes(ax)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FS,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.35,
        borderpad=0.0,
    )
    savefig(fig, output_dir / "05_log_log_self_protection_scatter.png")
    conflicts.to_csv(
        output_dir / "grpo_dtv_negative_cross_log_scatter_samples.csv",
        index=False,
    )
    ratios = conflicts["self_to_abs_cross_ratio"].to_numpy(dtype=np.float64)
    return {
        "negative_cross_active_samples": int(len(conflicts)),
        "log_scatter_plotted_samples": int(len(plotted)),
        "self_to_abs_cross_ratio_median": float(np.quantile(ratios, 0.50)),
        "self_to_abs_cross_ratio_p90": float(np.quantile(ratios, 0.90)),
        "self_to_abs_cross_ratio_p95": float(np.quantile(ratios, 0.95)),
        "self_to_abs_cross_ratio_p99": float(np.quantile(ratios, 0.99)),
        "ratio_above_1_fraction": float(np.mean(ratios > 1.0)),
        "ratio_above_10_fraction": float(np.mean(ratios > 10.0)),
        "ratio_above_100_fraction": float(np.mean(ratios > 100.0)),
        "ratio_above_1000_fraction": float(np.mean(ratios > 1000.0)),
    }


def write_validation_report(
    samples: pd.DataFrame,
    selection_files: list[Path],
    seed_labels: list[str],
    scatter_report: dict[str, float | int],
    log_scatter_report: dict[str, float | int],
    plot_config: dict[str, Any],
    output_dir: Path,
) -> None:
    decisions = {
        "both_keep": int(
            (samples["kept_dtv"] & samples["kept_loo"]).sum()
        ),
        "dtv_keeps_only": int(
            (samples["kept_dtv"] & ~samples["kept_loo"]).sum()
        ),
        "both_drop": int(
            (~samples["kept_dtv"] & ~samples["kept_loo"]).sum()
        ),
        "loo_keeps_only": int(
            (~samples["kept_dtv"] & samples["kept_loo"]).sum()
        ),
    }
    report = {
        "selection_files": [str(path) for path in selection_files],
        "seed_labels": seed_labels,
        "num_seeds": int(samples["seed"].nunique()),
        "steps_per_seed": {
            str(seed): int(group["step"].nunique())
            for seed, group in samples.groupby("seed", sort=False)
        },
        "samples_per_seed": {
            str(seed): int(len(group))
            for seed, group in samples.groupby("seed", sort=False)
        },
        "decision_counts_raw_threshold": decisions,
        "active_gradient_samples": int(samples["active_gradient"].sum()),
        "inactive_zero_gradient_samples": int((~samples["active_gradient"]).sum()),
        **scatter_report,
        **log_scatter_report,
        "plot_config": plot_config,
        "notes": [
            "Threshold-faithful statistics include finite exact-zero scores.",
            "Trend means/std use one common joint central DTV/Self/Cross mask.",
            "Fixed score/scatter axis limits affect display only.",
            "Drop-ratio bars average adjacent steps for display only.",
            "Optional centered smoothing affects plotted mean/std curves only.",
            "Drop ratios and decision counts use untrimmed threshold decisions.",
            "LOO decisions use the raw zero-score threshold, not the final cap mask.",
            "No advantage or reward fields are loaded or analyzed.",
        ],
    }
    path = output_dir / "input_validation_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SAVE] {path}")


def write_overall_summary(samples: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for seed, group in samples.groupby("seed", sort=False):
        total_samples = len(group)
        rows.append(
            {
                "seed": seed,
                "total_samples": total_samples,
                "active_gradient_samples": int(group["active_gradient"].sum()),
                "inactive_zero_gradient_samples": int((~group["active_gradient"]).sum()),
                "self_mean": group["self_term"].mean(),
                "cross_mean": group["cross_term"].mean(),
                "dtv_mean": group["dtv_score"].mean(),
                "loo_mean": group["loo_score"].mean(),
                "dtv_drop_ratio": 1.0 - group["kept_dtv"].mean(),
                "loo_drop_ratio": 1.0 - group["kept_loo"].mean(),
                "self_protected_conflict_ratio": (
                    group["kept_dtv"] & ~group["kept_loo"]
                ).mean(),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "grpo_dtv_diagnostic_summary.csv", index=False
    )


def write_distribution_statistics(
    samples: pd.DataFrame,
    trend_samples: pd.DataFrame,
    trend_per_seed_step: pd.DataFrame,
    trend_summary: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    output_dir: Path,
) -> None:
    quantiles = {
        "p01": 0.01,
        "p02_5": 0.025,
        "p05": 0.05,
        "p50": 0.50,
        "p90": 0.90,
        "p95": 0.95,
        "p97_5": 0.975,
        "p98": 0.98,
        "p99": 0.99,
    }

    def describe(frame: pd.DataFrame, columns: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {"count": int(len(frame)), "components": {}}
        for component, column in columns.items():
            values = frame[column].to_numpy(dtype=np.float64)
            component_stats = {
                "min": float(np.min(values)),
                **{
                    name: float(np.quantile(values, quantile))
                    for name, quantile in quantiles.items()
                },
                "max": float(np.max(values)),
            }
            result["components"][component] = component_stats
        return result

    report = {
        "completion_level_threshold_population": describe(
            samples,
            {
                "dtv": "dtv_score",
                "self": "self_term",
                "cross": "cross_term",
            },
        ),
        "trend_quantile_bounds": bounds,
        "trend_retained_fraction": float(len(trend_samples) / len(samples)),
        "completion_level_trend_inliers": describe(
            trend_samples,
            {"dtv": "dtv_score", "self": "self_term", "cross": "cross_term"},
        ),
        "per_seed_step_trend_mean": describe(
            trend_per_seed_step,
            {
                "dtv": "dtv_mean",
                "self": "self_mean",
                "cross": "cross_mean",
            },
        ),
        "available_seed_aggregated_step_mean": describe(
            trend_summary,
            {
                "dtv": "dtv_mean",
                "self": "self_mean",
                "cross": "cross_mean",
            },
        ),
    }
    path = output_dir / "grpo_dtv_distribution_statistics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SAVE] {path}")


def main() -> None:
    args = parse_args()
    configure_style()

    selection_files = [Path(value).expanduser().resolve() for value in args.selection_files]
    missing = [str(path) for path in selection_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing selection files: " + ", ".join(missing))
    seed_labels = (
        [str(value) for value in args.seed_labels]
        if args.seed_labels is not None
        else [str(index) for index in range(len(selection_files))]
    )
    if len(set(seed_labels)) != len(seed_labels):
        raise ValueError("--seed-labels must be unique")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        load_selection_file(path, seed_label)
        for path, seed_label in zip(selection_files, seed_labels)
    ]
    samples = pd.concat(frames, ignore_index=True)
    validate_seed_alignment(samples)

    completion_bounds = completion_quantile_bounds(
        samples, args.completion_central_coverage
    )
    samples = apply_threshold_faithful_trend_mask(samples, completion_bounds)
    trend_samples = samples[samples["trend_inlier"]].copy()

    threshold_per_seed_step = build_per_seed_step(samples)
    threshold_summary = aggregate_over_seeds(threshold_per_seed_step)
    trend_per_seed_step = build_per_seed_step(trend_samples)
    trend_summary = aggregate_over_seeds(trend_per_seed_step)
    conflict_summary, conflict_bounds, conflict_retained_fraction = (
        build_conflict_summary(samples, args.conflict_central_coverage)
    )
    coverage_summary = build_coverage_summary(samples, args.coverage_bin_size)

    # CSVs retain every finite value; scatter display clipping is not applied.
    samples.to_csv(output_dir / "grpo_dtv_decomposition_samples.csv", index=False)
    threshold_per_seed_step.to_csv(
        output_dir / "grpo_dtv_threshold_per_seed_step.csv", index=False
    )
    threshold_summary.to_csv(
        output_dir / "grpo_dtv_threshold_per_step.csv", index=False
    )
    trend_per_seed_step.to_csv(
        output_dir / "grpo_dtv_decomposition_per_seed_step.csv", index=False
    )
    trend_summary.to_csv(
        output_dir / "grpo_dtv_decomposition_per_step.csv", index=False
    )
    conflict_summary.to_csv(
        output_dir / "grpo_dtv_self_protected_conflicts_per_step.csv", index=False
    )
    coverage_summary.to_csv(
        output_dir / "grpo_dtv_retained_coverage_by_step_bin.csv", index=False
    )
    write_overall_summary(samples, output_dir)
    write_distribution_statistics(
        samples,
        trend_samples,
        trend_per_seed_step,
        trend_summary,
        completion_bounds,
        output_dir,
    )

    if args.score_y_limits is not None:
        decomposition_y_limits = tuple(args.score_y_limits)
        conflict_y_limits = tuple(args.score_y_limits)
        print(
            "[WARN] --score-y-limits is deprecated; use "
            "--decomposition-y-limits and --conflict-y-limits"
        )
    else:
        decomposition_y_limits = (
            tuple(args.decomposition_y_limits)
            if args.decomposition_y_limits is not None
            else _nice_quantile_limits(
                trend_summary,
                args.decomposition_lower_quantile,
                args.decomposition_upper_quantile,
            )
        )
        conflict_y_limits = (
            tuple(args.conflict_y_limits)
            if args.conflict_y_limits is not None
            else _nice_conflict_limits(conflict_summary)
        )
    decision_x_limits = tuple(args.decision_x_limits)
    decision_y_limits = tuple(args.decision_y_limits)
    drop_y_limits = (
        tuple(args.drop_y_limits) if args.drop_y_limits is not None else None
    )
    print(
        "[DISPLAY] decomposition ylim="
        f"({decomposition_y_limits[0]:.6g}, {decomposition_y_limits[1]:.6g})"
    )
    print(
        "[TREND] central coverage diagnostics="
        f"{args.completion_central_coverage:.3f}; "
        f"joint DTV/Self/Cross retained={len(trend_samples) / len(samples):.3%}; "
        f"bounds={completion_bounds}"
    )
    plot_score_decomposition(
        trend_summary,
        output_dir,
        decomposition_y_limits,
        args.smooth_window,
        args.decomposition_band_mode,
        tuple(args.decomposition_y_ticks),
        coverage_summary,
        args.show_coverage_markers,
        tuple(args.coverage_y_limits),
        tuple(args.coverage_y_ticks),
        args.coverage_alpha,
        args.coverage_marker_max,
    )
    _, actual_drop_y_limits = plot_drop_ratio_story(
        threshold_summary,
        output_dir,
        args.drop_bin_size,
        drop_y_limits,
    )
    scatter_report = plot_decision_regions(
        samples,
        output_dir,
        args.sample_limit,
        decision_x_limits,
        decision_y_limits,
        args.sampling_seed,
    )
    plot_conflict_means(
        conflict_summary,
        output_dir,
        conflict_y_limits,
        args.smooth_window,
        args.conflict_band_mode,
        tuple(args.conflict_y_ticks),
        args.conflict_hide_above_limit,
        args.conflict_overflow_bin_size,
        args.conflict_overflow_marker_threshold,
    )
    log_scatter_report = plot_log_log_self_protection(
        samples,
        output_dir,
        args.sample_limit,
        args.sampling_seed,
    )
    plot_config = {
        "decomposition_ymin": decomposition_y_limits[0],
        "decomposition_ymax": decomposition_y_limits[1],
        "decomposition_lower_quantile": args.decomposition_lower_quantile,
        "decomposition_upper_quantile": args.decomposition_upper_quantile,
        "decomposition_band_mode": args.decomposition_band_mode,
        "completion_central_coverage": args.completion_central_coverage,
        "completion_quantile_bounds": completion_bounds,
        "trend_retained_fraction": len(trend_samples) / len(samples),
        "decomposition_y_ticks": list(args.decomposition_y_ticks),
        "coverage_bin_size": args.coverage_bin_size,
        "coverage_y_limits": list(args.coverage_y_limits),
        "coverage_y_ticks": list(args.coverage_y_ticks),
        "coverage_alpha": args.coverage_alpha,
        "coverage_marker_max": args.coverage_marker_max,
        "show_coverage_markers": args.show_coverage_markers,
        "conflict_ymin": conflict_y_limits[0],
        "conflict_ymax": conflict_y_limits[1],
        "conflict_band_mode": args.conflict_band_mode,
        "conflict_central_coverage": args.conflict_central_coverage,
        "conflict_y_ticks": list(args.conflict_y_ticks),
        "conflict_hide_above_limit": args.conflict_hide_above_limit,
        "conflict_overflow_bin_size": args.conflict_overflow_bin_size,
        "conflict_overflow_marker_threshold": (
            args.conflict_overflow_marker_threshold
        ),
        "conflict_quantile_bounds": conflict_bounds,
        "conflict_retained_fraction": conflict_retained_fraction,
        "drop_bin_size": args.drop_bin_size,
        "drop_ymin": actual_drop_y_limits[0],
        "drop_ymax": actual_drop_y_limits[1],
        "smooth_window": args.smooth_window,
    }
    write_validation_report(
        samples,
        selection_files,
        seed_labels,
        scatter_report,
        log_scatter_report,
        plot_config,
        output_dir,
    )
    print(f"[DONE] figures and diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
