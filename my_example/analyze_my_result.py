#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def _read_csv_points(path: Path) -> list[ScalarPoint]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty csv")

        points: list[ScalarPoint] = []
        for row in reader:
            if row.get("step") is None or row.get("value") is None:
                continue
            step = int(float(row["step"]))
            value = float(row["value"])
            wall_time = float(row.get("wall_time", 0.0) or 0.0)
            points.append(ScalarPoint(step=step, value=value, wall_time=wall_time))

    points.sort(key=lambda p: p.step)
    return points


def _last_point(points: Iterable[ScalarPoint]) -> ScalarPoint:
    last: ScalarPoint | None = None
    for p in points:
        if last is None or p.step > last.step:
            last = p
    if last is None:
        raise ValueError("no points")
    return last


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), None
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _variant_from_label(label: str) -> str:
    return label.split("__", 1)[0]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _get_metric(
    accuracy_meta: dict[str, Any] | None, phase: str, key: str
) -> float | int | None:
    if not accuracy_meta:
        return None
    phase_obj = accuracy_meta.get(phase)
    if not isinstance(phase_obj, dict):
        return None
    val = phase_obj.get(key)
    if isinstance(val, (int, float)):
        return val
    return None


def _padded_limits(values: list[float]) -> tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    span = vmax - vmin
    if span <= 0:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    pad = span * 0.05
    return vmin - pad, vmax + pad


def _plot_overlay_png(
    out_path: Path,
    series: list[tuple[str, list[ScalarPoint]]],
    title: str,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PIL is required to write overlay png") from exc

    width, height = 1200, 600
    legend_line_h = 14
    margin_l, margin_r, margin_t = 90, 30, 50
    margin_b = max(70, 30 + len(series) * legend_line_h)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_steps = [float(p.step) for _, pts in series for p in pts]
    all_vals = [float(p.value) for _, pts in series for p in pts]
    x0, x1 = _padded_limits(all_steps)
    y0, y1 = _padded_limits(all_vals)

    def x_map(step: float) -> float:
        if x1 == x0:
            return float(margin_l)
        return margin_l + (step - x0) / (x1 - x0) * plot_w

    def y_map(val: float) -> float:
        if y1 == y0:
            return float(height - margin_b)
        return margin_t + (y1 - val) / (y1 - y0) * plot_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    axis_color = (40, 40, 40)
    grid_color = (220, 220, 220)
    draw.line([(margin_l, margin_t), (margin_l, height - margin_b)], fill=axis_color, width=2)
    draw.line([(margin_l, height - margin_b), (width - margin_r, height - margin_b)], fill=axis_color, width=2)

    for i in range(6):
        y = margin_t + i / 5.0 * plot_h
        draw.line([(margin_l, y), (width - margin_r, y)], fill=grid_color, width=1)

    draw.text((margin_l, 15), title, fill=(0, 0, 0), font=font)

    colors = [
        (36, 99, 235),
        (234, 88, 12),
        (22, 163, 74),
        (147, 51, 234),
        (220, 38, 38),
        (20, 184, 166),
        (234, 179, 8),
        (100, 116, 139),
        (219, 39, 119),
        (59, 130, 246),
    ]
    legend_x = margin_l
    legend_y = height - margin_b + 15

    for idx, (name, pts) in enumerate(series):
        if not pts:
            continue
        color = colors[idx % len(colors)]
        xy = [(x_map(p.step), y_map(p.value)) for p in pts]
        if len(xy) == 1:
            x, y = xy[0]
            r = 3
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, fill=color)
        else:
            draw.line(xy, fill=color, width=3, joint="curve")
            for x, y in xy:
                r = 2
                draw.ellipse([x - r, y - r, x + r, y + r], outline=color, fill=color)

        label = f"{name} (last step={pts[-1].step}, value={pts[-1].value:.4f})"
        draw.rectangle(
            [legend_x, legend_y + idx * legend_line_h + 3, legend_x + 10, legend_y + idx * legend_line_h + 11],
            fill=color,
        )
        draw.text((legend_x + 14, legend_y + idx * legend_line_h), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _plot_variant_mean_std_band_png(
    out_path: Path,
    variants: list[tuple[str, list[list[ScalarPoint]]]],
    title: str,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PIL is required to write overlay png") from exc

    width, height = 1200, 600
    legend_line_h = 14
    margin_l, margin_r, margin_t = 90, 30, 50
    margin_b = max(70, 30 + len(variants) * legend_line_h)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    variant_stats: list[tuple[str, int, list[int], list[float], list[float]]] = []
    all_steps: list[float] = []
    all_vals: list[float] = []

    for variant, runs in variants:
        step_to_vals: dict[int, list[float]] = {}
        for pts in runs:
            for p in pts:
                step_to_vals.setdefault(p.step, []).append(float(p.value))
        steps = sorted(step_to_vals)
        means: list[float] = []
        stds: list[float] = []
        for step in steps:
            vals = step_to_vals[step]
            mean = float(statistics.mean(vals)) if vals else 0.0
            std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
            means.append(mean)
            stds.append(std)
            all_steps.append(float(step))
            all_vals.append(mean + std)
            all_vals.append(mean - std)
        if steps:
            variant_stats.append((variant, len(runs), steps, means, stds))

    if not variant_stats:
        raise ValueError("no data to plot")

    x0, x1 = _padded_limits(all_steps)
    y0, y1 = _padded_limits(all_vals)

    def x_map(step: float) -> float:
        if x1 == x0:
            return float(margin_l)
        return margin_l + (step - x0) / (x1 - x0) * plot_w

    def y_map(val: float) -> float:
        if y1 == y0:
            return float(height - margin_b)
        return margin_t + (y1 - val) / (y1 - y0) * plot_h

    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    axis_color = (40, 40, 40, 255)
    grid_color = (220, 220, 220, 255)
    draw.line([(margin_l, margin_t), (margin_l, height - margin_b)], fill=axis_color, width=2)
    draw.line([(margin_l, height - margin_b), (width - margin_r, height - margin_b)], fill=axis_color, width=2)

    for i in range(6):
        y = margin_t + i / 5.0 * plot_h
        draw.line([(margin_l, y), (width - margin_r, y)], fill=grid_color, width=1)

    draw.text((margin_l, 15), title, fill=(0, 0, 0, 255), font=font)

    colors = [
        (36, 99, 235),
        (234, 88, 12),
        (22, 163, 74),
        (147, 51, 234),
        (220, 38, 38),
        (20, 184, 166),
        (234, 179, 8),
        (100, 116, 139),
        (219, 39, 119),
        (59, 130, 246),
    ]
    legend_x = margin_l
    legend_y = height - margin_b + 15

    for idx, (variant, n_runs, steps, means, stds) in enumerate(variant_stats):
        color = colors[idx % len(colors)]
        line_color = (color[0], color[1], color[2], 255)
        band_color = (color[0], color[1], color[2], 55)

        upper_xy = [(x_map(float(s)), y_map(m + st)) for s, m, st in zip(steps, means, stds)]
        lower_xy = [(x_map(float(s)), y_map(m - st)) for s, m, st in zip(steps, means, stds)]

        if len(upper_xy) >= 2 and len(lower_xy) >= 2:
            draw.polygon(upper_xy + list(reversed(lower_xy)), fill=band_color)

        mean_xy = [(x_map(float(s)), y_map(m)) for s, m in zip(steps, means)]
        if len(mean_xy) == 1:
            x, y = mean_xy[0]
            r = 3
            draw.ellipse([x - r, y - r, x + r, y + r], outline=line_color, fill=line_color)
        else:
            draw.line(mean_xy, fill=line_color, width=3, joint="curve")

        label = f"{variant} (n={n_runs}, last step={steps[-1]}, mean={means[-1]:.4f}, std={stds[-1]:.4f})"
        draw.rectangle(
            [legend_x, legend_y + idx * legend_line_h + 3, legend_x + 10, legend_y + idx * legend_line_h + 11],
            fill=line_color,
        )
        draw.text((legend_x + 14, legend_y + idx * legend_line_h), label, fill=(0, 0, 0, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent / "my result"),
        help="Directory containing exported CSV/JSON artifacts",
    )
    parser.add_argument("--since-epoch", type=float, default=None, help="Only include files newer than this epoch")
    parser.add_argument("--output-prefix", required=True, help="Prefix for summary files")
    parser.add_argument("--tag", default="global/eval/rewards/sum", help="Scalar tag name (used for filename suffix)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"--results-dir does not exist: {results_dir}")

    tag_slug = _sanitize(str(args.tag))
    suffix = f"__{tag_slug}.csv"

    since = float(args.since_epoch) if args.since_epoch is not None else None

    reward_csvs: list[Path] = []
    for p in results_dir.glob(f"*{suffix}"):
        if since is not None and p.stat().st_mtime < since:
            continue
        reward_csvs.append(p)
    reward_csvs.sort(key=lambda p: p.stat().st_mtime)

    runs: list[dict[str, Any]] = []
    series_for_plot: list[tuple[str, list[ScalarPoint]]] = []
    variant_to_runs: dict[str, list[list[ScalarPoint]]] = {}

    for csv_path in reward_csvs:
        label = csv_path.name[: -len(suffix)]
        variant = _variant_from_label(label)
        points = _read_csv_points(csv_path)
        last = _last_point(points)

        acc_meta_path = results_dir / f"{label}__eval_accuracy__meta.json"
        if since is not None and acc_meta_path.exists() and acc_meta_path.stat().st_mtime < since:
            acc_meta_path = Path("__missing__")
        acc_meta = _load_json(acc_meta_path) if acc_meta_path.exists() else None

        runs.append(
            {
                "variant": variant,
                "label": label,
                "reward_last_step": last.step,
                "reward_last_value": last.value,
                "reward_num_points": len(points),
                "reward_csv": str(csv_path),
                "pre_accuracy": _get_metric(acc_meta, "pre-train", "accuracy"),
                "post_accuracy": _get_metric(acc_meta, "post-train", "accuracy"),
                "pre_partial_accuracy": _get_metric(acc_meta, "pre-train", "partial_accuracy"),
                "post_partial_accuracy": _get_metric(acc_meta, "post-train", "partial_accuracy"),
                "pre_format_accuracy": _get_metric(acc_meta, "pre-train", "format_accuracy"),
                "post_format_accuracy": _get_metric(acc_meta, "post-train", "format_accuracy"),
            }
        )
        series_for_plot.append((label, points))
        variant_to_runs.setdefault(variant, []).append(points)

    if not runs:
        raise SystemExit(f"no matching runs found in {results_dir} (suffix={suffix!r})")

    variants = sorted({r["variant"] for r in runs})
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        subset = [r for r in runs if r["variant"] == variant]
        reward_last_values = [float(r["reward_last_value"]) for r in subset if r["reward_last_value"] is not None]
        post_acc = [float(r["post_accuracy"]) for r in subset if r["post_accuracy"] is not None]
        post_partial = [float(r["post_partial_accuracy"]) for r in subset if r["post_partial_accuracy"] is not None]
        post_format = [float(r["post_format_accuracy"]) for r in subset if r["post_format_accuracy"] is not None]

        reward_mean, reward_std = _mean_std(reward_last_values)
        acc_mean, acc_std = _mean_std(post_acc)
        partial_mean, partial_std = _mean_std(post_partial)
        format_mean, format_std = _mean_std(post_format)

        summaries.append(
            {
                "variant": variant,
                "n_runs": len(subset),
                "reward_last_mean": reward_mean,
                "reward_last_std": reward_std,
                "post_accuracy_mean": acc_mean,
                "post_accuracy_std": acc_std,
                "post_partial_accuracy_mean": partial_mean,
                "post_partial_accuracy_std": partial_std,
                "post_format_accuracy_mean": format_mean,
                "post_format_accuracy_std": format_std,
            }
        )

    out_prefix = _sanitize(str(args.output_prefix))
    per_run_csv = results_dir / f"{out_prefix}__per_run.csv"
    summary_csv = results_dir / f"{out_prefix}__summary.csv"
    summary_json = results_dir / f"{out_prefix}__summary.json"
    overlay_png = results_dir / f"{out_prefix}__{tag_slug}__overlay.png"
    mean_std_overlay_png = results_dir / f"{out_prefix}__{tag_slug}__mean_std__overlay.png"

    per_run_fields = [
        "variant",
        "label",
        "reward_last_step",
        "reward_last_value",
        "reward_num_points",
        "pre_accuracy",
        "post_accuracy",
        "pre_partial_accuracy",
        "post_partial_accuracy",
        "pre_format_accuracy",
        "post_format_accuracy",
        "reward_csv",
    ]
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_run_fields)
        writer.writeheader()
        writer.writerows(runs)

    summary_fields = [
        "variant",
        "n_runs",
        "reward_last_mean",
        "reward_last_std",
        "post_accuracy_mean",
        "post_accuracy_std",
        "post_partial_accuracy_mean",
        "post_partial_accuracy_std",
        "post_format_accuracy_mean",
        "post_format_accuracy_std",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    summary_json.write_text(
        json.dumps(
            {
                "results_dir": str(results_dir),
                "tag": str(args.tag),
                "since_epoch": since,
                "per_run_csv": str(per_run_csv),
                "summary_csv": str(summary_csv),
                "overlay_png": str(overlay_png),
                "mean_std_overlay_png": str(mean_std_overlay_png),
                "runs": runs,
                "summary_by_variant": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    try:
        _plot_overlay_png(overlay_png, series_for_plot, title=str(args.tag))
    except Exception as exc:
        print(f"[warn] overlay plot skipped: {exc}")

    try:
        variant_spec = [(v, variant_to_runs.get(v, [])) for v in sorted(variant_to_runs)]
        _plot_variant_mean_std_band_png(
            mean_std_overlay_png, variant_spec, title=f"{args.tag} (mean ± std)"
        )
    except Exception as exc:
        print(f"[warn] mean±std overlay plot skipped: {exc}")

    print(f"wrote: {per_run_csv}")
    print(f"wrote: {summary_csv}")
    print(f"wrote: {summary_json}")
    if overlay_png.exists():
        print(f"wrote: {overlay_png}")
    if mean_std_overlay_png.exists():
        print(f"wrote: {mean_std_overlay_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
