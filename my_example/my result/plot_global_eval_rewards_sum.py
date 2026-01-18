#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def discover_run_dirs(tb_root: Path) -> list[Path]:
    candidates = [p for p in tb_root.iterdir() if p.is_dir() and p.name.startswith("grpo")]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def discover_runs_with_tag(tb_root: Path, tag: str, latest: int) -> list[Path]:
    selected: list[Path] = []
    for run_dir in discover_run_dirs(tb_root):
        try:
            accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            selected.append(run_dir)
        except Exception:
            continue
        if latest > 0 and len(selected) >= latest:
            break
    return selected


def read_scalar_points(run_dir: Path, tag: str) -> list[ScalarPoint]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        raise ValueError(f"tag not found: {tag}")
    points = [
        ScalarPoint(step=int(e.step), value=float(e.value), wall_time=float(e.wall_time))
        for e in accumulator.Scalars(tag)
    ]
    points.sort(key=lambda p: p.step)
    return points


def write_csv(path: Path, points: Iterable[ScalarPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "value", "wall_time"])
        for p in points:
            writer.writerow([p.step, p.value, p.wall_time])


def read_csv_points(path: Path) -> list[ScalarPoint]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty csv")
        fields = set(reader.fieldnames)
        if "step" not in fields or "value" not in fields:
            raise ValueError(f"expected columns step,value; got {sorted(fields)}")
        points = [
            ScalarPoint(
                step=int(float(row["step"])),
                value=float(row["value"]),
                wall_time=float(row.get("wall_time", 0.0) or 0.0),
            )
            for row in reader
        ]
    points.sort(key=lambda p: p.step)
    return points


def discover_csv_series(outdir: Path, tag: str) -> list[tuple[str, Path]]:
    tag_slug = _sanitize(tag)
    suffix = f"__{tag_slug}.csv"
    csv_paths = sorted(outdir.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    series: list[tuple[str, Path]] = []
    for p in csv_paths:
        name = p.name[: -len(suffix)]
        series.append((name, p))
    return series


def _padded_limits(values: list[float]) -> tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    span = vmax - vmin
    if span <= 0:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    pad = span * 0.05
    return vmin - pad, vmax + pad


def plot_overlay_png(
    out_path: Path,
    series: list[tuple[str, list[ScalarPoint]]],
    title: str,
) -> None:
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


def main() -> int:
    run_labels = {
        "grpo_20260115_185431": "outlier-l2",
        "grpo_20260116_072833": "selfinf-batch",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="global/eval/rewards/sum")
    parser.add_argument("--tb-root", default="/tmp/content/tmp/tensorboard")
    parser.add_argument(
        "--latest",
        type=int,
        default=4,
        help="Auto-discover latest N runs (used when --run is not provided; 0 = no limit)",
    )
    parser.add_argument("--run", action="append", default=None, help="TensorBoard run dir (repeatable)")
    parser.add_argument("--csv", action="append", default=None, help="CSV path (repeatable)")
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Only plot series with at least N points",
    )
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    tag = str(args.tag)
    min_points = int(args.min_points)

    series: list[tuple[str, list[ScalarPoint]]] = []

    if args.csv or (not args.run and discover_csv_series(outdir, tag)):
        csv_specs: list[tuple[str, Path]]
        if args.csv:
            csv_specs = [(Path(p).stem, Path(p)) for p in args.csv]
        else:
            csv_specs = discover_csv_series(outdir, tag)

        for name, path in csv_specs:
            points = read_csv_points(path)
            if len(points) < min_points:
                print(f"skip {name}: n={len(points)} (<{min_points})")
                continue
            print(f"{name}: n={len(points)} last={points[-1].step} {points[-1].value}")
            series.append((name, points))
    else:
        runs: list[str]
        if args.run:
            runs = list(args.run)
        else:
            tb_root = Path(args.tb_root)
            runs = [str(p) for p in discover_runs_with_tag(tb_root, tag, latest=int(args.latest))]
            if not runs:
                raise SystemExit(f"no runs found under {tb_root} with tag {tag!r}")

        for run in runs:
            run_dir = Path(run)
            points = read_scalar_points(run_dir, tag)
            label = run_labels.get(run_dir.name, run_dir.name)
            if len(points) < min_points:
                print(f"skip {label} ({run_dir.name}): n={len(points)} (<{min_points})")
                continue
            print(f"{label} ({run_dir.name}): n={len(points)} last={points[-1].step} {points[-1].value}")
            csv_path = outdir / f"{_sanitize(label)}__{_sanitize(tag)}.csv"
            write_csv(csv_path, points)
            series.append((label, points))

    if not series:
        raise SystemExit("no series to plot")

    png_path = outdir / f"{_sanitize(tag)}__overlay.png"
    plot_overlay_png(png_path, series, title=tag)
    print(f"wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
