#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def _read_scalar_points(run_dir: Path, tag: str) -> list[ScalarPoint]:
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


def _write_csv(path: Path, points: Iterable[ScalarPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "value", "wall_time"])
        for p in points:
            writer.writerow([p.step, p.value, p.wall_time])


def _write_meta_json(path: Path, run_dir: Path, tag: str, points: list[ScalarPoint]) -> None:
    meta = {
        "tag": tag,
        "logdir": str(run_dir),
        "num_points": len(points),
        "first_step": points[0].step if points else None,
        "last_step": points[-1].step if points else None,
        "first_wall_time": points[0].wall_time if points else None,
        "last_wall_time": points[-1].wall_time if points else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


_EVAL_LINE_RE = re.compile(r"^(pre-train|post-train):\s*(.*)$")


def _parse_eval_metrics(log_text: str) -> dict[str, dict[str, float | int]]:
    parsed: dict[str, dict[str, float | int]] = {}
    for line in log_text.splitlines():
        m = _EVAL_LINE_RE.match(line.strip())
        if not m:
            continue
        phase = str(m.group(1))
        rest = str(m.group(2))
        metrics: dict[str, float | int] = {}
        for token in rest.split(","):
            token = token.strip()
            if not token or "=" not in token:
                continue
            key, raw = token.split("=", 1)
            key = key.strip()
            raw = raw.strip().rstrip("%")
            try:
                val = float(raw)
            except ValueError:
                continue
            if key in {"num_correct", "total"}:
                metrics[key] = int(val)
            else:
                metrics[key] = float(val)
        if metrics:
            parsed[phase] = metrics
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb-logdir", required=True, help="TensorBoard run directory")
    parser.add_argument("--label", required=True, help="Output prefix label")
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "my result"),
        help="Output directory (default: my_example/my result)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Scalar tag to export (repeatable). Example: global/eval/rewards/sum",
    )
    parser.add_argument("--stdout-log", default=None, help="Path to captured stdout/stderr log")
    args = parser.parse_args()

    run_dir = Path(args.tb_logdir)
    if not run_dir.exists():
        raise SystemExit(f"--tb-logdir does not exist: {run_dir}")

    outdir = Path(args.outdir)
    label = _sanitize(str(args.label))
    tags = list(args.tag) or [
        "global/eval/rewards/sum",
        "actor/train/skipped_samples",
    ]

    for tag in tags:
        tag_slug = _sanitize(tag)
        try:
            points = _read_scalar_points(run_dir, tag)
        except Exception as exc:
            print(f"skip tag {tag!r}: {exc}")
            continue

        csv_path = outdir / f"{label}__{tag_slug}.csv"
        meta_path = outdir / f"{label}__{tag_slug}__meta.json"
        _write_csv(csv_path, points)
        _write_meta_json(meta_path, run_dir, tag, points)
        print(f"wrote: {csv_path}")
        print(f"wrote: {meta_path}")

    if args.stdout_log:
        log_path = Path(args.stdout_log)
        if log_path.exists():
            metrics = _parse_eval_metrics(log_path.read_text(errors="replace"))
            if metrics:
                out_path = outdir / f"{label}__eval_accuracy__meta.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
                print(f"wrote: {out_path}")
            else:
                print(f"no eval metrics found in: {log_path}")
        else:
            print(f"--stdout-log not found: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
