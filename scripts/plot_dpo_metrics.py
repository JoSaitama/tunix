#!/usr/bin/env python3
"""Plots key DPO metrics from a TensorBoard logdir into one summary image."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


FULL_PLOT_SPECS = (
    (
        "Loss",
        (
            ("dpo/train/loss", "Train"),
            ("dpo/eval/loss", "Eval"),
        ),
    ),
    (
        "Reward Margin",
        (
            ("dpo/train/rewards/margin", "Train"),
            ("dpo/eval/rewards/margin", "Eval"),
        ),
    ),
    (
        "Reward Accuracy",
        (
            ("dpo/train/rewards/accuracy", "Train"),
            ("dpo/eval/rewards/accuracy", "Eval"),
        ),
    ),
    (
        "Chosen vs Rejected Rewards",
        (
            ("dpo/train/rewards/chosen", "Train chosen"),
            ("dpo/train/rewards/rejected", "Train rejected"),
            ("dpo/eval/rewards/chosen", "Eval chosen"),
            ("dpo/eval/rewards/rejected", "Eval rejected"),
        ),
    ),
    (
        "Chosen vs Rejected Log Probs",
        (
            ("dpo/train/log_probs/chosen", "Train chosen"),
            ("dpo/train/log_probs/rejected", "Train rejected"),
            ("dpo/eval/log_probs/chosen", "Eval chosen"),
            ("dpo/eval/log_probs/rejected", "Eval rejected"),
        ),
    ),
    (
        "Step Time (sec)",
        (
            ("dpo/train/step_time_sec", "Train"),
            ("dpo/eval/step_time_sec", "Eval"),
        ),
    ),
)

REPORT_PLOT_SPECS = (
    FULL_PLOT_SPECS[0],
    FULL_PLOT_SPECS[1],
    FULL_PLOT_SPECS[2],
    FULL_PLOT_SPECS[3],
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--logdir", required=True, help="TensorBoard logdir")
  parser.add_argument(
      "--output",
      required=True,
      help="Output PNG path for the summary figure",
  )
  parser.add_argument(
      "--title",
      default="DPO Metrics Summary",
      help="Figure title",
  )
  parser.add_argument(
      "--preset",
      choices=("full", "report"),
      default="full",
      help="Plot preset: full=6 panels, report=4 key panels",
  )
  return parser.parse_args()


def load_scalar_series(
    accumulator: event_accumulator.EventAccumulator, tag: str
) -> tuple[np.ndarray, np.ndarray] | None:
  if tag not in accumulator.Tags().get("scalars", []):
    return None
  events = accumulator.Scalars(tag)
  if not events:
    return None
  steps = np.array([event.step for event in events], dtype=float)
  values = np.array([event.value for event in events], dtype=float)
  return steps, values


def maybe_filter_initial_point(
    title: str, steps: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  if title == "Loss":
    mask = steps > 0
    if mask.any():
      return steps[mask], values[mask]
  return steps, values


def smooth_series(values: np.ndarray) -> np.ndarray:
  if len(values) < 25:
    return values
  window = min(51, max(9, len(values) // 30))
  if window % 2 == 0:
    window += 1
  kernel = np.ones(window, dtype=float) / float(window)
  return np.convolve(values, kernel, mode="same")


def plot_line(ax: plt.Axes, steps: np.ndarray, values: np.ndarray, label: str):
  if len(values) >= 50 and label.lower().startswith("train"):
    raw_line = ax.plot(steps, values, alpha=0.2, linewidth=1.0)[0]
    ax.plot(
        steps,
        smooth_series(values),
        label=label,
        linewidth=2.0,
        color=raw_line.get_color(),
    )
    return
  ax.plot(steps, values, label=label, linewidth=2.0)


def main():
  args = parse_args()
  logdir = pathlib.Path(args.logdir)
  output = pathlib.Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)

  accumulator = event_accumulator.EventAccumulator(str(logdir))
  accumulator.Reload()

  plot_specs = (
      REPORT_PLOT_SPECS if args.preset == "report" else FULL_PLOT_SPECS
  )
  if args.preset == "report":
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
  else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
  axes = axes.flatten()

  max_step = 0
  for ax, (title, series_specs) in zip(axes, plot_specs, strict=True):
    for tag, label in series_specs:
      series = load_scalar_series(accumulator, tag)
      if series is None:
        continue
      steps, values = series
      steps, values = maybe_filter_initial_point(title, steps, values)
      if len(steps) == 0:
        continue
      max_step = max(max_step, int(steps[-1]))
      plot_line(ax, steps, values, label)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.grid(alpha=0.25)
    if ax.lines:
      ax.legend(frameon=False, fontsize=9)
    else:
      ax.text(0.5, 0.5, "No data", ha="center", va="center")
      ax.set_axis_off()

  for ax in axes[len(plot_specs) :]:
    ax.set_axis_off()

  fig.suptitle(f"{args.title}  |  latest step={max_step}", fontsize=16)
  fig.savefig(output, dpi=180, bbox_inches="tight")
  print(output)


if __name__ == "__main__":
  main()
