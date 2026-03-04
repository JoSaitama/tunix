from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
from tunix.utils import math_utils


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _extract_answer_text(response: Any) -> str | None:
    if response is None:
        return None
    text = str(response)
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return math_utils.extract_answer(text)


class RewardSignalDiagnostics:
    """Optional diagnostics for sparse-reward debugging."""

    def __init__(self, print_every: int = 200):
        self._calls = 0
        self._print_every = max(int(print_every), 1)

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        rewards: Sequence[float],
        **kwargs,
    ):
        del prompts
        mode = kwargs.get("mode", "unknown")

        completion_texts = ["" if c is None else str(c) for c in completions]
        total = len(completion_texts)
        if total == 0:
            boxed_rate = 0.0
            parseable_rate = 0.0
        else:
            boxed_count = sum("\\boxed" in text for text in completion_texts)
            parseable_count = sum(
                _extract_answer_text(text) is not None for text in completion_texts
            )
            boxed_rate = boxed_count / total
            parseable_rate = parseable_count / total

        reward_array = np.asarray(rewards, dtype=np.float32)
        reward_nonzero_rate = (
            float(np.mean(np.abs(reward_array) > 1e-6)) if reward_array.size else 0.0
        )
        reward_mean = float(np.mean(reward_array)) if reward_array.size else 0.0

        self._calls += 1
        if self._calls % self._print_every == 0:
            print(
                f"[diag][{mode}] call={self._calls} "
                f"reward_nonzero_rate={reward_nonzero_rate:.4f} "
                f"boxed_rate={boxed_rate:.4f} "
                f"answer_parseable_rate={parseable_rate:.4f} "
                f"reward_mean={reward_mean:.4f}"
            )

        return {
            "diag/reward_nonzero_rate": (reward_nonzero_rate, np.mean),
            "diag/boxed_rate": (boxed_rate, np.mean),
            "diag/answer_parseable_rate": (parseable_rate, np.mean),
            "diag/reward_mean": (reward_mean, np.mean),
        }


def build_optional_metric_fns():
    if not _env_flag("TUNIX_QWEN_AIME_DIAG"):
        return []

    print_every = _env_int("TUNIX_QWEN_AIME_DIAG_PRINT_EVERY", 200)
    print(
        "[diag] enabled: "
        f"TUNIX_QWEN_AIME_DIAG=1, print_every={max(print_every, 1)}"
    )
    return [RewardSignalDiagnostics(print_every=print_every)]
