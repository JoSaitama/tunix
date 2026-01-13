from __future__ import annotations

import re
from typing import Iterable, List

from .prompts import reasoning_start, reasoning_end, solution_start, solution_end


MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

MATCH_NUMBERS = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)

# More permissive numeric extractor: allows sign and scientific notation.
_LOOSE_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_numeric_value(response: str) -> float | None:
    """Extract a numeric guess from the model response.

    Priority:
    1) If the strict format regex matches, use its captured group.
    2) Otherwise, fall back to the first loose numeric token.
    """
    match = MATCH_FORMAT.search(response)
    candidate = match.group(1) if match else None
    if candidate is None:
        fallback = _LOOSE_NUMBER.search(response)
        candidate = fallback.group(0) if fallback else None
    if candidate is None:
        return None
    try:
        return float(candidate.strip())
    except Exception:
        return None


def match_format_exactly(prompts, completions, **kwargs):
    return [0.0 if MATCH_FORMAT.search(response) is None else 3.0 for response in completions]


def match_format_approximately(prompts, completions, **kwargs):
    scores: List[float] = []
    for response in completions:
        score = 0.0
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.find(reasoning_start) == 0 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    extracted = [
        guess.group(1) if r is not None and (guess := MATCH_FORMAT.search(r)) is not None else None
        for r in completions
    ]

    scores: List[float] = []
    assert len(extracted) == len(answer), (
        f"{extracted} and {answer} have mismatching length"
    )

    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        if guess == true_answer:
            scores.append(3.0)
            continue
        if guess.strip() == true_answer.strip():
            scores.append(1.5)
            continue

        try:
            ratio = float(guess) / float(true_answer)
            if 0.9 <= ratio <= 1.1:
                scores.append(0.5)
            elif 0.8 <= ratio <= 1.2:
                scores.append(0.25)
            else:
                scores.append(-1.0)
        except Exception:
            scores.append(-0.5)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    extracted = [
        guess.group(1) if (guess := MATCH_NUMBERS.search(r)) is not None else None
        for r in completions
    ]

    scores: List[float] = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            true_answer_val = float(true_answer.strip())
            guess_val = float(guess.strip())
            scores.append(1.5 if guess_val == true_answer_val else 0.0)
        except Exception:
            scores.append(0.0)
    return scores


# === Accuracy-focused reward components (opt-in) ===

def format_integrity_reward(prompts, completions, **kwargs):
    """Lightweight format incentive: reward correct tags, penalize missing."""
    scores: List[float] = []
    for response in completions:
        scores.append(1.0 if MATCH_FORMAT.search(response) is not None else -1.0)
    return scores


def numeric_relative_reward(prompts, completions, answer, **kwargs):
    """Dense numeric reward aligned with eval accuracy (relative-error based)."""
    scores: List[float] = []
    for resp, true_answer in zip(completions, answer):
        pred_val = _parse_numeric_value(resp)
        try:
            true_val = float(true_answer.strip())
        except Exception:
            scores.append(0.0)
            continue

        if pred_val is None:
            scores.append(-1.0)  # Penalize unparseable answers.
            continue

        abs_err = abs(pred_val - true_val)
        denom = max(1.0, abs(true_val))
        rel_err = abs_err / denom

        if abs_err == 0.0:
            scores.append(4.0)
        elif rel_err <= 0.01:
            scores.append(3.0)
        elif rel_err <= 0.05:
            scores.append(2.0)
        elif rel_err <= 0.10:
            scores.append(1.0)
        elif rel_err <= 0.25:
            scores.append(0.25)
        else:
            scores.append(-1.0)
    return scores
