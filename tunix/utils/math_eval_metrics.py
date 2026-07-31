# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Math eval aggregation helpers for DeepScaler-style @k metrics."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import io
import numbers
from typing import Any, Iterable, Mapping, Sequence

from tunix.utils import math_rewards
from tunix.utils import math_utils


@dataclasses.dataclass(frozen=True)
class MathEvalSample:
  """One generated math-eval sample."""

  problem_id: str | int
  response: str | None
  ground_truth: Any
  token_count: int | None = None
  tokens: Sequence[Any] | None = None
  truncated: bool | None = None


def extract_response_answer(response: str | None) -> str | None:
  """Extracts the boxed final answer from a model response."""
  if response is None or response == "":
    return None
  if math_rewards.THOUGHT_DELIMITER_END in response:
    response = response.split(math_rewards.THOUGHT_DELIMITER_END, 1)[1]
  return math_utils.extract_answer(response)


def _processed_ground_truths(ground_truths: Any) -> list[str]:
  if ground_truths is None:
    return []
  if isinstance(ground_truths, (str, numbers.Number)):
    ground_truths = [ground_truths]

  processed = []
  for truth in ground_truths:
    truth = str(truth)
    if "\\boxed" in truth:
      extracted_truth = math_utils.extract_answer(truth)
      if extracted_truth is not None:
        processed.append(extracted_truth)
    else:
      processed.append(truth)
  return processed


def is_extracted_answer_correct(
    extracted_answer: str | None, ground_truths: Any
) -> bool:
  """Grades an already-extracted answer against one or more ground truths."""
  if extracted_answer is None:
    return False
  processed_ground_truths = _processed_ground_truths(ground_truths)
  if not processed_ground_truths:
    return False

  for ground_truth in processed_ground_truths:
    # grade_answer_mathd prints on every comparison; keep eval aggregation quiet.
    with contextlib.redirect_stdout(io.StringIO()):
      is_mathd_correct = math_utils.grade_answer_mathd(
          extracted_answer, ground_truth
      )
    if (
        is_mathd_correct
        or math_utils.grade_answer_sympy(extracted_answer, ground_truth)
        or math_utils.grade_answer_special_handling(
            extracted_answer, ground_truth
        )
    ):
      return True
  return False


def is_response_correct(response: str | None, ground_truths: Any) -> bool:
  """Extracts and grades a model response."""
  return is_extracted_answer_correct(
      extract_response_answer(response), ground_truths
  )


def _coerce_sample(sample: MathEvalSample | Mapping[str, Any]) -> MathEvalSample:
  if isinstance(sample, MathEvalSample):
    return sample
  return MathEvalSample(
      problem_id=sample["problem_id"],
      response=sample.get("response"),
      ground_truth=sample["ground_truth"],
      token_count=sample.get("token_count"),
      tokens=sample.get("tokens"),
      truncated=sample.get("truncated"),
  )


def _sample_token_count(sample: MathEvalSample) -> int:
  if sample.token_count is not None:
    return int(sample.token_count)
  if sample.tokens is not None:
    return len(sample.tokens)
  raise ValueError(
      "Each eval sample must include either token_count or tokens to compute "
      "avg_tokens."
  )


def _sample_truncated(
    sample: MathEvalSample, max_generation_steps: int | None
) -> bool:
  if sample.truncated is not None:
    return bool(sample.truncated)
  if max_generation_steps is None:
    raise ValueError(
        "Samples without explicit truncated values require max_generation_steps "
        "to compute truncation_rate."
    )
  return _sample_token_count(sample) >= max_generation_steps


def _vote_key(extracted_answer: str | None) -> str | None:
  if extracted_answer is None:
    return None
  normalized = math_utils.mathd_normalize_answer(extracted_answer)
  if normalized is None or normalized == "":
    return None
  return normalized


def _majority_answer(samples: Sequence[MathEvalSample]) -> str | None:
  counts: collections.Counter[str] = collections.Counter()
  first_index: dict[str, int] = {}
  representative: dict[str, str] = {}

  for index, sample in enumerate(samples):
    extracted_answer = extract_response_answer(sample.response)
    key = _vote_key(extracted_answer)
    if key is None:
      continue
    counts[key] += 1
    first_index.setdefault(key, index)
    representative.setdefault(key, extracted_answer)

  if not counts:
    return None
  winner = max(counts, key=lambda key: (counts[key], -first_index[key]))
  return representative[winner]


def compute_at_k_metrics(
    samples: Iterable[MathEvalSample | Mapping[str, Any]],
    *,
    k: int = 16,
    max_generation_steps: int | None = None,
    strict: bool = True,
) -> dict[str, float | int]:
  """Computes avg@k, pass@k, maj@k, avg_tokens, and truncation_rate.

  Args:
    samples: Per-generation eval samples. Each sample must include a problem id,
      response, ground truth, and token count information.
    k: Number of generated samples expected for each problem.
    max_generation_steps: Generation cap used to infer truncation when a sample
      does not include an explicit truncated flag.
    strict: Whether to require exactly k samples for each problem.

  Returns:
    A flat metrics dictionary with keys like "avg@16" and "pass@16".
  """
  grouped: dict[str | int, list[MathEvalSample]] = collections.defaultdict(list)
  for raw_sample in samples:
    sample = _coerce_sample(raw_sample)
    grouped[sample.problem_id].append(sample)

  if not grouped:
    raise ValueError("Cannot compute eval metrics from an empty sample set.")

  if strict:
    bad_counts = {
        problem_id: len(group_samples)
        for problem_id, group_samples in grouped.items()
        if len(group_samples) != k
    }
    if bad_counts:
      raise ValueError(
          f"Expected exactly {k} samples per problem, got {bad_counts}."
      )

  sample_correct = []
  sample_token_counts = []
  sample_truncated = []
  pass_scores = []
  maj_scores = []

  for group_samples in grouped.values():
    group_correct = [
        is_response_correct(sample.response, sample.ground_truth)
        for sample in group_samples
    ]
    sample_correct.extend(group_correct)
    sample_token_counts.extend(_sample_token_count(s) for s in group_samples)
    sample_truncated.extend(
        _sample_truncated(s, max_generation_steps) for s in group_samples
    )

    pass_scores.append(any(group_correct))
    majority_answer = _majority_answer(group_samples)
    maj_scores.append(
        is_extracted_answer_correct(
            majority_answer, group_samples[0].ground_truth
        )
    )

  num_samples = len(sample_correct)
  num_problems = len(grouped)
  return {
      f"avg@{k}": sum(sample_correct) / num_samples,
      f"pass@{k}": sum(pass_scores) / num_problems,
      f"maj@{k}": sum(maj_scores) / num_problems,
      "avg_tokens": sum(sample_token_counts) / num_samples,
      "truncation_rate": sum(sample_truncated) / num_samples,
      "num_problems": num_problems,
      "num_samples": num_samples,
      "k": k,
  }
