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

"""Tests for DeepScaler math eval @k metrics."""

from absl.testing import absltest

from tunix.utils import math_eval_metrics


def _sample(
    *,
    problem_id: str = "p0",
    answer: str = "42",
    ground_truth: str = "42",
    token_count: int = 10,
    truncated: bool | None = False,
):
  return math_eval_metrics.MathEvalSample(
      problem_id=problem_id,
      response=fr"We get \boxed{{{answer}}}.",
      ground_truth=ground_truth,
      token_count=token_count,
      truncated=truncated,
  )


class MathEvalMetricsTest(absltest.TestCase):

  def test_perfect_group(self):
    samples = [_sample() for _ in range(16)]

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertEqual(metrics["avg@16"], 1.0)
    self.assertEqual(metrics["pass@16"], 1.0)
    self.assertEqual(metrics["maj@16"], 1.0)
    self.assertEqual(metrics["avg_tokens"], 10.0)
    self.assertEqual(metrics["truncation_rate"], 0.0)

  def test_pass_can_be_true_when_majority_is_wrong(self):
    samples = [_sample(answer="42") for _ in range(5)]
    samples.extend(_sample(answer="17") for _ in range(11))

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertAlmostEqual(metrics["avg@16"], 5 / 16)
    self.assertEqual(metrics["pass@16"], 1.0)
    self.assertEqual(metrics["maj@16"], 0.0)

  def test_majority_correct(self):
    samples = [_sample(answer="42") for _ in range(9)]
    samples.extend(_sample(answer="17") for _ in range(7))

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertAlmostEqual(metrics["avg@16"], 9 / 16)
    self.assertEqual(metrics["pass@16"], 1.0)
    self.assertEqual(metrics["maj@16"], 1.0)

  def test_multiple_problems_are_averaged(self):
    samples = [_sample(problem_id="p0", answer="42") for _ in range(16)]
    samples.extend(
        _sample(problem_id="p1", answer="17", ground_truth="42")
        for _ in range(16)
    )

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertEqual(metrics["num_problems"], 2)
    self.assertEqual(metrics["num_samples"], 32)
    self.assertEqual(metrics["avg@16"], 0.5)
    self.assertEqual(metrics["pass@16"], 0.5)
    self.assertEqual(metrics["maj@16"], 0.5)

  def test_majority_tie_breaks_by_first_tied_answer(self):
    samples = [_sample(answer="17") for _ in range(8)]
    samples.extend(_sample(answer="42") for _ in range(8))

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertEqual(metrics["pass@16"], 1.0)
    self.assertEqual(metrics["maj@16"], 0.0)

  def test_empty_answers_are_incorrect_and_ignored_for_majority(self):
    samples = [
        math_eval_metrics.MathEvalSample(
            problem_id="p0",
            response="No boxed answer here.",
            ground_truth="42",
            token_count=8,
            truncated=False,
        )
        for _ in range(10)
    ]
    samples.extend(_sample(answer="42") for _ in range(6))

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertAlmostEqual(metrics["avg@16"], 6 / 16)
    self.assertEqual(metrics["pass@16"], 1.0)
    self.assertEqual(metrics["maj@16"], 1.0)

  def test_token_count_can_come_from_tokens(self):
    samples = [
        {
            "problem_id": "p0",
            "response": r"\boxed{42}",
            "ground_truth": "42",
            "tokens": list(range(i + 1)),
            "truncated": False,
        }
        for i in range(16)
    ]

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertEqual(metrics["avg_tokens"], 8.5)

  def test_truncation_can_be_explicit_or_inferred(self):
    samples = [
        _sample(token_count=10, truncated=(i < 4))
        for i in range(8)
    ]
    samples.extend(
        _sample(token_count=20 if i < 4 else 7, truncated=None)
        for i in range(8)
    )

    metrics = math_eval_metrics.compute_at_k_metrics(
        samples, k=16, max_generation_steps=20
    )

    self.assertEqual(metrics["truncation_rate"], 8 / 16)

  def test_strict_k_rejects_incomplete_groups(self):
    samples = [_sample() for _ in range(15)]

    with self.assertRaisesRegex(ValueError, "Expected exactly 16 samples"):
      math_eval_metrics.compute_at_k_metrics(samples, k=16)

  def test_equivalent_math_answers_use_existing_grader(self):
    samples = [
        _sample(answer=r"\frac{1}{2}", ground_truth="0.5")
        for _ in range(16)
    ]

    metrics = math_eval_metrics.compute_at_k_metrics(samples, k=16)

    self.assertEqual(metrics["avg@16"], 1.0)
    self.assertEqual(metrics["pass@16"], 1.0)
    self.assertEqual(metrics["maj@16"], 1.0)


if __name__ == "__main__":
  absltest.main()
