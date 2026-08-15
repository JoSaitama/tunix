"""CPU tests for detailed and paired AIME evaluation analysis."""

from absl.testing import absltest

from examples.deepscaler import analyze_aime_eval
from examples.deepscaler import compare_aime_evals


def _records(correct_counts):
  records = []
  for problem_id, correct_count in enumerate(correct_counts):
    for sample_index in range(16):
      correct = sample_index < correct_count
      answer = "42" if correct else "17"
      records.append({
          "problem_id": problem_id,
          "sample_index": sample_index,
          "question": f"problem {problem_id}",
          "ground_truth": "42",
          "response": fr"Reasoning. \boxed{{{answer}}}",
          "extracted_answer": answer,
          "is_correct": correct,
          "token_count": 10 + sample_index,
          "truncated": False,
      })
  return records


class AimeEvalAnalysisTest(absltest.TestCase):

  def test_detailed_analysis_preserves_per_problem_counts(self):
    summary, rows = analyze_aime_eval.analyze_records(
        _records([4, 16]), k=16, max_generation_steps=8192
    )

    self.assertEqual(summary["num_records"], 32)
    self.assertEqual(summary["metrics"]["correct_count"], 20)
    self.assertEqual(rows[0]["correct_count"], 4)
    self.assertEqual(rows[1]["correct_count"], 16)

  def test_paired_bootstrap_is_reproducible(self):
    _, baseline = analyze_aime_eval.analyze_records(
        _records([2, 4, 6]), k=16, max_generation_steps=8192
    )
    _, dtv = analyze_aime_eval.analyze_records(
        _records([4, 5, 8]), k=16, max_generation_steps=8192
    )

    first = compare_aime_evals.compare(
        {"baseline": baseline, "dtv": dtv},
        reference="baseline",
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    second = compare_aime_evals.compare(
        {"baseline": baseline, "dtv": dtv},
        reference="baseline",
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    self.assertEqual(first, second)
    self.assertGreater(
        first["paired_comparisons"]["dtv"]["avg"]["delta"], 0
    )

  def test_comparison_rejects_misaligned_problems(self):
    _, baseline = analyze_aime_eval.analyze_records(
        _records([2]), k=16, max_generation_steps=8192
    )
    _, dtv = analyze_aime_eval.analyze_records(
        _records([4]), k=16, max_generation_steps=8192
    )
    dtv[0]["question"] = "different problem"

    with self.assertRaisesRegex(ValueError, "not problem-aligned"):
      compare_aime_evals.compare(
          {"baseline": baseline, "dtv": dtv},
          reference="baseline",
          bootstrap_samples=10,
          bootstrap_seed=7,
      )


if __name__ == "__main__":
  absltest.main()
