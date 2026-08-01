"""Tests for the DeepScaleR dataset recipe."""

from absl.testing import absltest
import pandas as pd
from tunix.cli.recipes import deepscaler_data


class DeepScalerDataTest(absltest.TestCase):

  def test_filters_empty_problem_and_answer_without_solution_recovery(self):
    frame = pd.DataFrame([
        {"problem": "valid", "answer": "42", "solution": "unused"},
        {"problem": "", "answer": "1", "solution": "unused"},
        {"problem": "empty answer", "answer": "  ", "solution": "\\boxed{7}"},
        {"problem": "none answer", "answer": None, "solution": "\\boxed{8}"},
    ])

    filtered = deepscaler_data._filter_training_rows(frame)

    self.assertEqual(filtered["problem"].tolist(), ["valid"])
    self.assertEqual(filtered["answer"].tolist(), ["42"])

  def test_rejects_missing_required_columns(self):
    with self.assertRaisesRegex(ValueError, "missing columns.*answer"):
      deepscaler_data._filter_training_rows(
          pd.DataFrame([{"problem": "valid"}])
      )

  def test_rejects_dataset_with_no_valid_rows(self):
    with self.assertRaisesRegex(ValueError, "no non-empty"):
      deepscaler_data._filter_training_rows(
          pd.DataFrame([{"problem": "", "answer": ""}])
      )


if __name__ == "__main__":
  absltest.main()
