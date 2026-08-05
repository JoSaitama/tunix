"""CPU tests for strict offline AIME dataset validation."""

import argparse
import tempfile

from absl.testing import absltest
import pandas as pd
from examples.deepscaler import eval_final_checkpoint_metrics


def _args(path, *, limit=None):
  return argparse.Namespace(
      dataset=path,
      limit=limit,
      question_col="problem",
      answer_col="answer",
      dataset_type="aime",
  )


class DeepScalerEvalTest(absltest.TestCase):

  def test_full_aime_requires_30_unique_complete_rows(self):
    frame = pd.DataFrame({
        "problem": [f"problem {index}" for index in range(30)],
        "answer": [str(index) for index in range(30)],
    })
    with tempfile.TemporaryDirectory() as temp_dir:
      path = f"{temp_dir}/aime.parquet"
      frame.to_parquet(path)
      loaded = eval_final_checkpoint_metrics._load_dataset(_args(path))

    self.assertLen(loaded, 30)
    self.assertEqual(loaded.attrs["validation"]["unique_questions"], 30)
    self.assertLen(loaded.attrs["validation"]["dataset_sha256"], 64)

  def test_full_aime_rejects_wrong_row_count(self):
    frame = pd.DataFrame({"problem": ["p"], "answer": ["1"]})
    with tempfile.TemporaryDirectory() as temp_dir:
      path = f"{temp_dir}/aime.parquet"
      frame.to_parquet(path)
      with self.assertRaisesRegex(ValueError, "exactly 30"):
        eval_final_checkpoint_metrics._load_dataset(_args(path))


if __name__ == "__main__":
  absltest.main()
