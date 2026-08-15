"""CPU tests for strict offline AIME dataset validation."""

import argparse
import tempfile
from unittest import mock

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

  def test_generate_once_forwards_seed_to_vllm(self):
    sampler = mock.Mock(return_value=object())
    args = argparse.Namespace(
        top_p=0.95,
        max_generation_steps=8192,
        max_prompt_length=2048,
        temperature=0.6,
        top_k=None,
    )

    result = eval_final_checkpoint_metrics._generate_once(
        args, sampler, ["p0", "p1"], 2029
    )

    self.assertIsNotNone(result)
    sampler.assert_called_once_with(
        input_strings=["p0", "p1"],
        max_generation_steps=8192,
        max_prompt_length=2048,
        temperature=0.6,
        top_p=0.95,
        top_k=None,
        seed=2029,
        echo=False,
        pad_output=False,
    )

  def test_eval_seed_argument_is_explicit_and_stable(self):
    with mock.patch(
        "sys.argv", ["eval", "--eval_seed", "20260707"]
    ):
      args = eval_final_checkpoint_metrics.parse_args()

    self.assertEqual(args.eval_seed, 20260707)
    self.assertEqual(args.problem_batch_size, 16)

  def test_generation_schedule_is_sample_slot_major(self):
    batches = list(eval_final_checkpoint_metrics._generation_batches(
        num_problems=30,
        num_samples=16,
        problem_batch_size=16,
        eval_seed=2026,
    ))

    self.assertLen(batches, 32)
    self.assertEqual(batches[0], (0, 0, 16, 2026))
    self.assertEqual(batches[1], (0, 16, 30, 2026))
    self.assertEqual(batches[2], (1, 0, 16, 2027))
    self.assertEqual(batches[-1], (15, 16, 30, 2041))


if __name__ == "__main__":
  absltest.main()
