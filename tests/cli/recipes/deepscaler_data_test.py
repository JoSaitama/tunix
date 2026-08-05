"""Tests for the DeepScaleR dataset recipe."""

from absl.testing import absltest
import grain
import json
import pandas as pd
import tempfile
from tunix.cli.recipes import deepscaler_data
from tunix.cli.utils import data as data_lib


class _LengthTokenizer:

  def encode(self, prompt):
    return prompt.split()


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

  def test_strict_post_init_selects_exact_complete_batches(self):
    source = [
        {"prompts": "one two", "question": f"q{i}", "answer": str(i)}
        for i in range(7)
    ]
    source.insert(
        2, {"prompts": "one two three four", "question": "long", "answer": "x"}
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      manifest_path = f"{temp_dir}/manifest.json"
      train_ds, _ = data_lib.post_init_dataset(
          grain.MapDataset.source(source),
          _LengthTokenizer(),
          batch_size=3,
          num_batches=2,
          max_prompt_length=3,
          require_complete_num_batches=True,
          selection_manifest_path=manifest_path,
      )
      batches = list(train_ds)
      with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    self.assertLen(batches, 2)
    self.assertEqual(sum(len(batch["prompts"]) for batch in batches), 6)
    self.assertEqual(manifest["input_rows"], 8)
    self.assertEqual(manifest["overlength_rows"], 1)
    self.assertEqual(manifest["valid_rows"], 7)
    self.assertEqual(manifest["selected_rows"], 6)
    self.assertLen(manifest["selection_sha256"], 64)

  def test_strict_post_init_rejects_insufficient_valid_rows(self):
    source = [
        {"prompts": "one two three", "question": "long", "answer": "x"},
        {"prompts": "one", "question": "q", "answer": "1"},
    ]
    with self.assertRaisesRegex(ValueError, "valid_rows=1, required_rows=2"):
      data_lib.post_init_dataset(
          grain.MapDataset.source(source),
          _LengthTokenizer(),
          batch_size=2,
          num_batches=1,
          max_prompt_length=1,
          require_complete_num_batches=True,
      )


if __name__ == "__main__":
  absltest.main()
