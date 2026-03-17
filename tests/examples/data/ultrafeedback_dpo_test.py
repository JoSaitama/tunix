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

from unittest import mock

from absl.testing import absltest

from tunix.examples.data import ultrafeedback_dpo


class UltraFeedbackDpoTest(absltest.TestCase):

  def test_prompt_partition_is_deterministic_and_disjoint(self):
    prompt = "Explain TPU sharding."
    is_sft = ultrafeedback_dpo._prompt_is_in_partition(
        prompt, partition="sft", sft_fraction=0.5, seed=42
    )
    is_dpo = ultrafeedback_dpo._prompt_is_in_partition(
        prompt, partition="dpo", sft_fraction=0.5, seed=42
    )

    self.assertNotEqual(is_sft, is_dpo)

  def test_eval_subset_is_deterministic_and_disjoint(self):
    prompt = "Explain TPU sharding."
    is_train = ultrafeedback_dpo._prompt_is_in_subset(
        prompt, subset="train", eval_fraction=0.1, seed=42
    )
    is_eval = ultrafeedback_dpo._prompt_is_in_subset(
        prompt, subset="eval", eval_fraction=0.1, seed=42
    )

    self.assertNotEqual(is_train, is_eval)

  def test_extract_response_text_supports_chat_messages(self):
    response = ultrafeedback_dpo._extract_response_text([
        {"role": "assistant", "content": "better answer"}
    ])
    self.assertEqual(response, "better answer")

  def test_to_preference_record_formats_prompt_and_responses(self):
    record = ultrafeedback_dpo._to_preference_record({
        "prompt": "Explain TPU sharding.",
        "chosen": [{"role": "assistant", "content": "Detailed answer"}],
        "rejected": [{"role": "assistant", "content": "Bad answer"}],
    })
    self.assertEqual(
        record["prompt"],
        [{"role": "user", "content": "Explain TPU sharding."}],
    )
    self.assertEqual(record["chosen_responses"], "Detailed answer")
    self.assertEqual(record["rejected_responses"], "Bad answer")

  def test_create_dataset_limits_and_wraps_dataset(self):
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 10
    dataset.shuffle.return_value = dataset
    dataset.select.return_value = dataset
    dataset.filter.return_value = dataset
    mapped_dataset = object()

    with (
        mock.patch.object(
            ultrafeedback_dpo, "load_dataset", return_value=dataset
        ) as mock_load_dataset,
        mock.patch.object(
            ultrafeedback_dpo.grain.MapDataset,
            "source",
            return_value=mock.Mock(map=mock.Mock(return_value=mapped_dataset)),
        ) as mock_source,
    ):
      output = ultrafeedback_dpo.create_dataset("train_prefs", limit=8, seed=7)

    self.assertIs(output, mapped_dataset)
    mock_load_dataset.assert_called_once_with(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
    )
    dataset.shuffle.assert_called_once_with(seed=7)
    dataset.select.assert_called_once_with(range(8))
    mock_source.assert_called_once_with(dataset)

  def test_create_dataset_applies_prompt_partition_before_shuffle(self):
    dataset = mock.MagicMock()
    dataset.filter.return_value = dataset
    dataset.shuffle.return_value = dataset
    mapped_dataset = object()

    with (
        mock.patch.object(
            ultrafeedback_dpo, "load_dataset", return_value=dataset
        ),
        mock.patch.object(
            ultrafeedback_dpo.grain.MapDataset,
            "source",
            return_value=mock.Mock(map=mock.Mock(return_value=mapped_dataset)),
        ),
    ):
      output = ultrafeedback_dpo.create_dataset(
          "train_prefs",
          partition="dpo",
          sft_fraction=0.4,
          subset="train",
          eval_fraction=0.1,
          seed=11,
      )

    self.assertIs(output, mapped_dataset)
    self.assertEqual(dataset.filter.call_count, 2)
    dataset.shuffle.assert_called_once_with(seed=11)


if __name__ == "__main__":
  absltest.main()
