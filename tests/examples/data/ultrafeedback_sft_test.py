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

from tunix.examples.data import ultrafeedback_sft


class UltraFeedbackSftTest(absltest.TestCase):

  def test_to_sft_record_uses_chosen_response(self):
    record = ultrafeedback_sft._to_sft_record({
        "prompt": "Explain TPU sharding.",
        "chosen": [{"role": "assistant", "content": "Detailed answer"}],
    })

    self.assertEqual(
        record["prompt"],
        [{"role": "user", "content": "Explain TPU sharding."}],
    )
    self.assertEqual(record["response"], "Detailed answer")

  def test_create_dataset_filters_partition_and_limits(self):
    dataset = mock.MagicMock()
    dataset.__len__.return_value = 10
    dataset.filter.return_value = dataset
    dataset.shuffle.return_value = dataset
    dataset.select.return_value = dataset
    mapped_dataset = object()

    with (
        mock.patch.object(
            ultrafeedback_sft.ultrafeedback_dpo,
            "load_dataset",
            return_value=dataset,
        ) as mock_load_dataset,
        mock.patch.object(
            ultrafeedback_sft.grain.MapDataset,
            "source",
            return_value=mock.Mock(map=mock.Mock(return_value=mapped_dataset)),
        ) as mock_source,
    ):
      output = ultrafeedback_sft.create_dataset(
          "train_prefs",
          partition="sft",
          sft_fraction=0.5,
          subset="train",
          eval_fraction=0.1,
          limit=8,
          seed=7,
      )

    self.assertIs(output, mapped_dataset)
    mock_load_dataset.assert_called_once_with(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
    )
    self.assertEqual(dataset.filter.call_count, 2)
    dataset.shuffle.assert_called_once_with(seed=7)
    dataset.select.assert_called_once_with(range(8))
    mock_source.assert_called_once_with(dataset)
