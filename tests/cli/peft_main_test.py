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

from tunix.cli import peft_main


class PeftMainTest(absltest.TestCase):

  def test_create_datasets_uses_translation_dataset_by_default(self):
    pipeline = object.__new__(peft_main.PeftPipeline)
    pipeline.config = {
        "dataset_name": "mtnt/en-fr",
        "batch_size": 2,
        "max_target_length": 256,
        "num_train_epochs": 1,
        "tfds_download": True,
    }

    with mock.patch.object(
        peft_main.translation_data_lib,
        "create_datasets",
        return_value=("train", "eval"),
    ) as mock_create_datasets:
      train_ds, eval_ds = pipeline.create_datasets("tokenizer")

    self.assertEqual((train_ds, eval_ds), ("train", "eval"))
    mock_create_datasets.assert_called_once_with(
        dataset_name="mtnt/en-fr",
        global_batch_size=2,
        max_target_length=256,
        num_train_epochs=1,
        tokenizer="tokenizer",
        tfds_download=True,
    )

  def test_create_datasets_uses_sft_modules_when_configured(self):
    pipeline = object.__new__(peft_main.PeftPipeline)
    pipeline.config = {
        "train_data_module": "train_module",
        "eval_data_module": "eval_module",
        "batch_size": 2,
        "eval_batch_size": 4,
        "num_batches": 6,
        "num_test_batches": 3,
    }
    pipeline.load_sft_dataset = mock.Mock(side_effect=["train-ds", "eval-ds"])

    train_ds, eval_ds = pipeline.create_datasets("tokenizer")

    self.assertEqual((train_ds, eval_ds), ("train-ds", "eval-ds"))
    pipeline.load_sft_dataset.assert_any_call(
        "train_module",
        "tokenizer",
        batch_size=2,
        num_batches=6,
        name="train",
    )
    pipeline.load_sft_dataset.assert_any_call(
        "eval_module",
        "tokenizer",
        batch_size=4,
        num_batches=3,
        name="eval",
    )

  def test_maybe_export_model_invokes_lora_exporter(self):
    pipeline = object.__new__(peft_main.PeftPipeline)
    pipeline.config = {
        "model_config": {
            "lora_config": {"rank": 64, "alpha": 64.0},
            "model_name": "qwen2.5-1.5b",
        },
        "exported_model_output_dir": "/tmp/exported",
    }

    with (
        mock.patch.object(
            peft_main.model_lib,
            "save_model_as_safetensors",
        ) as mock_save_model,
    ):
      pipeline.maybe_export_model(model="actor", model_path="/tmp/model")

    mock_save_model.assert_called_once_with(
        model_name="qwen2.5-1.5b",
        local_model_path="/tmp/model",
        output_dir="/tmp/exported",
        model_obj="actor",
        lora_config={"rank": 64, "alpha": 64.0},
    )

  def test_maybe_export_model_invokes_full_exporter(self):
    pipeline = object.__new__(peft_main.PeftPipeline)
    pipeline.config = {
        "model_config": {
            "model_name": "qwen2.5-1.5b",
        },
        "exported_model_output_dir": "/tmp/exported",
    }

    with mock.patch.object(
        peft_main.model_lib,
        "save_model_as_safetensors",
    ) as mock_save_model:
      pipeline.maybe_export_model(model="full-model", model_path="/tmp/model")

    mock_save_model.assert_called_once_with(
        model_name="qwen2.5-1.5b",
        local_model_path="/tmp/model",
        output_dir="/tmp/exported",
        model_obj="full-model",
        lora_config=None,
    )
