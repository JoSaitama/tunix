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

from tunix.cli import dpo_main


class _FakeTokenizer:

  def __init__(self, lengths):
    self._lengths = lengths

  def encode(self, text):
    return [0] * self._lengths[text]

  def bos_id(self):
    return None

  def dedup_bos_ids(self, ids):
    return ids


class _FakeDataset:

  def __init__(self, items):
    self._items = list(items)

  def __iter__(self):
    return iter(self._items)

  def __len__(self):
    return len(self._items)

  def filter(self, predicate):
    return _FakeDataset([item for item in self._items if predicate(item)])


class DpoMainTest(absltest.TestCase):

  def test_filter_dpo_dataset_drops_overlong_records(self):
    dataset = _FakeDataset([
        {
            "prompts": "prompt-ok",
            "chosen_responses": "chosen-ok",
            "rejected_responses": "rejected-ok",
        },
        {
            "prompts": "prompt-too-long",
            "chosen_responses": "chosen-ok",
            "rejected_responses": "rejected-ok",
        },
        {
            "prompts": "prompt-ok",
            "chosen_responses": "chosen-too-long",
            "rejected_responses": "rejected-ok",
        },
    ])
    tokenizer = _FakeTokenizer({
        "prompt-ok": 4,
        "prompt-too-long": 9,
        "chosen-ok": 5,
        "chosen-too-long": 10,
        "rejected-ok": 6,
    })

    filtered = dpo_main.filter_dpo_dataset(
        dataset,
        tokenizer=tokenizer,
        max_prompt_length=8,
        max_response_length=8,
        dataset_name="train",
    )

    self.assertLen(list(filtered), 1)

  def test_create_dpo_training_config_merges_base_and_dpo_settings(self):
    pipeline = object.__new__(dpo_main.DpoPipeline)
    pipeline.config = {
        "dpo_config": {
            "beta": 0.2,
            "label_smoothing": 0.1,
            "max_prompt_length": 32,
            "max_response_length": 16,
        }
    }
    pipeline.obtain_training_config_dict = mock.Mock(return_value={
        "eval_every_n_steps": 10,
        "max_steps": 20,
    })

    training_config = pipeline.create_dpo_training_config()

    self.assertEqual(training_config.eval_every_n_steps, 10)
    self.assertEqual(training_config.max_steps, 20)
    self.assertEqual(training_config.beta, 0.2)
    self.assertEqual(training_config.label_smoothing, 0.1)
    self.assertEqual(training_config.max_prompt_length, 32)
    self.assertEqual(training_config.max_response_length, 16)

  def test_create_optimizer_with_clipping_wraps_base_optimizer(self):
    pipeline = object.__new__(dpo_main.DpoPipeline)
    pipeline.config = {"optimizer_config": {"max_grad_norm": 0.5}}
    pipeline.create_optimizer = mock.Mock(return_value="base-optimizer")

    with (
        mock.patch.object(
            dpo_main.optax,
            "clip_by_global_norm",
            return_value="clip-op",
        ) as mock_clip,
        mock.patch.object(
            dpo_main.optax, "chain", return_value="wrapped-optimizer"
        ) as mock_chain,
    ):
      optimizer = pipeline.create_optimizer_with_clipping()

    self.assertEqual(optimizer, "wrapped-optimizer")
    mock_clip.assert_called_once_with(0.5)
    mock_chain.assert_called_once_with("clip-op", "base-optimizer")

  def test_maybe_save_merged_lora_invokes_model_saver(self):
    pipeline = object.__new__(dpo_main.DpoPipeline)
    pipeline.config = {
        "actor_model_config": {
            "lora_config": {"rank": 64, "alpha": 64.0},
            "model_name": "qwen3-4b-instruct-2507",
        },
        "merged_model_output_dir": "/tmp/merged",
    }
    save_fn = mock.Mock()
    params_module = mock.Mock(
        save_lora_merged_model_as_safetensors=save_fn,
    )

    with (
        mock.patch.object(
            dpo_main.automodel,
            "get_model_module",
            return_value=params_module,
        ) as mock_get_model_module,
        mock.patch.object(dpo_main.os, "makedirs") as mock_makedirs,
    ):
      pipeline.maybe_save_merged_lora(actor_model="actor", model_path="/tmp/model")

    mock_get_model_module.assert_called_once()
    mock_makedirs.assert_called_once_with("/tmp/merged", exist_ok=True)
    save_fn.assert_called_once_with(
        local_model_path="/tmp/model",
        output_dir="/tmp/merged",
        lora_model="actor",
        rank=64,
        alpha=64.0,
    )


if __name__ == "__main__":
  absltest.main()
