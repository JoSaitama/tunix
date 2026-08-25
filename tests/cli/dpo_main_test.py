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
            "use_dynamic_batch_curation": True,
            "curation_variant": "self-inf-batch",
            "curation_threshold": 2.5,
            "curation_keep_ratio": 0.9,
            "curation_seed": 7,
            "self_influence_dot_threshold": 0.25,
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
    self.assertTrue(training_config.use_dynamic_batch_curation)
    self.assertEqual(training_config.curation_variant, "self_inf_batch")
    self.assertEqual(training_config.curation_threshold, 2.5)
    self.assertEqual(training_config.curation_keep_ratio, 0.9)
    self.assertEqual(training_config.curation_seed, 7)
    self.assertEqual(training_config.self_influence_dot_threshold, 0.25)

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

  def test_create_models_and_tokenizer_loads_separate_full_models(self):
    pipeline = object.__new__(dpo_main.DpoPipeline)
    pipeline.config = {
        "actor_model_config": {
            "model_name": "qwen2.5-1.5b",
            "model_id": "Qwen/Qwen2.5-1.5B",
            "model_source": "internal",
            "model_path": "/tmp/model",
        },
        "reference_model_config": {
            "model_name": "qwen2.5-1.5b",
            "model_id": "Qwen/Qwen2.5-1.5B",
            "model_source": "internal",
            "model_path": "/tmp/model",
        },
        "tokenizer_config": {"tokenizer_path": "tok"},
    }
    pipeline.create_mesh = mock.Mock(side_effect=["ref-mesh", "actor-mesh"])
    reference_model = mock.Mock(name="reference-model")
    actor_model = mock.Mock(name="actor-model")

    with (
        mock.patch.object(
            dpo_main.model_lib,
            "create_model",
            side_effect=[
                (reference_model, "tok-path", "/tmp/reference"),
                (actor_model, "tok-path", "/tmp/actor"),
            ],
        ) as mock_create_model,
        mock.patch.object(
            dpo_main.model_lib,
            "create_tokenizer",
            return_value="tokenizer",
        ) as mock_create_tokenizer,
        mock.patch.object(
            dpo_main.model_lib,
            "apply_lora_to_model",
        ) as mock_apply_lora,
    ):
      result = pipeline.create_models_and_tokenizer()

    self.assertEqual(
        result,
        (
            actor_model,
            reference_model,
            "tokenizer",
            "tok-path",
            "/tmp/actor",
            "actor-mesh",
        ),
    )
    self.assertEqual(mock_create_model.call_count, 2)
    mock_create_tokenizer.assert_called_once_with({"tokenizer_path": "tok"}, "tok-path")
    mock_apply_lora.assert_not_called()

  def test_maybe_export_model_invokes_lora_exporter(self):
    pipeline = object.__new__(dpo_main.DpoPipeline)
    pipeline.config = {
        "actor_model_config": {
            "lora_config": {"rank": 64, "alpha": 64.0},
            "model_name": "qwen3-4b-instruct-2507",
        },
        "exported_model_output_dir": "/tmp/exported",
    }

    with mock.patch.object(
        dpo_main.model_lib,
        "save_model_as_safetensors",
    ) as mock_save_model:
      pipeline.maybe_export_model(actor_model="actor", model_path="/tmp/model")

    mock_save_model.assert_called_once_with(
        model_name="qwen3-4b-instruct-2507",
        local_model_path="/tmp/model",
        output_dir="/tmp/exported",
        model_obj="actor",
        lora_config={"rank": 64, "alpha": 64.0},
    )

  def test_maybe_export_model_invokes_full_exporter(self):
    pipeline = object.__new__(dpo_main.DpoPipeline)
    pipeline.config = {
        "actor_model_config": {
            "model_name": "qwen2.5-1.5b",
        },
        "exported_model_output_dir": "/tmp/exported",
    }

    with mock.patch.object(
        dpo_main.model_lib,
        "save_model_as_safetensors",
    ) as mock_save_model:
      pipeline.maybe_export_model(actor_model="actor", model_path="/tmp/model")

    mock_save_model.assert_called_once_with(
        model_name="qwen2.5-1.5b",
        local_model_path="/tmp/model",
        output_dir="/tmp/exported",
        model_obj="actor",
        lora_config=None,
    )

  def test_run_dpo_trainer_uses_curated_trainer_when_enabled(self):
    pipeline = object.__new__(dpo_main.DpoPipeline)
    pipeline.config = {
        "train_data_module": "train",
        "eval_data_module": "eval",
        "batch_size": 2,
        "eval_batch_size": 2,
    }
    pipeline.create_models_and_tokenizer = mock.Mock(return_value=(
        "actor-model",
        "reference-model",
        "tokenizer",
        "tokenizer-path",
        "model-path",
        mock.MagicMock(),
    ))
    pipeline.load_dataset = mock.Mock(side_effect=["train-ds", "eval-ds"])
    pipeline.create_optimizer_with_clipping = mock.Mock(return_value="optimizer")
    pipeline.maybe_export_model = mock.Mock()
    training_config = dpo_main.dpo_trainer_lib.DPOTrainingConfig(
        eval_every_n_steps=10,
        max_steps=20,
        use_dynamic_batch_curation=True,
    )
    pipeline.create_dpo_training_config = mock.Mock(return_value=training_config)
    trainer = mock.MagicMock()

    with mock.patch.object(
        dpo_main.dpo_trainer_lib,
        "CuratedDPOTrainer",
        return_value=trainer,
    ) as mock_curated_trainer:
      pipeline.run_dpo_trainer()

    mock_curated_trainer.assert_called_once_with(
        model="actor-model",
        ref_model="reference-model",
        optimizer="optimizer",
        training_config=training_config,
        tokenizer="tokenizer",
    )
    trainer.train.assert_called_once_with("train-ds", "eval-ds")


if __name__ == "__main__":
  absltest.main()
