# Copyright 2025 Google LLC
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

"""Main entry point for DPO training."""

from __future__ import annotations

import os
from typing import Any

from absl import app
from absl import logging
import optax
from tunix.cli import config
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_lib
from tunix.models import automodel
from tunix.sft.dpo import dpo_trainer as dpo_trainer_lib


def _token_length(text: str, tokenizer) -> int:
  """Computes the token length used by DPOTrainer string preprocessing."""
  input_ids = tokenizer.encode(text)
  bos_tokens = [tokenizer.bos_id()] if tokenizer.bos_id() else []
  return len(tokenizer.dedup_bos_ids(bos_tokens + input_ids))


def _record_within_length_limits(
    record: dict[str, Any],
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
) -> bool:
  """Checks whether a DPO record fits the configured token budgets."""
  return (
      _token_length(record["prompts"], tokenizer) <= max_prompt_length
      and _token_length(record["chosen_responses"], tokenizer)
      <= max_response_length
      and _token_length(record["rejected_responses"], tokenizer)
      <= max_response_length
  )


def filter_dpo_dataset(
    dataset,
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
    dataset_name: str,
):
  """Filters out DPO examples whose prompt or responses exceed the limits."""

  def predicate(record: dict[str, Any]) -> bool:
    return _record_within_length_limits(
        record, tokenizer, max_prompt_length, max_response_length
    )

  total_examples = len(dataset)
  kept_examples = sum(1 for record in dataset if predicate(record))
  dropped_examples = total_examples - kept_examples
  logging.info(
      "Filtered overlong DPO samples for %s: kept %d/%d, dropped %d "
      "(max_prompt_length=%d, max_response_length=%d).",
      dataset_name,
      kept_examples,
      total_examples,
      dropped_examples,
      max_prompt_length,
      max_response_length,
  )
  return dataset.filter(predicate)


class DpoPipeline(config.HyperParameters):
  """Pipeline for DPO training with a frozen reference model."""

  def _validate_dpo_model_pair(self) -> None:
    actor_config = self.config["actor_model_config"]
    reference_config = self.config["reference_model_config"]
    for key in ("model_name", "model_id", "model_source"):
      if actor_config.get(key) != reference_config.get(key):
        raise ValueError(
            "DPO actor and reference models must share the same base model. "
            f"Mismatch for '{key}': actor={actor_config.get(key)!r}, "
            f"reference={reference_config.get(key)!r}."
        )

  def create_optimizer_with_clipping(self):
    optimizer = self.create_optimizer("optimizer_config")
    max_grad_norm = self.config["optimizer_config"].get("max_grad_norm")
    if max_grad_norm is not None:
      optimizer = optax.chain(
          optax.clip_by_global_norm(max_grad_norm),
          optimizer,
      )
    return optimizer

  def create_dpo_training_config(self):
    training_kwargs = dict(self.obtain_training_config_dict("training_config"))
    training_kwargs.update(self.config["dpo_config"])
    return dpo_trainer_lib.DPOTrainingConfig(**training_kwargs)

  def create_models_and_tokenizer(self):
    self._validate_dpo_model_pair()
    reference_mesh = self.create_mesh("reference_model_config")
    actor_mesh = self.create_mesh("actor_model_config")

    reference_model, tokenizer_path, model_path = model_lib.create_model(
        self.config["reference_model_config"],
        self.config["tokenizer_config"],
        reference_mesh,
        return_model_path=True,
    )
    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"], tokenizer_path
    )

    actor_model = reference_model
    actor_lora_config = self.config["actor_model_config"].get("lora_config")
    if actor_lora_config:
      actor_model = model_lib.apply_lora_to_model(
          reference_model,
          actor_mesh,
          actor_lora_config,
      )
    else:
      logging.warning(
          "actor_model_config has no lora_config. DPO will train full weights."
      )

    return (
        actor_model,
        reference_model,
        tokenizer,
        tokenizer_path,
        model_path,
        actor_mesh,
    )

  def load_dataset(self, module_spec: str, tokenizer, batch_size: int, name: str):
    dataset = data_lib.get_dataset_from_module(module_spec, tokenizer)
    dpo_config = self.config["dpo_config"]
    dataset = filter_dpo_dataset(
        dataset,
        tokenizer=tokenizer,
        max_prompt_length=dpo_config["max_prompt_length"],
        max_response_length=dpo_config["max_response_length"],
        dataset_name=name,
    )
    dataset = dataset.to_iter_dataset()
    return data_lib.post_init_dataset(
        dataset,
        tokenizer,
        batch_size=batch_size,
        num_batches=None,
        max_prompt_length=None,
    )

  def maybe_save_merged_lora(self, actor_model, model_path: str) -> None:
    actor_lora_config = self.config["actor_model_config"].get("lora_config")
    if not actor_lora_config:
      return

    output_dir = self.config.get("merged_model_output_dir")
    if not output_dir:
      return

    model_name = self.config["actor_model_config"]["model_name"]
    params_modules = [
        automodel.get_model_module(model_name, automodel.ModelModule.PARAMS)
    ]
    if model_name.startswith("gemma"):
      params_modules.append(
          automodel.get_model_module(
              model_name, automodel.ModelModule.PARAMS_SAFETENSORS
          )
      )

    save_fn = None
    for params_module in params_modules:
      save_fn = getattr(
          params_module, "save_lora_merged_model_as_safetensors", None
      )
      if save_fn is not None:
        break

    if save_fn is None:
      raise AttributeError(
          "No LoRA merge saver found for model "
          f"{self.config['actor_model_config']['model_name']}."
      )

    os.makedirs(output_dir, exist_ok=True)
    logging.info("Saving merged LoRA model to %s", output_dir)
    save_fn(
        local_model_path=model_path,
        output_dir=output_dir,
        lora_model=actor_model,
        rank=actor_lora_config["rank"],
        alpha=actor_lora_config["alpha"],
    )

  def run_dpo_trainer(self):
    actor_model, reference_model, tokenizer, _, model_path, mesh = (
        self.create_models_and_tokenizer()
    )
    train_dataset = self.load_dataset(
        self.config["train_data_module"],
        tokenizer,
        batch_size=self.config["batch_size"],
        name="train",
    )
    eval_dataset = None
    if self.config.get("eval_data_module"):
      eval_dataset = self.load_dataset(
          self.config["eval_data_module"],
          tokenizer,
          batch_size=self.config.get(
              "eval_batch_size", self.config["batch_size"]
          ),
          name="eval",
      )

    trainer = dpo_trainer_lib.DPOTrainer(
        model=actor_model,
        ref_model=reference_model,
        optimizer=self.create_optimizer_with_clipping(),
        training_config=self.create_dpo_training_config(),
        tokenizer=tokenizer,
    )

    with mesh:
      trainer.train(train_dataset, eval_dataset)

    self.maybe_save_merged_lora(actor_model, model_path)


def main(argv, **kwargs):
  pipeline = DpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching DPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_dpo_trainer()


if __name__ == "__main__":
  app.run(main)
