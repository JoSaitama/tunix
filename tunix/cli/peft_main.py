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

"""Main entry point for PEFT training."""
from collections.abc import Callable
from typing import Any
from absl import app
from absl import logging
from flax import nnx
import jax
from tunix.cli import config
from tunix.cli.utils import data as cli_data_lib
from tunix.cli.utils import model as model_lib
from tunix.examples.data import translation_dataset as translation_data_lib
from tunix.sft import peft_trainer
from tunix.sft import utils


class PeftPipeline(config.HyperParameters):
  """Class for running the Peft trainer."""

  def load_sft_dataset(
      self,
      module_spec: str,
      tokenizer,
      batch_size: int,
      num_batches: int | None,
      name: str,
  ):
    dataset = cli_data_lib.get_sft_dataset_from_module(
        module_spec,
        tokenizer=tokenizer,
        max_target_length=self.config["max_target_length"],
    )
    dataset = dataset.to_iter_dataset()
    logging.info("Loaded SFT dataset module for %s: %s", name, module_spec)
    return cli_data_lib.post_init_dataset(
        dataset,
        tokenizer,
        batch_size=batch_size,
        num_batches=num_batches,
        max_prompt_length=None,
    )

  def create_datasets(self, tokenizer):
    if self.config.get("train_data_module"):
      train_ds = self.load_sft_dataset(
          self.config["train_data_module"],
          tokenizer,
          batch_size=self.config["batch_size"],
          num_batches=self.config.get("num_batches"),
          name="train",
      )
      eval_ds = None
      if self.config.get("eval_data_module"):
        eval_ds = self.load_sft_dataset(
            self.config["eval_data_module"],
            tokenizer,
            batch_size=self.config.get("eval_batch_size", self.config["batch_size"]),
            num_batches=self.config.get("num_test_batches"),
            name="eval",
        )
      return train_ds, eval_ds

    return translation_data_lib.create_datasets(
        dataset_name=self.config["dataset_name"],
        global_batch_size=self.config["batch_size"],
        max_target_length=self.config["max_target_length"],
        num_train_epochs=self.config["num_train_epochs"],
        tokenizer=tokenizer,
        tfds_download=self.config["tfds_download"],
    )

  def maybe_export_model(self, model, model_path: str) -> None:
    lora_config = self.config["model_config"].get("lora_config")
    output_dir = self.config.get("exported_model_output_dir")
    if output_dir is None and lora_config:
      output_dir = self.config.get("merged_model_output_dir")
    if not output_dir:
      return

    model_name = self.config["model_config"]["model_name"]
    logging.info("Saving exported PEFT model to %s", output_dir)
    model_lib.save_model_as_safetensors(
        model_name=model_name,
        local_model_path=model_path,
        output_dir=output_dir,
        model_obj=model,
        lora_config=lora_config,
    )

  def run_peft_trainer(self):
    """Run the PEFT trainer."""
    mesh: jax.sharding.Mesh = self.create_mesh('model_config')
    model: nnx.Module | None = None
    tokenizer: Any | None = None
    my_gen_model_input_fn: (
        Callable[[peft_trainer.TrainingInput], dict[str, Any]] | None
    ) = None
    model, tokenizer_path, model_path = model_lib.create_model(
        self.config['model_config'],
        self.config['tokenizer_config'],
        mesh,
        return_model_path=True,
    )
    if model is None:
      raise ValueError('model is None')
    tokenizer = model_lib.create_tokenizer(
        self.config['tokenizer_config'], tokenizer_path
    )
    optimizer = self.create_optimizer('optimizer_config')
    trainer = peft_trainer.PeftTrainer(
        model,
        optimizer,
        peft_trainer.TrainingConfig(
            **self.obtain_training_config_dict('training_config')
        ),
    )

    def gen_model_input_fn(x: peft_trainer.TrainingInput):
      seq_len = x.input_tokens.shape[-1]
      response_mask = x.input_mask.astype(jax.numpy.int32)
      last_response_from_end = jax.numpy.argmax(
          jax.numpy.flip(response_mask, axis=-1), axis=-1
      )
      valid_lengths = seq_len - last_response_from_end
      valid_lengths = jax.numpy.where(
          jax.numpy.any(response_mask, axis=-1), valid_lengths, 0
      )
      pad_mask = (
          jax.numpy.arange(seq_len)[None, :]
          < valid_lengths[:, None]
      )

      positions = utils.build_positions_from_mask(pad_mask)
      attention_mask = utils.make_causal_attn_mask(pad_mask)
      return {
          'input_tokens': x.input_tokens,
          'input_mask': x.input_mask,
          'positions': positions,
          'attention_mask': attention_mask,
      }

    my_gen_model_input_fn = gen_model_input_fn
    trainer = trainer.with_gen_model_input_fn(my_gen_model_input_fn)

    train_ds, eval_ds = self.create_datasets(tokenizer)

    with mesh:
      trainer.train(train_ds, eval_ds)
      self.maybe_export_model(model, model_path)


def main(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  pipeline.run_peft_trainer()


if __name__ == '__main__':
  app.run(main)
