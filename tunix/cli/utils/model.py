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
"""Utilities for creating and managing models in Tunix CLI."""

import os
from typing import Any, Tuple

from absl import logging
from flax import nnx
import jax
import qwix
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import automodel
from tunix.rl import reshard

_DEFAULT_TOKENIZER_PATH = 'meta-llama/Llama-3.1-8B'


def _get_model_export_fn(model_name: str, export_fn_name: str):
  """Returns a model-specific safetensors export function."""
  params_modules = [
      automodel.get_model_module(model_name, automodel.ModelModule.PARAMS)
  ]
  if model_name.startswith('gemma'):
    params_modules.append(
        automodel.get_model_module(
            model_name, automodel.ModelModule.PARAMS_SAFETENSORS
        )
    )

  for params_module in params_modules:
    save_fn = getattr(params_module, export_fn_name, None)
    if save_fn is not None:
      return save_fn

  raise AttributeError(
      f'No {export_fn_name} saver found for model {model_name}.'
  )


def apply_lora_to_model(base_model, mesh, lora_config):
  """Apply Lora to the base model if given lora config."""
  logging.info('lora_config %r', lora_config)
  # Basic keyword arguments for LoraProvider
  lora_kwargs = {
      'module_path': lora_config['module_path'],
      'rank': lora_config['rank'],
      'alpha': lora_config['alpha'],
  }
  has_tile_size = 'tile_size' in lora_config
  has_weight_qtype = 'weight_qtype' in lora_config
  if has_tile_size:
    lora_kwargs['tile_size'] = lora_config['tile_size']
  if has_weight_qtype:
    lora_kwargs['weight_qtype'] = lora_config['weight_qtype']
    logging.info('Qlora is applied')
  else:
    logging.info('Lora is applied')

  try:
    lora_provider = qwix.LoraProvider(**lora_kwargs)
  except TypeError as e:
    logging.error(
        'Error initializing qwix.LoraProvider: %s. Kwargs: %s', e, lora_kwargs
    )
    # Depending on desired behavior, you might re-raise or return base_model
    raise

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )
  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)
  return lora_model


def create_tokenizer(tokenizer_config, tokenizer_path: str | None):
  if not tokenizer_path:
    tokenizer_path = tokenizer_config['tokenizer_path']
  tokenizer_type, add_bos, add_eos = (
      tokenizer_config['tokenizer_type'],
      tokenizer_config['add_bos'],
      tokenizer_config['add_eos'],
  )

  return tokenizer_lib.Tokenizer(
      tokenizer_type,
      tokenizer_path,
      add_bos,
      add_eos,
      os.environ.get('HF_TOKEN'),
  )


def create_model(
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    mesh: jax.sharding.Mesh,
    return_model_path: bool = False,
) -> Tuple[nnx.Module, str] | Tuple[nnx.Module, str, str]:
  """Creates a model and determines the tokenizer path based on the model config.

  This function handles model loading from various sources (GCS, Kaggle, HF)
  and applies LoRA if specified in the config.

  Args:
      model_config: A dictionary containing model configuration, including
        'model_name', 'model_source', 'model_id', 'model_download_path',
        'intermediate_ckpt_dir', and optionally 'lora_config'.
      tokenizer_config: A dictionary containing tokenizer configuration,
        including 'tokenizer_path'.
      mesh: The JAX sharding Mesh object.

  Returns:
      A tuple containing:
          - model: The loaded and potentially LoRA-applied nnx.Module.
          - tokenizer_path: The determined path to the tokenizer model.
          - model_path: The resolved local model path, only when
            `return_model_path=True`.
  """
  tokenizer_path: str = tokenizer_config['tokenizer_path']
  model_name = model_config['model_name']
  model_source_str = model_config['model_source']

  # Create Model
  try:
    model_source = automodel.ModelSource(model_source_str)
  except ValueError as exc:
    raise ValueError(
        f'Unsupported model source: {model_source_str}. '
        f'Available sources: {[s.value for s in automodel.ModelSource]}'
    ) from exc

  model, model_path = automodel.AutoModel.from_pretrained(
      model_id=model_config['model_id'],
      mesh=mesh,
      model_source=model_source,
      model_download_path=model_config.get('model_download_path'),
      intermediate_ckpt_dir=model_config.get('intermediate_ckpt_dir'),
      rng_seed=model_config.get('rng_seed', 0),
      model_path=model_config.get('model_path'),
  )

  # Handle Tokenizer Path overrides
  if model_name.startswith('gemma3') and model_source_str == 'gcs':
    # Use the provided tokenizer path, unless it is the default from base_config
    if not tokenizer_path or tokenizer_path == _DEFAULT_TOKENIZER_PATH:
      tokenizer_path = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'
  elif model_name.startswith('gemma') and model_source_str == 'kaggle':
    # Use the provided tokenizer path, unless it is the default from base_config
    if not tokenizer_path or tokenizer_path == _DEFAULT_TOKENIZER_PATH:
      tokenizer_path = os.path.join(model_path, 'tokenizer.model')

  if model_config.get('lora_config'):
    # Apply Lora to model if given lora config
    model = apply_lora_to_model(model, mesh, model_config['lora_config'])
  else:
    logging.info('Training with Full Weight')

  if model_config['model_display']:
    nnx.display(model)

  if return_model_path:
    return model, tokenizer_path, model_path
  return model, tokenizer_path


def save_model_as_safetensors(
    model_name: str,
    local_model_path: str,
    output_dir: str,
    model_obj: nnx.Module,
    lora_config: dict[str, Any] | None = None,
) -> None:
  """Exports a trained model as a safetensors directory."""
  if lora_config:
    save_fn = _get_model_export_fn(
        model_name, 'save_lora_merged_model_as_safetensors'
    )
    save_fn(
        local_model_path=local_model_path,
        output_dir=output_dir,
        lora_model=model_obj,
        rank=lora_config['rank'],
        alpha=lora_config['alpha'],
    )
    return

  save_fn = _get_model_export_fn(model_name, 'save_full_model_as_safetensors')
  save_fn(
      local_model_path=local_model_path,
      output_dir=output_dir,
      model=model_obj,
  )
