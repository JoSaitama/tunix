# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampler for sglang-jax-style autoregressive decoding using JAX and NNX models."""

import dataclasses
import math
import os
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from sgl_jax.srt.entrypoints.engine import Engine
from tunix.generate import base_sampler
from tunix.generate import mappings
from tunix.generate import utils
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl import reshard


@dataclasses.dataclass
class SglangJaxConfig:
  model_version: str
  context_length: int
  mesh: jax.sharding.Mesh
  mem_fraction_static: float
  init_with_random_weights: bool
  disable_radix_cache: bool
  enable_deterministic_sampling: bool
  mapping_config: mappings.MappingConfig
  # Note: use_sort_for_toppk_minp may be removed in the future. It depends on SGLang-Jax.
  use_sort_for_toppk_minp: bool = True
  precompile_bs_paddings: Optional[List] = None
  precompile_token_paddings: Optional[List] = None
  chunked_prefill_size: int = -1
  page_size: int = 64
  dtype: str = "auto"
  kv_cache_dtype: str = "auto"


class SglangJaxSampler(base_sampler.BaseSampler):  # pylint: disable=invalid-name
  """A sampler for sglang-jax-style autoregressive decoding using JAX and NNX models.

  This class wraps an NNX model and tokenizer for performing inference
  with optimized KV cache allocation based on available HBM memory.

  Inherits from:
      base_sampler.BaseSampler
  """

  def __init__(
      self,
      tokenizer: Any,
      config: SglangJaxConfig,
  ):
    """Initializes the SglangJaxSampler.

    Args:
        tokenizer (Any): A tokenizer compatible with the model.
        config: The sglang-jax related configurations
    """
    self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.args = self._sglang_jax_config(config)
    self.engine = Engine(**self.args)
    # sglang-jax Engine owns an internal asyncio loop and is not re-entrant
    # across concurrent threads. Guard all engine-facing calls with one lock.
    self._engine_lock = threading.RLock()

    self.mappings = config.mapping_config.to_hf_mappings
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

  # TODO(b/434969743): Optimize weight sharing between trainer and sglang-jax sampler.
  # TODO(b/434975493): Consider Release KV cache on the fly
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    del filter_types
    with self._engine_lock:
      new_state = utils.transfer_state_with_mappings(
          src_state=updated_weights,
          dst_state=self.transformer_state,
          key_mappings=self.mappings,
          transpose_keys=self.to_hf_transpose_keys,
          reshard_fn=reshard.reshard_pytree,
      )
      new_model_state_leaves, _ = jax.tree_util.tree_flatten(new_state)
      self._model_runner.model_state_leaves = new_model_state_leaves

  def load_checkpoint(self, path_or_weights: str | jaxtyping.PyTree):
    # TODO(b/434741253): Consider support orbax checkpoint loading
    if isinstance(path_or_weights, jaxtyping.PyTree):
      self.update_params(updated_weights=path_or_weights, filter_types=None)
    else:
      raise NotImplementedError("Only support in memory weight sync as of now.")

  def _find_tp_size(self, mesh: jax.sharding.Mesh) -> int:
    """Finds the tensor parallel size from the mesh."""
    # since sglang-jax doesn't support DP yet, simply return the total rank size.
    return math.prod(mesh.shape.values())

  def _sglang_jax_config(self, config: SglangJaxConfig):
    args = {}
    args["model_path"] = config.model_version
    args["context_length"] = config.context_length
    args["tp_size"] = self._find_tp_size(config.mesh)
    args["device_indexes"] = config.mesh.device_ids.flatten().tolist()
    args["mem_fraction_static"] = config.mem_fraction_static
    args["enable_single_process"] = True
    # Overlap scheduling can trigger event-loop/thread re-entry issues in
    # agentic rollout usage; keep it disabled for stability.
    args["disable_overlap_schedule"] = True
    if config.disable_radix_cache:
      args["disable_radix_cache"] = True
    if config.enable_deterministic_sampling:
      args["enable_deterministic_sampling"] = True
    if config.init_with_random_weights:
      args["load_format"] = "dummy"
    args["use_sort_for_toppk_minp"] = config.use_sort_for_toppk_minp
    args["precompile_bs_paddings"] = config.precompile_bs_paddings
    args["precompile_token_paddings"] = config.precompile_token_paddings
    args["chunked_prefill_size"] = config.chunked_prefill_size
    args["page_size"] = config.page_size
    args["dtype"] = config.dtype
    args["kv_cache_dtype"] = config.kv_cache_dtype
    return args

  @property
  def _model_runner(self):
    if "scheduler" in self.engine.scheduler_info:
      return self.engine.scheduler_info[
          "scheduler"
      ].tp_worker.worker.model_runner
    else:
      return None

  @property
  def transformer(self):
    # sglang-jax doesn't expose the underlying model
    return None

  @property
  def transformer_state(self):
    return nnx.split(self._model_runner.model)[1]

  def tokenize(self, input_string: str) -> jax.Array | list[int]:
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = (
        [self.tokenizer.bos_id()]
        if (self.tokenizer.bos_id() and input_ids[0] != self.tokenizer.bos_id())
        else []
    )
    eos_tok = (
        [self.tokenizer.eos_id()]
        if input_ids[-1] != self.tokenizer.eos_id()
        else []
    )
    return bos_tok + input_ids + eos_tok

  @staticmethod
  def _normalize_output_ids(
      output_ids: Any, max_generation_steps: int
  ) -> np.ndarray:
    """Normalizes engine output ids into a 1D int32 array."""
    token_ids = np.asarray(output_ids, dtype=np.int32)
    while token_ids.ndim > 1:
      token_ids = np.asarray(token_ids[0], dtype=np.int32)
    if token_ids.ndim == 0:
      token_ids = token_ids.reshape(1)
    if token_ids.shape[0] > max_generation_steps:
      token_ids = token_ids[:max_generation_steps]
    return token_ids

  @staticmethod
  def _normalize_output_text(output_text: Any) -> str:
    """Normalizes engine output text into a scalar string."""
    if isinstance(output_text, list):
      if not output_text:
        return ""
      return str(output_text[0])
    return str(output_text)

  def __call__(
      self,
      input_strings: List[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      beam_size: int | None = None,
      seed: Optional[Union[List[int], int]] = None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
  ) -> base_sampler.SamplerOutput:
    # max_generation_steps: maximum number of tokens to generate
    if (
        self.args["context_length"] is not None
        and max_generation_steps > self.args["context_length"]
    ):
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`context_length`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['context_length']}."
      )

    sampling_params_template = self.engine.get_default_sampling_params()
    sampling_params_template.max_new_tokens = max_generation_steps
    sampling_params_template.n = multi_sampling
    sampling_params_template.temperature = temperature
    sampling_params_template.stop_token_ids = [self.tokenizer.eos_id()]
    sampling_params_template.skip_special_tokens = True

    if top_p is not None:
      sampling_params_template.top_p = top_p
    if top_k is not None:
      sampling_params_template.top_k = top_k
    sampling_params = [
        sampling_params_template.convert_to_dict() for _ in input_strings
    ]
    if seed is not None:
      if type(seed) is List:
        assert len(seed) == len(
            input_strings
        ), "seed and input_strings must have same length"
        for i, seed_i in enumerate(seed):
          sampling_params[i]["sampling_seed"] = seed_i
      else:
        for i, _ in enumerate(input_strings):
          sampling_params[i]["sampling_seed"] = seed

    prompt_ids = [self.tokenize(x) for x in input_strings]
    with self._engine_lock:
      outputs = self.engine.generate(
          input_ids=[ids for ids in prompt_ids],
          sampling_params=sampling_params,
      )

    max_tokens_length = max(len(x) for x in prompt_ids)

    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)
    all_input_ids = [
        utils.pad_to_length(
            np.array(x, dtype=np.int32),
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in prompt_ids
    ]
    all_input_ids = np.array(all_input_ids, dtype=np.int32)

    all_output_ids = [
        utils.pad_to_length(
            self._normalize_output_ids(x["output_ids"], max_generation_steps),
            target_length=max_generation_steps,
            pad_value=self.tokenizer.pad_id(),
            left=False,
        )
        for x in outputs
    ]
    all_output_ids = jnp.array(all_output_ids)
    output_texts = [self._normalize_output_text(o["text"]) for o in outputs]
    # To support multisampling, just return the whole list of SamplerOutput
    return base_sampler.SamplerOutput(
        text=output_texts,
        logits=None,
        tokens=all_output_ids,
        padded_prompt_tokens=all_input_ids,
        logprobs=None,
    )
