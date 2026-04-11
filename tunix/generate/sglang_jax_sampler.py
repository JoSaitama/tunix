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

import asyncio
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
    self._weights_offloaded_to_host = False

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
          key_mapping_hook_fns=self.to_hf_hook_fns,
          transpose_keys=self.to_hf_transpose_keys,
          reshard_fn=self._reshard_params_to_engine,
      )
      new_model_state_leaves, _ = jax.tree_util.tree_flatten(new_state)
      self._model_runner.model_state_leaves = new_model_state_leaves
      self._weights_offloaded_to_host = False

  @staticmethod
  def _reshard_params_to_engine(
      source: jaxtyping.PyTree, dst_shardings: jaxtyping.PyTree
  ) -> jaxtyping.PyTree:
    # Route through host memory to avoid TPU device-order incompatibilities
    # when trainer and rollout meshes enumerate the same devices differently.
    return reshard.reshard_pytree(jax.device_get(source), dst_shardings)

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

  def _cancel_engine_asyncio_tasks(self) -> None:
    """Cancels tokenizer-manager tasks to avoid pending-task warnings on exit."""
    tokenizer_manager = getattr(self.engine, "tokenizer_manager", None)
    loop = getattr(self.engine, "loop", None)
    if tokenizer_manager is None or loop is None or loop.is_closed():
      return

    tasks = list(getattr(tokenizer_manager, "asyncio_tasks", ()))
    if not tasks:
      return
    for task in tasks:
      task.cancel()

    if loop.is_running():
      future = asyncio.run_coroutine_threadsafe(
          asyncio.gather(*tasks, return_exceptions=True), loop
      )
      future.result(timeout=5)
    else:
      loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    tokenizer_manager.asyncio_tasks.clear()

  def close(self) -> None:
    """Releases engine resources and stops its internal event loop."""
    with self._engine_lock:
      self._cancel_engine_asyncio_tasks()
      shutdown = getattr(self.engine, "shutdown", None)
      if callable(shutdown):
        shutdown()

  def flush_cache(self) -> None:
    """Flushes engine-side KV/cache state when rollout is idle."""
    with self._engine_lock:
      flush = getattr(self.engine, "flush_cache", None)
      if callable(flush):
        flush()

  def release_memory_occupation(self) -> None:
    """Moves rollout weights to host so actor training can reclaim TPU HBM."""
    with self._engine_lock:
      model_runner = self._model_runner
      if model_runner is None or getattr(
          self, "_weights_offloaded_to_host", False
      ):
        return
      model = getattr(model_runner, "model", None)
      if model is None:
        return
      model_def, model_state = nnx.split(model)
      host_state = jax.device_get(model_state)
      model_runner.model = nnx.merge(model_def, host_state)
      host_leaves, _ = jax.tree_util.tree_flatten(host_state)
      model_runner.model_state_leaves = host_leaves
      self._weights_offloaded_to_host = True

  def resume_memory_occupation(self) -> None:
    """Restores rollout weights onto device after a host offload."""
    if not getattr(self, "_weights_offloaded_to_host", False):
      return
    self.update_params(updated_weights=self.transformer_state, filter_types=None)

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

  @classmethod
  def _normalize_multisampled_output_ids(
      cls, output_ids: Any, max_generation_steps: int, multi_sampling: int
  ) -> list[np.ndarray]:
    """Normalizes one engine output into 1..N sampled token sequences."""
    if multi_sampling <= 1:
      return [cls._normalize_output_ids(output_ids, max_generation_steps)]
    if output_ids is None:
      return [np.array([], dtype=np.int32) for _ in range(multi_sampling)]
    if isinstance(output_ids, list):
      if output_ids and isinstance(output_ids[0], (list, tuple, np.ndarray)):
        return [
            cls._normalize_output_ids(sample, max_generation_steps)
            for sample in output_ids
        ]
      return [cls._normalize_output_ids(output_ids, max_generation_steps)]

    output_ids_arr = np.asarray(output_ids, dtype=object)
    if output_ids_arr.ndim == 2 and output_ids_arr.shape[0] == multi_sampling:
      return [
          cls._normalize_output_ids(output_ids_arr[i], max_generation_steps)
          for i in range(output_ids_arr.shape[0])
      ]
    return [cls._normalize_output_ids(output_ids, max_generation_steps)]

  @classmethod
  def _normalize_multisampled_output_text(
      cls, output_text: Any, multi_sampling: int
  ) -> list[str]:
    """Normalizes one engine output into 1..N sampled strings."""
    if multi_sampling <= 1:
      return [cls._normalize_output_text(output_text)]
    if isinstance(output_text, list):
      return [str(text) for text in output_text]
    return [str(output_text)]

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
    repeated_prompt_ids: list[list[int]] = []
    output_texts: list[str] = []
    all_output_ids: list[np.ndarray] = []

    # Keep prompt-major, generation-minor ordering so RL callers can index the
    # flattened outputs as `[prompt_i * n + generation_i]`.
    if multi_sampling > 1 and len(outputs) == len(prompt_ids) * multi_sampling:
      for output_i, output in enumerate(outputs):
        prompt_i = output_i // multi_sampling
        repeated_prompt_ids.append(prompt_ids[prompt_i])
        output_texts.append(self._normalize_output_text(output["text"]))
        all_output_ids.append(
            utils.pad_to_length(
                self._normalize_output_ids(
                    output["output_ids"], max_generation_steps
                ),
                target_length=max_generation_steps,
                pad_value=self.tokenizer.pad_id(),
                left=False,
            )
        )
    else:
      if len(outputs) != len(prompt_ids):
        raise ValueError(
            "SGLang-JAX returned an unexpected number of outputs. Expected "
            f"{len(prompt_ids)} prompt outputs, got {len(outputs)}."
        )
      for prompt_token_ids, output in zip(prompt_ids, outputs):
        normalized_output_ids = self._normalize_multisampled_output_ids(
            output.get("output_ids"), max_generation_steps, multi_sampling
        )
        normalized_texts = self._normalize_multisampled_output_text(
            output.get("text"), multi_sampling
        )
        if len(normalized_output_ids) != len(normalized_texts):
          raise ValueError(
              "SGLang-JAX returned mismatched token/text sample counts: "
              f"{len(normalized_output_ids)} token outputs vs "
              f"{len(normalized_texts)} text outputs."
          )
        if multi_sampling > 1 and len(normalized_output_ids) != multi_sampling:
          raise ValueError(
              "SGLang-JAX returned an unexpected number of multi-samples. "
              f"Expected {multi_sampling}, got {len(normalized_output_ids)}."
          )
        for normalized_ids, normalized_text in zip(
            normalized_output_ids, normalized_texts
        ):
          repeated_prompt_ids.append(prompt_token_ids)
          output_texts.append(normalized_text)
          all_output_ids.append(
              utils.pad_to_length(
                  normalized_ids,
                  target_length=max_generation_steps,
                  pad_value=self.tokenizer.pad_id(),
                  left=False,
              )
          )

    all_input_ids = [
        utils.pad_to_length(
            np.array(prompt_token_ids, dtype=np.int32),
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for prompt_token_ids in repeated_prompt_ids
    ]
    all_input_ids = np.array(all_input_ids, dtype=np.int32)
    all_output_ids = jnp.array(all_output_ids)
    return base_sampler.SamplerOutput(
        text=output_texts,
        logits=None,
        tokens=all_output_ids,
        padded_prompt_tokens=all_input_ids,
        logprobs=None,
    )
