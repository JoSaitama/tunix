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

"""sglang jax rollout worker with Tunix sampler."""

from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
from tunix.generate import mappings
from tunix.generate import sglang_jax_sampler
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl import common
from tunix.rl.rollout import base_rollout


class SglangJaxRollout(base_rollout.BaseRollout):
  """sglang jax rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
  ):
    self.mesh = mesh
    self._tokenizer_adapter = tok_adapter.TokenizerAdapter(tokenizer)
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=model,
        backend="sglang_jax",
    )
    self._tokenizer = tokenizer
    self._sampler_config = sglang_jax_sampler.SglangJaxConfig(
        mesh=mesh,
        context_length=rollout_config.rollout_sglang_jax_context_length,
        model_version=rollout_config.rollout_sglang_jax_model_version,
        mem_fraction_static=rollout_config.rollout_sglang_jax_mem_fraction_static,
        init_with_random_weights=rollout_config.rollout_sglang_jax_init_with_random_weights,
        disable_radix_cache=rollout_config.rollout_sglang_jax_disable_radix_cache,
        enable_deterministic_sampling=rollout_config.rollout_sglang_jax_enable_deterministic_sampling,
        mapping_config=mapping_config,
        precompile_bs_paddings=rollout_config.rollout_sglang_jax_precompile_bs_paddings,
        precompile_token_paddings=rollout_config.rollout_sglang_jax_precompile_token_paddings,
        chunked_prefill_size=rollout_config.rollout_sglang_jax_chunked_prefill_size,
        page_size=rollout_config.rollout_sglang_jax_page_size,
        dtype=rollout_config.rollout_sglang_jax_dtype,
        kv_cache_dtype=rollout_config.rollout_sglang_jax_kv_cache_dtype,
    )
    self._sampler = self._create_sampler()
    # Keep only trainable params when syncing trainer -> rollout weights.
    # Non-param states (e.g. RNG streams) can carry a different mesh context.
    self._latest_params = nnx.state(model).filter(nnx.Param)
    self._sampler.load_checkpoint(self._latest_params)

  def _create_sampler(self) -> sglang_jax_sampler.SglangJaxSampler:
    return sglang_jax_sampler.SglangJaxSampler(
        tokenizer=self._tokenizer,
        config=self._sampler_config,
    )

  def _ensure_sampler(self) -> None:
    if self._sampler is not None:
      return
    self._sampler = self._create_sampler()
    if self._latest_params is not None:
      self._sampler.load_checkpoint(self._latest_params)

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    self._ensure_sampler()
    self.output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        multi_sampling=kwargs.get("multi_sampling", 1),
        echo=False,
        pad_output=True,
    )

    return base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    self._ensure_sampler()
    return common.compute_per_token_logps(
        self.model(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
        completion_mask=completion_mask,
    )[0]

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    if hasattr(params, "filter"):
      params = params.filter(nnx.Param)
    self._latest_params = params
    if self._sampler is not None:
      self._sampler.update_params(params, filter_types)

  def pad_id(self) -> int:
    return self._tokenizer_adapter.pad_id()

  def eos_id(self) -> int:
    return self._tokenizer_adapter.eos_id()

  def model(self) -> nnx.Module:
    if self._sampler is None:
      return None
    return self._sampler.transformer

  def close(self) -> None:
    if self._sampler is not None:
      self._sampler.close()
      self._sampler = None

  def flush_cache(self) -> None:
    if self._sampler is not None:
      self._sampler.flush_cache()

  def release_memory_occupation(self) -> None:
    self.close()
