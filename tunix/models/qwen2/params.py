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

"""Utils for loading and converting Qwen2 PT weights."""

import jax
import jax.numpy as jnp
import numpy as np
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.qwen2 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
  return {
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      # attention projection weights
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"layers.\1.attn.q_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"layers.\1.attn.k_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"layers.\1.attn.v_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.o_proj.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
          r"layers.\1.attn.q_bias",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
          r"layers.\1.attn.k_bias",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
          r"layers.\1.attn.v_bias",
          None,
      ),
      # mlp
      r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      # norms
      r"model\.norm\.weight": ("final_norm.w", None),
      # layer norms (pre/post attention)
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.input_layernorm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_layernorm.w",
          None,
      ),
      r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen2:
  """Load tensors from the safetensors file and create a Qwen2 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen2,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=None,
      dtype=dtype,
  )


def _qwen2_state_key_to_safetensors_key(lora_name: str) -> str:
  """Transforms a Qwen2 layer path into a safetensors state key."""
  return f"model.{lora_name}.weight".replace(".attn.", ".self_attn.")


def _build_full_state_dict(model: model_lib.Qwen2) -> dict[str, np.ndarray]:
  """Builds a Hugging Face-compatible safetensors state dict from a Qwen2."""
  cfg = model.config
  state_dict = {
      "model.embed_tokens.weight": np.asarray(
          model.embedder.input_embedding.value
      ),
      "model.norm.weight": np.asarray(model.final_norm.w.value),
  }

  for layer_idx, layer in enumerate(model.layers):
    prefix = f"model.layers.{layer_idx}"
    state_dict[f"{prefix}.input_layernorm.weight"] = np.asarray(
        layer.input_layernorm.w.value
    )
    state_dict[f"{prefix}.post_attention_layernorm.weight"] = np.asarray(
        layer.post_attention_layernorm.w.value
    )
    state_dict[f"{prefix}.self_attn.q_proj.weight"] = np.asarray(
        layer.attn.q_proj.w.value
    ).reshape(cfg.embed_dim, -1).T
    state_dict[f"{prefix}.self_attn.k_proj.weight"] = np.asarray(
        layer.attn.k_proj.w.value
    ).reshape(cfg.embed_dim, -1).T
    state_dict[f"{prefix}.self_attn.v_proj.weight"] = np.asarray(
        layer.attn.v_proj.w.value
    ).reshape(cfg.embed_dim, -1).T
    state_dict[f"{prefix}.self_attn.o_proj.weight"] = np.asarray(
        layer.attn.o_proj.w.value
    ).transpose(2, 0, 1).reshape(cfg.embed_dim, -1)
    state_dict[f"{prefix}.self_attn.q_proj.bias"] = np.asarray(
        layer.attn.q_bias.value
    )
    state_dict[f"{prefix}.self_attn.k_proj.bias"] = np.asarray(
        layer.attn.k_bias.value
    )
    state_dict[f"{prefix}.self_attn.v_proj.bias"] = np.asarray(
        layer.attn.v_bias.value
    )
    state_dict[f"{prefix}.mlp.gate_proj.weight"] = np.asarray(
        layer.mlp.gate_proj.kernel.value
    ).T
    state_dict[f"{prefix}.mlp.up_proj.weight"] = np.asarray(
        layer.mlp.up_proj.kernel.value
    ).T
    state_dict[f"{prefix}.mlp.down_proj.weight"] = np.asarray(
        layer.mlp.down_proj.kernel.value
    ).T

  if not cfg.use_tied_embedding:
    state_dict["lm_head.weight"] = np.asarray(model.lm_head.w.value).T

  return state_dict


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Qwen2,
    rank: int,
    alpha: float,
):
  """Saves a Qwen2 model with LoRA weights merged in safetensors format."""
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=_qwen2_state_key_to_safetensors_key,
      field_patterns=(
          "q_proj",
          "k_proj",
          "v_proj",
          "o_proj",
          "gate_proj",
          "up_proj",
          "down_proj",
      ),
  )


def save_full_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    model: model_lib.Qwen2,
):
  """Saves a full Qwen2 model in Hugging Face-compatible safetensors format."""
  safetensors_saver.save_full_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      model_state=_build_full_state_dict(model),
  )
