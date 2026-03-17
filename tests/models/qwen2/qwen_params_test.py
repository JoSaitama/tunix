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

import os

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen2 import params as qwen2_params
from tunix.tests import lora_params_test_base


class Qwen2ParamsTest(lora_params_test_base.LoraParamsTestBase):
  """Tests for Qwen2 model parameters and LoRA merging."""

  def create_config(self):
    return qwen2_model.ModelConfig(
        num_layers=2,
        vocab_size=256,
        embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        head_dim=32,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=True,
    )

  def get_model_class(self):
    return qwen2_model.Qwen2

  def get_lora_module_path(self) -> str:
    return (
        ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"
    )

  def get_projection_keys(self, layer_idx: int) -> list[str]:
    prefix = f"model.layers.{layer_idx}"
    return [
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]

  def save_merged_model(self, lora_model):
    qwen2_params.save_lora_merged_model_as_safetensors(
        local_model_path=self.base_checkpoint_dir,
        output_dir=self.merged_output_dir,
        lora_model=lora_model,
        rank=self.rank,
        alpha=self.alpha,
    )

  def create_model_from_checkpoint(self, checkpoint_dir: str):
    return qwen2_params.create_model_from_safe_tensors(
        file_dir=checkpoint_dir,
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
    )

  def _create_test_inputs(self):
    batch_size = 2
    seq_len = 10

    input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    attention_mask = jnp.tril(jnp.ones((batch_size, seq_len, seq_len)))
    return input_tokens, positions, attention_mask

  def create_checkpoint(self, model) -> str:
    os.makedirs(self.base_checkpoint_dir, exist_ok=True)

    base_state = {}
    base_state["model.embed_tokens.weight"] = np.array(
        model.embedder.input_embedding.value
    )
    base_state["model.norm.weight"] = np.array(model.final_norm.w.value)

    for layer_idx, layer in enumerate(model.layers):
      prefix = f"model.layers.{layer_idx}"
      base_state[f"{prefix}.input_layernorm.weight"] = np.array(
          layer.input_layernorm.w.value
      )
      base_state[f"{prefix}.post_attention_layernorm.weight"] = np.array(
          layer.post_attention_layernorm.w.value
      )

      base_state[f"{prefix}.self_attn.q_proj.weight"] = np.array(
          layer.attn.q_proj.w.value
      ).reshape(self.config.embed_dim, -1).T
      base_state[f"{prefix}.self_attn.k_proj.weight"] = np.array(
          layer.attn.k_proj.w.value
      ).reshape(self.config.embed_dim, -1).T
      base_state[f"{prefix}.self_attn.v_proj.weight"] = np.array(
          layer.attn.v_proj.w.value
      ).reshape(self.config.embed_dim, -1).T
      base_state[f"{prefix}.self_attn.o_proj.weight"] = np.array(
          layer.attn.o_proj.w.value
      ).transpose(2, 0, 1).reshape(self.config.embed_dim, -1)

      base_state[f"{prefix}.self_attn.q_proj.bias"] = np.array(
          layer.attn.q_bias.value
      )
      base_state[f"{prefix}.self_attn.k_proj.bias"] = np.array(
          layer.attn.k_bias.value
      )
      base_state[f"{prefix}.self_attn.v_proj.bias"] = np.array(
          layer.attn.v_bias.value
      )

      base_state[f"{prefix}.mlp.gate_proj.weight"] = np.array(
          layer.mlp.gate_proj.kernel.value
      ).T
      base_state[f"{prefix}.mlp.up_proj.weight"] = np.array(
          layer.mlp.up_proj.kernel.value
      ).T
      base_state[f"{prefix}.mlp.down_proj.weight"] = np.array(
          layer.mlp.down_proj.kernel.value
      ).T

    safe_np.save_file(
        base_state, os.path.join(self.base_checkpoint_dir, "model.safetensors")
    )
    with open(os.path.join(self.base_checkpoint_dir, "config.json"), "w") as f:
      f.write('{"model_type": "qwen2"}')

    return self.base_checkpoint_dir

  def test_save_full_model_round_trip(self):
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    qwen2_params.save_full_model_as_safetensors(
        local_model_path=self.base_checkpoint_dir,
        output_dir=self.merged_output_dir,
        model=base_model,
    )

    reloaded_model = self.create_model_from_checkpoint(self.merged_output_dir)
    input_tokens, positions, attention_mask = self._create_test_inputs()
    base_output, _ = self._run_forward_pass(
        base_model, input_tokens, positions, attention_mask
    )
    reloaded_output, _ = self._run_forward_pass(
        reloaded_model, input_tokens, positions, attention_mask
    )

    np.testing.assert_allclose(base_output, reloaded_output, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
