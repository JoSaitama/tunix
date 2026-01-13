from __future__ import annotations

import os
from typing import Tuple

from flax import nnx
from huggingface_hub import snapshot_download
import jax
import qwix
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

from .auth import get_hf_token
from .config import ModelConfig, LoraConfig


def resolve_model_config(model_id: str) -> gemma_lib.ModelConfig:
    if "gemma-3-270m" in model_id:
        return (
            gemma_lib.ModelConfig.gemma3_270m_it()
            if model_id.endswith("-it")
            else gemma_lib.ModelConfig.gemma3_270m()
        )
    if "gemma-3-1b" in model_id:
        return (
            gemma_lib.ModelConfig.gemma3_1b_it()
            if model_id.endswith("-it")
            else gemma_lib.ModelConfig.gemma3_1b_pt()
        )
    raise ValueError(f"Unknown model id: {model_id}")


def download_model(config: ModelConfig) -> str:
    token = get_hf_token()
    print(f"Downloading {config.model_id} from Hugging Face...")
    local_model_path = snapshot_download(
        repo_id=config.model_id,
        ignore_patterns=list(config.ignore_patterns),
        token=token,
    )
    print(f"Model successfully downloaded to: {local_model_path}")
    return local_model_path


def load_model(model_path: str, model_config: gemma_lib.ModelConfig, mesh) -> nnx.Module:
    with mesh:
        gemma3 = params_safetensors_lib.create_model_from_safe_tensors(
            model_path, model_config, mesh
        )
    return gemma3


def apply_lora(base_model: nnx.Module, lora: LoraConfig, mesh) -> nnx.Module:
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
            ".*attn_vec_einsum"
        ),
        rank=lora.rank,
        alpha=lora.alpha,
    )

    model_input = base_model.get_model_input()

    # Create a JIT-compiled function to ensure logical shapes are used
    # This prevents the eager execution mismatch where Worker 0 sees global shape (4608)
    # and Worker 1 sees local shard shape (1152).
    # Create a JIT-compiled function to ensure logical shapes are used
    # This prevents the eager execution mismatch where Worker 0 sees global shape (4608)
    # and Worker 1 sees local shard shape (1152).
    # Capture lora_provider via closure to avoid passing it as an argument (it's not a pytree)
    @nnx.jit
    def _apply_lora_jit(model, inputs):
        return qwix.apply_lora_to_model(
            model,
            lora_provider,
            rngs=nnx.Rngs(0),
            **inputs,
        )

    # Run inside JIT
    lora_model = _apply_lora_jit(base_model, model_input)

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


def load_eos_tokens(model_path: str) -> list[int]:
    eos_tokens = []
    generation_config_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(generation_config_path):
        import json

        with open(generation_config_path, "r") as f:
            generation_configs = json.load(f)
        eos_tokens = generation_configs.get("eos_token_id", [])
        print(f"Using EOS token IDs: {eos_tokens}")
    return eos_tokens


def save_merged_lora(
    model_path: str,
    output_dir: str,
    lora_model: nnx.Module,
    lora: LoraConfig,
) -> None:
    gemma_params.save_lora_merged_model_as_safetensors(
        local_model_path=model_path,
        output_dir=output_dir,
        lora_model=lora_model,
        rank=lora.rank,
        alpha=lora.alpha,
    )
