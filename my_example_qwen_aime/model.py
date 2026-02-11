from __future__ import annotations

import os

from flax import nnx
from huggingface_hub import snapshot_download
import jax
from jax import numpy as jnp
import qwix
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import safetensors_saver
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params

from .auth import get_hf_token
from .config import LoraConfig, ModelConfig


def resolve_model_config(model_id: str) -> qwen2_lib.ModelConfig:
    lowered = model_id.lower()
    if "qwen2.5-1.5b" in lowered:
        return qwen2_lib.ModelConfig.qwen2p5_1p5b()
    if "deepseek-r1-distill-qwen-1.5b" in lowered:
        return qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()
    if "deepscaler-1.5b-preview" in lowered:
        return qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()
    raise ValueError(f"Unknown or unsupported model id: {model_id}")


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


def load_tokenizer(config: ModelConfig):
    token = get_hf_token()
    tokenizer_source = config.tokenizer_path or config.model_id
    return tokenizer_lib.Tokenizer(
        tokenizer_type="huggingface",
        tokenizer_path=tokenizer_source,
        add_bos=False,
        add_eos=False,
        hf_access_token=token,
    )


def load_model(
    model_path: str,
    model_config: qwen2_lib.ModelConfig,
    mesh,
) -> nnx.Module:
    with mesh:
        return qwen2_params.create_model_from_safe_tensors(
            file_dir=model_path,
            config=model_config,
            mesh=mesh,
            dtype=jnp.float32,
        )


def apply_lora(base_model: nnx.Module, lora: LoraConfig, mesh) -> nnx.Module:
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
            ".*gate_proj|.*down_proj|.*up_proj"
        ),
        rank=lora.rank,
        alpha=lora.alpha,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)

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

        with open(generation_config_path, "r", encoding="utf-8") as handle:
            generation_configs = json.load(handle)
        eos_tokens = generation_configs.get("eos_token_id", [])
        print(f"Using EOS token IDs from generation config: {eos_tokens}")
    return eos_tokens


def _qwen2_state_key_to_safetensors_key(lora_name: str) -> str:
    return f"model.{lora_name}.weight".replace(".attn.", ".self_attn.")


def save_merged_lora(
    model_path: str,
    output_dir: str,
    lora_model: nnx.Module,
    lora: LoraConfig,
) -> None:
    safetensors_saver.save_lora_merged_model_as_safetensors(
        local_model_path=model_path,
        output_dir=output_dir,
        lora_model=lora_model,
        rank=lora.rank,
        alpha=lora.alpha,
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
