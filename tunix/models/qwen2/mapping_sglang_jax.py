"""Mappings for converting Qwen2 weights to the Sglang-jax JAX backend."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


def _repeat_qkv_bias_for_target_shape(
    val: jnp.ndarray,
    tgt_shape: Tuple[int, ...],
    src_key: str,
) -> jnp.ndarray:
  """Repeats Qwen2 q/k/v bias vectors when the rollout TP target is wider."""
  del src_key
  if val.ndim != 1 or len(tgt_shape) != 1 or val.shape == tgt_shape:
    return val
  if val.shape[0] > tgt_shape[0] or tgt_shape[0] % val.shape[0] != 0:
    return val
  repeat_factor = tgt_shape[0] // val.shape[0]
  return jnp.repeat(val, repeat_factor)


def _to_sglang_jax_mappings() -> Dict[str, MappingEntry]:
  return {
      'lm_head.w': ('lm_head.embedding', (None, 'model')),
      'embedder.input_embedding': (
          'model.embed_tokens.embedding',
          ('model', None),
      ),
      'layers.*.input_layernorm.w': (
          'model.layers.*.input_layernorm.scale',
          (None,),
      ),
      'layers.*.mlp.down_proj.kernel': (
          'model.layers.*.mlp.down_proj.weight',
          ('model', None),
      ),
      'layers.*.mlp.gate_proj.kernel': (
          'model.layers.*.mlp.gate_proj.weight',
          (None, 'model'),
      ),
      'layers.*.mlp.up_proj.kernel': (
          'model.layers.*.mlp.up_proj.weight',
          (None, 'model'),
      ),
      'layers.*.post_attention_layernorm.w': (
          'model.layers.*.post_attention_layernorm.scale',
          (None,),
      ),
      'layers.*.attn.k_proj.w': (
          'model.layers.*.self_attn.k_proj.weight',
          (None, 'model', None),
      ),
      'layers.*.attn.k_bias': (
          'model.layers.*.self_attn.k_proj.bias',
          (None,),
      ),
      'layers.*.attn.o_proj.w': (
          'model.layers.*.self_attn.o_proj.weight',
          ('model', None, None),
      ),
      'layers.*.attn.q_proj.w': (
          'model.layers.*.self_attn.q_proj.weight',
          (None, 'model', None),
      ),
      'layers.*.attn.q_bias': (
          'model.layers.*.self_attn.q_proj.bias',
          (None,),
      ),
      'layers.*.attn.v_proj.w': (
          'model.layers.*.self_attn.v_proj.weight',
          (None, 'model', None),
      ),
      'layers.*.attn.v_bias': (
          'model.layers.*.self_attn.v_proj.bias',
          (None,),
      ),
      'final_norm.w': ('model.norm.scale', (None,)),
  }


def _lora_to_sglang_jax_mappings() -> Dict[str, MappingEntry] | None:
  """The lora parameter key mapping between Tunix vanilla model and Sglang-jax Jax backend"""
  return None


def _to_sglang_jax_transpose_keys():
  return {
      'lm_head.w': (1, 0),
  }


def _to_sglang_jax_hook_fns() -> Dict[str, Any] | None:
  """Additional parameter manipulation hook between Tunix vanilla model and Sglang Jax backend"""
  return {
      'layers.*.attn.q_bias': _repeat_qkv_bias_for_target_shape,
      'layers.*.attn.k_bias': _repeat_qkv_bias_for_target_shape,
      'layers.*.attn.v_bias': _repeat_qkv_bias_for_target_shape,
  }


SGLANG_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': _to_sglang_jax_mappings(),
    'lora_to_hf_mappings': _lora_to_sglang_jax_mappings(),
    'to_hf_transpose_keys': _to_sglang_jax_transpose_keys(),
    'to_hf_hook_fns': _to_sglang_jax_hook_fns(),
}

__all__ = [
    'SGLANG_JAX_MAPPING',
]
