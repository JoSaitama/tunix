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

"""DPO trainer."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from absl import logging
import flax
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
# TODO(abheesht): We should move TokenizerAdapter outside `generate`.
from tunix.generate import tokenizer_adapter
from tunix.rl import common
from tunix.sft import peft_trainer
from tunix.sft import profiler
from tunix.sft import progress_bar
from tunix.sft import sharding_utils
from tunix.sft import utils as sft_utils
from typing_extensions import override


@flax.struct.dataclass(frozen=True)
class DataInput:
  """Training data input for DPO.

  This can be used when inputs are raw strings. Tokenization, padding and
  preprocessing is taken care of by `DPOTrainer`.

  Attributes:
    prompts: A list of prompts.
    chosen_responses: A list of chosen responses.
    rejected_responses: A list of rejected responses.
  """

  prompts: list[str]
  chosen_responses: list[str]
  rejected_responses: list[str]


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  """Tokenized training input for DPO.

  This can be used when inputs are already tokenized, padded and preprocessed.

  Attributes:
    prompt_ids: Prompt IDs. Should be left-padded.
    prompt_mask: Prompt mask. Should be left-padded.
    chosen_ids: Chosen response IDs. Should be right-padded.
    chosen_mask: Chosen response mask. Should be right-padded.
    rejected_ids: Rejected response IDs. Should be right-padded.
    rejected_mask: Rejected response mask. Should be right-padded.
  """

  # Prompt IDs should be left padded.
  prompt_ids: jax.Array | np.ndarray
  prompt_mask: jax.Array | np.ndarray
  # Chosen IDs should be right padded.
  chosen_ids: jax.Array | np.ndarray
  chosen_mask: jax.Array | np.ndarray
  # Rejected IDs should be right padded.
  rejected_ids: jax.Array | np.ndarray
  rejected_mask: jax.Array | np.ndarray


@flax.struct.dataclass(frozen=True)
class TrainExample:
  input_ids: jax.Array  # Concatenated [prompt_ids, completion_ids]
  positions: jax.Array
  attention_mask: jax.Array
  ref_chosen_logps: jax.Array | None
  ref_rejected_logps: jax.Array | None
  completion_mask: jax.Array
  logits_to_keep: int = flax.struct.field(pytree_node=False)


@dataclasses.dataclass(slots=True, kw_only=True)
class DPOTrainingConfig(peft_trainer.TrainingConfig):
  """DPO/ORPO Training Config."""

  algorithm: str = "dpo"  # "dpo" or "orpo"
  beta: float = (
      0.1  # 𝛽 for KL penalty (DPO only) https://arxiv.org/pdf/2305.18290
  )
  lambda_orpo: float = 0.1  # Weight for preference loss (ORPO only)
  label_smoothing: float = 0.0

  # Should be specified only if your input has strings instead of tokenized IDs.
  max_prompt_length: int | None = None
  max_response_length: int | None = None
  use_dynamic_batch_curation: bool = False
  curation_variant: str = "outlier_l2"
  curation_threshold: float = 3.0
  self_influence_dot_threshold: float = 0.0

  def __post_init__(self) -> None:
    self.curation_variant = _normalize_curation_variant(self.curation_variant)


@nnx.jit(static_argnums=(4,))
def compute_logps(
    model,
    input_ids,
    positions,
    attention_mask,
    logits_to_keep,
    completion_mask,
):
  """Computes the log probabilities for chosen and rejected tokens."""
  token_logps = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
  )
  token_logps = (token_logps * completion_mask).sum(axis=-1)

  batch_size = token_logps.shape[0]
  chosen_logps = token_logps[: batch_size // 2]
  rejected_logps = token_logps[batch_size // 2 :]
  return chosen_logps, rejected_logps


def _stack_preference_pairs(train_example: TrainExample) -> TrainExample:
  """Regroups concatenated chosen/rejected rows into per-example pairs."""
  batch_size = train_example.input_ids.shape[0] // 2

  def stack_pairs(value):
    if value is None:
      return None
    return jnp.stack([value[:batch_size], value[batch_size:]], axis=1)

  return TrainExample(
      input_ids=stack_pairs(train_example.input_ids),
      positions=stack_pairs(train_example.positions),
      attention_mask=stack_pairs(train_example.attention_mask),
      ref_chosen_logps=train_example.ref_chosen_logps,
      ref_rejected_logps=train_example.ref_rejected_logps,
      completion_mask=stack_pairs(train_example.completion_mask),
      logits_to_keep=train_example.logits_to_keep,
  )


def dpo_loss_fn_per_sample(
    model: nnx.Module,
    train_example: TrainExample,
    algorithm: str = "dpo",
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Returns unreduced per-sample DPO/ORPO losses and aux metrics."""
  chosen_logps, rejected_logps = compute_logps(
      model,
      train_example.input_ids,
      train_example.positions,
      train_example.attention_mask,
      train_example.logits_to_keep,
      train_example.completion_mask,
  )

  if algorithm == "orpo":
    batch_size = train_example.completion_mask.shape[0] // 2
    chosen_mask = train_example.completion_mask[:batch_size]
    chosen_lengths = jnp.maximum(chosen_mask.sum(axis=-1), 1.0)
    sft_loss = -chosen_logps / chosen_lengths
    log_odds = (chosen_logps - rejected_logps) - (
        jnp.log1p(-jnp.exp(chosen_logps)) - jnp.log1p(-jnp.exp(rejected_logps))
    )
    or_loss = -(
        jax.nn.log_sigmoid(log_odds) * (1 - label_smoothing)
        + jax.nn.log_sigmoid(-log_odds) * label_smoothing
    )
    total_loss = sft_loss + lambda_orpo * or_loss
    chosen_rewards = lambda_orpo * chosen_logps
    rejected_rewards = lambda_orpo * rejected_logps
    odds_ratio = jnp.exp(log_odds)

    return total_loss, {
        "rewards/chosen": chosen_rewards,
        "rewards/rejected": rejected_rewards,
        "rewards/margin": chosen_rewards - rejected_rewards,
        "rewards/accuracy": (chosen_rewards > rejected_rewards).astype(
            jnp.float32
        ),
        "log_probs/chosen": chosen_logps,
        "log_probs/rejected": rejected_logps,
        "odds_ratio": odds_ratio,
        "sft_loss": sft_loss,
        "or_loss": or_loss,
    }

  chosen_log_ratio = chosen_logps
  if train_example.ref_chosen_logps is not None:
    chosen_log_ratio = chosen_log_ratio - train_example.ref_chosen_logps
  rejected_log_ratio = rejected_logps
  if train_example.ref_rejected_logps is not None:
    rejected_log_ratio = rejected_log_ratio - train_example.ref_rejected_logps
  delta = chosen_log_ratio - rejected_log_ratio
  losses = -(
      jax.nn.log_sigmoid(beta * delta) * (1 - label_smoothing)
      + jax.nn.log_sigmoid(-beta * delta) * label_smoothing
  )

  chosen_rewards = beta * chosen_log_ratio
  rejected_rewards = beta * rejected_log_ratio
  return losses, {
      "rewards/chosen": chosen_rewards,
      "rewards/rejected": rejected_rewards,
      "rewards/margin": chosen_rewards - rejected_rewards,
      "rewards/accuracy": (chosen_rewards > rejected_rewards).astype(
          jnp.float32
      ),
      "log_probs/chosen": chosen_logps,
      "log_probs/rejected": rejected_logps,
  }


def _masked_mean(values: jax.Array, weights: jax.Array) -> jax.Array:
  """Computes the mean of batched values using 1-D sample weights."""
  broadcast_weights = weights.reshape(
      (weights.shape[0],) + (1,) * (values.ndim - 1)
  )
  return jnp.sum(values * broadcast_weights, axis=0) / jnp.maximum(
      weights.sum(), 1.0
  )


def _normalize_curation_variant(variant: str) -> str:
  """Normalizes user-facing DBC variant names to canonical DPO values."""
  normalized_variant = variant.strip().lower().replace("-", "_")
  alias_map = {
      "outlier_l2": "outlier_l2",
      "self_inf_batch": "self_inf_batch",
      "self_influence_batch": "self_inf_batch",
  }
  if normalized_variant not in alias_map:
    raise ValueError(
        "curation_variant must be one of "
        "'outlier_l2', 'outlier-l2', 'self_inf_batch', "
        "'self-inf-batch', or 'self_influence_batch'. "
        f"Received: {variant!r}"
    )
  return alias_map[normalized_variant]


def _finite_mean_and_std(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns a finite mask together with finite-only mean/std statistics."""
  finite_mask = jnp.isfinite(values)
  finite_weights = finite_mask.astype(jnp.float32)
  safe_values = jnp.where(finite_mask, values, 0.0)
  finite_count = jnp.maximum(finite_weights.sum(), 1.0)
  mean_value = safe_values.sum() / finite_count
  centered_values = jnp.where(finite_mask, values - mean_value, 0.0)
  std_value = jnp.sqrt(jnp.sum(centered_values**2) / finite_count)
  return finite_mask, mean_value, std_value


def _fallback_keep_mask(finite_mask: jax.Array) -> jax.Array:
  """Returns a non-empty keep mask, preferring finite samples."""
  return jnp.where(
      jnp.any(finite_mask),
      finite_mask,
      jnp.ones_like(finite_mask, dtype=bool),
  )


def _dbc_common_aux(
    *,
    keep_weights: jax.Array,
    total_samples: jax.Array,
    grad_norm_mean: jax.Array,
    grad_norm_std: jax.Array,
) -> dict[str, jax.Array]:
  kept_samples = keep_weights.sum()
  filtered_samples = total_samples - kept_samples
  return {
      "dbc/num_samples_total": total_samples,
      "dbc/num_samples_kept": kept_samples,
      "dbc/num_samples_filtered": filtered_samples,
      "dbc/keep_ratio": kept_samples / jnp.maximum(total_samples, 1.0),
      "dbc/grad_norm_mean": grad_norm_mean,
      "dbc/grad_norm_std": grad_norm_std,
  }


def aggregate_curated_step(
    per_sample_grads: Any,
    per_sample_losses: jax.Array,
    per_sample_aux: dict[str, jax.Array],
    per_sample_grad_norms: jax.Array,
    curation_threshold: float,
) -> tuple[Any, jax.Array, dict[str, jax.Array]]:
  """Applies outlier filtering over a full accumulation window."""
  finite_mask, mean_norm, std_norm = _finite_mean_and_std(per_sample_grad_norms)
  cutoff = mean_norm + curation_threshold * std_norm
  keep_mask = finite_mask & (per_sample_grad_norms <= cutoff)
  fallback_mask = _fallback_keep_mask(finite_mask)
  keep_mask = jnp.where(jnp.any(keep_mask), keep_mask, fallback_mask)
  keep_weights = keep_mask.astype(jnp.float32)

  final_grads = jax.tree.map(
      lambda grad: _masked_mean(grad, keep_weights), per_sample_grads
  )
  final_loss = _masked_mean(per_sample_losses, keep_weights)
  final_aux = jax.tree.map(
      lambda value: _masked_mean(value, keep_weights), per_sample_aux
  )
  final_aux = dict(final_aux)
  total_samples = jnp.array(per_sample_grad_norms.shape[0], dtype=jnp.float32)
  final_aux.update(
      _dbc_common_aux(
          keep_weights=keep_weights,
          total_samples=total_samples,
          grad_norm_mean=mean_norm,
          grad_norm_std=std_norm,
      )
  )
  final_aux["dbc/grad_norm_cutoff"] = cutoff
  return final_grads, final_loss, final_aux


def _self_influence_batch_scores(per_sample_grads: Any) -> jax.Array:
  """Computes per-sample dot products with the batch mean gradient."""
  mean_grads = jax.tree.map(lambda grad: jnp.mean(grad, axis=0), per_sample_grads)

  def leaf_score(grad, mean_grad):
    reduce_axes = tuple(range(1, grad.ndim))
    return jnp.sum(grad * mean_grad, axis=reduce_axes)

  scores = jax.tree.map(leaf_score, per_sample_grads, mean_grads)
  return jax.tree_util.tree_reduce(lambda left, right: left + right, scores)


def aggregate_self_influence_curated_step(
    per_sample_grads: Any,
    per_sample_losses: jax.Array,
    per_sample_aux: dict[str, jax.Array],
    per_sample_grad_norms: jax.Array,
    dot_threshold: float,
) -> tuple[Any, jax.Array, dict[str, jax.Array]]:
  """Applies self-influence filtering over a full accumulation window."""
  grad_norm_finite_mask, grad_norm_mean, grad_norm_std = _finite_mean_and_std(
      per_sample_grad_norms
  )
  self_inf_scores = _self_influence_batch_scores(per_sample_grads)
  score_finite_mask, score_mean, score_std = _finite_mean_and_std(
      self_inf_scores
  )
  finite_mask = grad_norm_finite_mask & score_finite_mask
  keep_mask = finite_mask & (self_inf_scores >= dot_threshold)
  fallback_mask = _fallback_keep_mask(finite_mask)
  keep_mask = jnp.where(jnp.any(keep_mask), keep_mask, fallback_mask)
  keep_weights = keep_mask.astype(jnp.float32)

  final_grads = jax.tree.map(
      lambda grad: _masked_mean(grad, keep_weights), per_sample_grads
  )
  final_loss = _masked_mean(per_sample_losses, keep_weights)
  final_aux = jax.tree.map(
      lambda value: _masked_mean(value, keep_weights), per_sample_aux
  )
  final_aux = dict(final_aux)
  total_samples = jnp.array(per_sample_grad_norms.shape[0], dtype=jnp.float32)
  final_aux.update(
      _dbc_common_aux(
          keep_weights=keep_weights,
          total_samples=total_samples,
          grad_norm_mean=grad_norm_mean,
          grad_norm_std=grad_norm_std,
      )
  )
  final_aux.update({
      "dbc/self_inf_dot_mean": score_mean,
      "dbc/self_inf_dot_std": score_std,
      "dbc/self_inf_dot_threshold": jnp.array(dot_threshold, dtype=jnp.float32),
  })
  return final_grads, final_loss, final_aux


class DPOTrainer(peft_trainer.PeftTrainer):
  """Direct Preference Optimization (DPO) and ORPO trainer.

  DPO is a preference tuning method for aligning large language models with
  human or AI preferences. It is a more efficient, performant alternative
  to RLHF.

  DPO is simpler because it eliminates the need for text generation in the
  training loop. Moreover, DPO bypasses the reward modeling step entirely, i.e.,
  we do not need to train a separate reward model. It uses a dataset of
  preferences (pairs of "chosen" and "rejected responses) to directly optimize
  the policy model by using a classification-style loss.

  ORPO (Odds Ratio Preference Optimization) is a memory-efficient variant that
  combines supervised fine-tuning with preference alignment without requiring
  a separate reference model, making it approximately 50% more memory-efficient.

  References:
  - DPO: https://arxiv.org/abs/2305.18290
  - ORPO: https://arxiv.org/abs/2403.07691
  """

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module | None,
      optimizer: optax.GradientTransformation,
      training_config: DPOTrainingConfig,
      tokenizer: Any | None = None,
  ):
    """Initializes the DPO/ORPO trainer.

    Args:
      model: The policy model to be trained.
      ref_model: The reference/anchor model which is kept fixed/frozen during
        training (DPO only). It is used to prevent the policy model from
        drifting too far from its original capabilities. For ORPO, this should
        be None. If `ref_model` is None for DPO, we don't use it in the loss
        term.
      optimizer: The optimizer used for training the policy model.
      training_config: A `DPOTrainingConfig` object containing DPO/ORPO-specific
        hyperparameters like `beta`, `lambda_orpo`, and `label_smoothing`.
      tokenizer: An optional tokenizer. If provided, the trainer can accept
        string inputs and tokenize them internally.
    """
    self.model = model
    self.ref_model = ref_model
    self.dpo_config = training_config
    self.algorithm = training_config.algorithm
    super().__init__(model, optimizer, training_config)
    self._configure_dpo_runtime(tokenizer)

  def _configure_dpo_runtime(self, tokenizer: Any | None) -> None:
    """Configures DPO-specific loss plumbing and metric logging."""
    self.tokenizer = (
        None
        if tokenizer is None
        else tokenizer_adapter.TokenizerAdapter(tokenizer)
    )

    self.with_loss_fn(dpo_loss_fn, has_aux=True)

    if self.algorithm == "orpo":
      self.with_gen_model_input_fn(
          lambda x: {
              "train_example": x,
              "algorithm": "orpo",
              "lambda_orpo": self.dpo_config.lambda_orpo,
              "label_smoothing": self.dpo_config.label_smoothing,
          }
      )
      self.gen_model_input_fn = lambda x: {
          "train_example": x,
          "algorithm": "orpo",
          "lambda_orpo": self.dpo_config.lambda_orpo,
          "label_smoothing": self.dpo_config.label_smoothing,
      }
    else:
      self.with_gen_model_input_fn(
          lambda x: {
              "train_example": x,
              "algorithm": "dpo",
              "beta": self.dpo_config.beta,
              "label_smoothing": self.dpo_config.label_smoothing,
          }
      )
      self.gen_model_input_fn = lambda x: {
          "train_example": x,
          "algorithm": "dpo",
          "beta": self.dpo_config.beta,
          "label_smoothing": self.dpo_config.label_smoothing,
      }

    self._has_aux = True

    # If reference model is not provided, we don't use it in the loss term.
    self._ref_model_exists = self.ref_model is not None

    self._aux_metrics_to_log = {
        "rewards/chosen": np.mean,
        "rewards/rejected": np.mean,
        "rewards/margin": np.mean,
        "rewards/accuracy": np.mean,
        "log_probs/chosen": np.mean,
        "log_probs/rejected": np.mean,
    }

    if self.algorithm == "orpo":
      self._aux_metrics_to_log["odds_ratio"] = np.mean

  @override
  def _prepare_inputs(
      self,
      training_input: dict[str, Any] | DataInput | TrainingInput,
  ) -> Any:
    if isinstance(training_input, dict):
      training_input = _preprocess_dict(training_input)

    # If the inputs are list of strings, let's tokenise them and pad them.
    if isinstance(training_input, DataInput):
      if self.tokenizer is None:
        raise ValueError(
            "Tokenizer must be provided if training input is not tokenized."
        )

      max_prompt_length = self.dpo_config.max_prompt_length
      max_response_length = self.dpo_config.max_response_length
      if (
          self.dpo_config.max_prompt_length is None
          or self.dpo_config.max_response_length is None
      ):
        raise ValueError(
            "max_prompt_length and max_response_length must be provided if "
            "training input is not tokenized. Received: "
            f"max_prompt_length={max_prompt_length}, "
            f"max_response_length={max_response_length}."
        )

      training_input = process_dpo_record(
          record={
              "prompts": training_input.prompts,
              "chosen_responses": training_input.chosen_responses,
              "rejected_responses": training_input.rejected_responses,
          },
          tokenizer=self.tokenizer,
          max_prompt_length=self.dpo_config.max_prompt_length,
          max_response_length=self.dpo_config.max_response_length,
      )

    # Concatenate chosen and rejected IDs so we can do a forward pass together.
    prompt_ids = jnp.concatenate(
        [training_input.prompt_ids, training_input.prompt_ids], axis=0
    )
    prompt_mask = jnp.concatenate(
        [training_input.prompt_mask, training_input.prompt_mask], axis=0
    )
    completion_ids = jnp.concatenate(
        [training_input.chosen_ids, training_input.rejected_ids], axis=0
    )
    completion_mask = jnp.concatenate(
        [training_input.chosen_mask, training_input.rejected_mask], axis=0
    )
    input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)

    # Compute positions, attention mask, etc., to be fed to the model.
    mask = jnp.concat([prompt_mask, completion_mask], axis=1)
    attention_mask = common.make_causal_attn_mask(mask)
    logits_to_keep = completion_ids.shape[1]
    positions = common.build_positions_from_mask(mask)

    # Compute the log probabilities for the chosen and rejected tokens.
    ref_chosen_logps = None
    ref_rejected_logps = None
    if self._ref_model_exists:
      ref_chosen_logps, ref_rejected_logps = compute_logps(
          self.ref_model,
          input_ids,
          positions,
          attention_mask,
          logits_to_keep,
          completion_mask,
      )
    return TrainExample(
        input_ids=input_ids,
        positions=positions,
        attention_mask=attention_mask,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        completion_mask=completion_mask,
        logits_to_keep=logits_to_keep,
    )

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    assert self._buffered_train_metrics is not None
    for metric_name, op in self._aux_metrics_to_log.items():
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    assert self._buffered_eval_metrics is not None
    for metric_name, op in self._aux_metrics_to_log.items():
      if metric_name not in self._buffered_eval_metrics.additional_metrics:
        self._buffered_eval_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_eval_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )


class CuratedDPOTrainer(DPOTrainer):
  """DPO trainer with curation applied across a full accumulation window."""

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module | None,
      optimizer: optax.GradientTransformation,
      training_config: DPOTrainingConfig,
      tokenizer: Any | None = None,
  ):
    self.model = model
    self.ref_model = ref_model
    self.dpo_config = training_config
    self.algorithm = training_config.algorithm
    self._gradient_accumulation_steps = training_config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    self.curation_variant = training_config.curation_variant
    self.curation_threshold = training_config.curation_threshold
    self.self_influence_dot_threshold = (
        training_config.self_influence_dot_threshold
    )
    manual_config = dataclasses.replace(
        training_config, gradient_accumulation_steps=None
    )
    peft_trainer.PeftTrainer.__init__(self, model, optimizer, manual_config)
    self.config = training_config
    self.dpo_config = training_config
    self.algorithm = training_config.algorithm
    self._iter_steps = self._train_steps * self._gradient_accumulation_steps
    max_step = None
    if self.config.max_steps is not None:
      max_step = self.config.max_steps * self._gradient_accumulation_steps
    self._prof = profiler.Profiler(
        initial_step=self._iter_steps,
        max_step=max_step,
        profiler_options=self.config.profiler_options,
    )
    self._configure_dpo_runtime(tokenizer)
    self._dbc_aux_metrics_to_log = {
        "dbc/num_samples_total": np.mean,
        "dbc/num_samples_kept": np.mean,
        "dbc/num_samples_filtered": np.mean,
        "dbc/keep_ratio": np.mean,
        "dbc/grad_norm_mean": np.mean,
        "dbc/grad_norm_std": np.mean,
        "dbc/grad_norm_cutoff": np.mean,
        "dbc/self_inf_dot_mean": np.mean,
        "dbc/self_inf_dot_std": np.mean,
        "dbc/self_inf_dot_threshold": np.mean,
    }
    self._jitted_microbatch_train_step_fn = None
    self._jitted_curated_apply_step_fn = None

  @override
  def clear_jit_cache(self):
    super().clear_jit_cache()
    self._jitted_microbatch_train_step_fn = None
    self._jitted_curated_apply_step_fn = None

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    super()._post_process_train_step(aux)
    assert self._buffered_train_metrics is not None
    for metric_name, op in self._dbc_aux_metrics_to_log.items():
      if metric_name not in aux:
        continue
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  def _compute_microbatch_grads(
      self, model: nnx.Module, inputs: Any
  ) -> tuple[jax.Array, dict[str, jax.Array], Any, jax.Array]:
    """Computes per-sample grads for one micro-batch without updating params."""
    model_inputs = self.gen_model_input_fn(inputs)
    train_example = _stack_preference_pairs(model_inputs["train_example"])

    def per_example_loss_fn(model, train_example):
      losses, aux = dpo_loss_fn_per_sample(
          model,
          train_example,
          algorithm=model_inputs["algorithm"],
          beta=model_inputs.get("beta", 0.1),
          lambda_orpo=model_inputs.get("lambda_orpo", 0.1),
          label_smoothing=model_inputs.get("label_smoothing", 0.0),
      )
      return losses[0], jax.tree.map(lambda value: value[0], aux)

    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    grad_fn = nnx.value_and_grad(
        per_example_loss_fn,
        argnums=nnx.DiffState(0, wrt),
        has_aux=True,
    )
    train_example_in_axes = jax.tree.map(lambda _: 0, train_example)
    (per_sample_losses, per_sample_aux), per_sample_grads = jax.vmap(
        grad_fn, in_axes=(None, train_example_in_axes)
    )(model, train_example)
    per_sample_grad_norms = jax.vmap(optax.global_norm)(per_sample_grads)
    return per_sample_losses, per_sample_aux, per_sample_grads, per_sample_grad_norms

  def _apply_curated_grads(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      per_sample_grads: Any,
      per_sample_losses: jax.Array,
      per_sample_aux: dict[str, jax.Array],
      per_sample_grad_norms: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Filters samples across the window and applies one update."""
    if self.curation_variant == "self_inf_batch":
      final_grads, final_loss, final_aux = (
          aggregate_self_influence_curated_step(
              per_sample_grads=per_sample_grads,
              per_sample_losses=per_sample_losses,
              per_sample_aux=per_sample_aux,
              per_sample_grad_norms=per_sample_grad_norms,
              dot_threshold=self.self_influence_dot_threshold,
          )
      )
    else:
      final_grads, final_loss, final_aux = aggregate_curated_step(
          per_sample_grads=per_sample_grads,
          per_sample_losses=per_sample_losses,
          per_sample_aux=per_sample_aux,
          per_sample_grad_norms=per_sample_grad_norms,
          curation_threshold=self.curation_threshold,
      )
    optimizer.update(model, final_grads)
    return final_loss, final_aux

  def _jit_train_and_eval_step(self, skip_jit: bool = False):
    eval_step = self.create_eval_step_fn()
    if skip_jit:
      return self._compute_microbatch_grads, self._apply_curated_grads, eval_step

    if self._jitted_microbatch_train_step_fn is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      self._jitted_microbatch_train_step_fn = nnx.jit(
          self._compute_microbatch_grads
      )
      self._jitted_curated_apply_step_fn = nnx.jit(
          self._apply_curated_grads, donate_argnames=("optimizer",)
      )
      self._jitted_eval_step_fn = nnx.jit(eval_step)
    return (
        self._jitted_microbatch_train_step_fn,
        self._jitted_curated_apply_step_fn,
        self._jitted_eval_step_fn,
    )

  @override
  def train(
      self,
      train_ds: Any,
      eval_ds: Any | None = None,
      skip_jit: bool = False,
      *,
      cache_nnx_graph: bool = False,
  ) -> None:
    micro_step, apply_step, eval_step = self._jit_train_and_eval_step(skip_jit)
    if not skip_jit:
      logging.info(
          "Training with curated DPO mesh: %s. Compiled micro_step cache size: %s",
          pxla.thread_resources.env.physical_mesh,
          micro_step.jitted_fn._cache_size(),  # pytype: disable=attribute-error,protected-access
      )

    if cache_nnx_graph:
      partial_micro_step = nnx.cached_partial(micro_step, self.model)
      partial_apply_step = nnx.cached_partial(
          apply_step, self.model, self.optimizer
      )
      partial_eval_step = nnx.cached_partial(eval_step, self.model)
    else:
      partial_micro_step = lambda inputs: micro_step(self.model, inputs)
      partial_apply_step = lambda grads, losses, aux, norms: apply_step(
          self.model, self.optimizer, grads, losses, aux, norms
      )
      partial_eval_step = lambda inputs: eval_step(self.model, inputs)

    if eval_ds:
      self._run_eval(eval_ds, partial_eval_step)

    if self.config.max_steps is not None and self._pbar is None:
      self._pbar = progress_bar.ProgressBar(
          metrics_prefix=self.metrics_prefix,
          metrics_logger=self.metrics_logger,
          initial_steps=self._train_steps,
          max_steps=self.config.max_steps,
          description=self.config.pbar_description,
      )

    if self.training_hooks:
      self.training_hooks.on_train_start(self)

    train_iterator = iter(train_ds)
    index = 0
    grad_window = []
    loss_window = []
    aux_window = []
    grad_norm_window = []
    last_step_completion_time = time.perf_counter()

    with sft_utils.time_measure("Train loop"):
      while True:
        self._prof.maybe_activate(self._iter_steps)
        with jax.profiler.StepTraceAnnotation(
            "train", step_num=self._iter_steps
        ):
          train_example = None
          if self.data_hooks:
            train_example = self.data_hooks.load_next_train_batch(self)
          else:
            try:
              train_example = next(train_iterator)
              if not self.is_managed_externally:
                if index < self._iter_steps:
                  index += 1
                  continue
              index += 1
            except StopIteration:
              pass

          if train_example is None:
            break
          if (
              self.config.max_steps is not None
              and self._train_steps >= self.config.max_steps
          ):
            break

          train_example = self._prepare_inputs(train_example)
          train_example = sharding_utils.shard_input(
              train_example, self.config.data_sharding_axis
          )

          if self.training_hooks:
            self.training_hooks.on_train_step_start(self)

          per_sample_loss, per_sample_aux, per_sample_grads, per_sample_grad_norms = (
              partial_micro_step(train_example)
          )
          grad_window.append(per_sample_grads)
          loss_window.append(per_sample_loss)
          aux_window.append(per_sample_aux)
          grad_norm_window.append(per_sample_grad_norms)
          self._iter_steps += 1

          if self._iter_steps % self._gradient_accumulation_steps != 0:
            continue

          window_grads = jax.tree.map(
              lambda *xs: jnp.concatenate(xs, axis=0), *grad_window
          )
          window_losses = jnp.concatenate(loss_window, axis=0)
          window_aux = jax.tree.map(
              lambda *xs: jnp.concatenate(xs, axis=0), *aux_window
          )
          window_grad_norms = jnp.concatenate(grad_norm_window, axis=0)
          train_loss, aux = partial_apply_step(
              window_grads,
              window_losses,
              window_aux,
              window_grad_norms,
          )
          grad_window.clear()
          loss_window.clear()
          aux_window.clear()
          grad_norm_window.clear()

          current_time = time.perf_counter()
          step_time_delta = current_time - last_step_completion_time
          last_step_completion_time = current_time

          self._buffered_train_metrics = self._buffer_metrics(
              self._buffered_train_metrics,
              loss=train_loss,
              step=self._train_steps,
              step_time_delta=step_time_delta,
          )
          self._post_process_train_step(aux)
          self._train_steps += 1
          self._write_train_metrics()
          self.checkpoint_manager.save(
              self._train_steps,
              self.model,
              save_only_lora_params=self._lora_enabled,
              custom_metadata=self.custom_checkpoint_metadata(),
          )
          if eval_ds and self._train_steps % self.config.eval_every_n_steps == 0:
            self._run_eval(eval_ds, partial_eval_step)

        self._prof.maybe_deactivate(self._iter_steps)

    if grad_window:
      logging.warning(
          "Dropping %d pending micro-batches because the accumulation window "
          "was incomplete at trainer shutdown.",
          len(grad_window),
      )
    if self.training_hooks:
      self.training_hooks.on_train_end(self)
    if not self.is_managed_externally:
      self.close()


def dpo_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    algorithm: str = "dpo",
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """DPO/ORPO loss function.

  Args:
    model: The model to compute loss for.
    train_example: Training example containing input_ids, masks, etc.
    algorithm: "dpo" or "orpo".
    beta: Weight for KL penalty (DPO only).
    lambda_orpo: Weight for preference loss (ORPO only).
    label_smoothing: Label smoothing factor.

  Returns:
    A tuple of (loss, auxiliary_metrics_dict).
  """
  losses, aux = dpo_loss_fn_per_sample(
      model,
      train_example,
      algorithm=algorithm,
      beta=beta,
      lambda_orpo=lambda_orpo,
      label_smoothing=label_smoothing,
  )
  return losses.mean(), jax.tree.map(jnp.mean, aux)


def _generate_ids_and_masks(
    input_strings: list[str],
    tokenizer: Any,
    max_length: int,
    left_pad: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Generates ids and masks for a list of strings."""
  tokens = [_tokenize(x, tokenizer) for x in input_strings]
  all_input_ids = jnp.array([
      common.pad_to_length(
          x[:max_length],
          target_length=max_length,
          pad_value=tokenizer.pad_id(),
          left=left_pad,
          axis=-1,
      )
      for x in tokens
  ])
  # generate masks
  all_input_mask = (all_input_ids != tokenizer.pad_id()).astype("int32")
  return all_input_ids, all_input_mask


def _tokenize(input_string: str, tokenizer: Any) -> jax.Array:
  """Tokenizes the input string."""
  input_ids = tokenizer.encode(input_string)
  bos_tok = [tokenizer.bos_id()] if tokenizer.bos_id() else []
  input_ids = jnp.array(
    tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=jnp.int32
  )
  return input_ids


def _preprocess_dict(
    training_input: dict[str, Any],
) -> DataInput | TrainingInput:
  """Wraps input dict with either DataInput or TrainingInput."""

  training_input_fields = [
      field.name for field in dataclasses.fields(DataInput)
  ]
  tokenized_input_fields = [
      field.name for field in dataclasses.fields(TrainingInput)
  ]

  # If the dict contains tokenized fields, we should wrap it with
  # TrainingInput.
  if all(field in training_input for field in tokenized_input_fields):
    return TrainingInput(
        **{field: training_input[field] for field in tokenized_input_fields}
    )
  elif all(field in training_input for field in training_input_fields):
    return DataInput(
        **{field: training_input[field] for field in training_input_fields}
    )
  else:
    raise ValueError(
        "Training input must contain either tokenized fields "
        f"({training_input_fields}) or raw string fields "
        f"({training_input_fields}). Received: {training_input.keys()}."
    )


def process_dpo_record(
    record: dict[str, str | list[str]],
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
) -> TrainingInput:
  """Processes and tokenizes a single record for DPO training.

  This function takes a dictionary containing a prompt, a chosen response,
  and a rejected response. It tokenizes each text field and creates the
  corresponding attention masks.

  Note: We use a dictionary here, to make it easier to use on any Grain dataset
  with `.map`.

  Args:
      record: A dictionary, containing "prompts", "chosen_responses", and
        "rejected_responses" as keys. The values can be a single string or a
        list of strings.
      tokenizer: The tokenizer to use for converting text into token IDs.
      max_prompt_length: The maximum length for the tokenized prompts. Any
        sequence longer than this will be truncated.
      max_response_length: The maximum length for the tokenized responses. Any
        sequence longer than this will be truncated.

  Returns:
      A `TrainingInput` object.
  """

  prompts = record["prompts"]
  chosen_responses = record["chosen_responses"]
  rejected_responses = record["rejected_responses"]

  unbatched = isinstance(prompts, str)

  if unbatched:
    prompts = [prompts]
  if isinstance(chosen_responses, str):
    chosen_responses = [chosen_responses]
  if isinstance(rejected_responses, str):
    rejected_responses = [rejected_responses]

  # Only prompt is left padded, others are right padded.
  prompt_ids, prompt_mask = _generate_ids_and_masks(
      prompts,
      tokenizer,
      max_prompt_length,
      left_pad=True,
  )
  chosen_ids, chosen_mask = _generate_ids_and_masks(
      chosen_responses, tokenizer, max_response_length, left_pad=False
  )
  rejected_ids, rejected_mask = _generate_ids_and_masks(
      rejected_responses, tokenizer, max_response_length, left_pad=False
  )

  if unbatched:
    prompt_ids = jnp.squeeze(prompt_ids, axis=0)
    chosen_ids = jnp.squeeze(chosen_ids, axis=0)
    rejected_ids = jnp.squeeze(rejected_ids, axis=0)
    prompt_mask = jnp.squeeze(prompt_mask, axis=0)
    chosen_mask = jnp.squeeze(chosen_mask, axis=0)
    rejected_mask = jnp.squeeze(rejected_mask, axis=0)

  return TrainingInput(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      chosen_ids=chosen_ids,
      chosen_mask=chosen_mask,
      rejected_ids=rejected_ids,
      rejected_mask=rejected_mask,
  )


DpoTrainingConfig = DPOTrainingConfig
DpoTrainer = DPOTrainer

# ORPO aliases
ORPOTrainingConfig = DPOTrainingConfig
ORPOTrainer = DPOTrainer
OrpoTrainingConfig = DPOTrainingConfig
OrpoTrainer = DPOTrainer
