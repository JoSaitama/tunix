#!/usr/bin/env python3

"""Shared helpers for offline evaluation of exported Qwen2.5 DPO runs."""

from __future__ import annotations

import ast
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from absl import logging
import jax
import numpy as np
import optax
from omegaconf import OmegaConf

from tunix.cli import dpo_main as dpo_main_lib
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_lib
from tunix.sft import metrics_logger as metrics_logger_lib
from tunix.sft.dpo import dpo_trainer as dpo_trainer_lib


@dataclass(slots=True)
class DPOEvalBundle:
  """Loaded actor/reference models plus DPO eval config."""

  actor_model: Any
  reference_model: Any
  tokenizer: Any
  actor_mesh: jax.sharding.Mesh
  reference_mesh: jax.sharding.Mesh
  training_config: dpo_trainer_lib.DPOTrainingConfig


def make_mesh(mesh_config: dict[str, Any]) -> jax.sharding.Mesh:
  axis_shapes = ast.literal_eval(mesh_config["shape"])
  axis_names = ast.literal_eval(mesh_config["axis_names"])
  return jax.make_mesh(
      tuple(axis_shapes),
      tuple(axis_names),
      axis_types=(jax.sharding.AxisType.Auto,) * len(tuple(axis_names)),
  )


def load_eval_bundle(
    *,
    base_cfg: Any,
    actor_model_path: str,
    reference_model_path: str,
    metrics_prefix: str,
) -> DPOEvalBundle:
  actor_config = OmegaConf.to_container(base_cfg.actor_model_config, resolve=True)
  reference_config = OmegaConf.to_container(
      base_cfg.reference_model_config, resolve=True
  )
  tokenizer_config = OmegaConf.to_container(base_cfg.tokenizer_config, resolve=True)

  actor_config["model_path"] = actor_model_path
  actor_config.pop("lora_config", None)
  reference_config["model_path"] = reference_model_path
  reference_config.pop("lora_config", None)

  actor_mesh = make_mesh(actor_config["mesh"])
  reference_mesh = make_mesh(reference_config["mesh"])

  actor_model, tokenizer_path, _ = model_lib.create_model(
      actor_config,
      tokenizer_config,
      actor_mesh,
      return_model_path=True,
  )
  reference_model, _, _ = model_lib.create_model(
      reference_config,
      tokenizer_config,
      reference_mesh,
      return_model_path=True,
  )
  tokenizer = model_lib.create_tokenizer(tokenizer_config, tokenizer_path)

  training_config = dpo_trainer_lib.DPOTrainingConfig(
      eval_every_n_steps=1,
      max_steps=None,
      gradient_accumulation_steps=1,
      checkpoint_root_directory=None,
      checkpointing_options=None,
      metrics_logging_options=None,
      profiler_options=None,
      data_sharding_axis=tuple(base_cfg.training_config.data_sharding_axis),
      max_inflight_computations=1,
      metrics_prefix=metrics_prefix,
      pbar_description=None,
      beta=base_cfg.dpo_config.beta,
      label_smoothing=base_cfg.dpo_config.label_smoothing,
      max_prompt_length=base_cfg.dpo_config.max_prompt_length,
      max_response_length=base_cfg.dpo_config.max_response_length,
      use_dynamic_batch_curation=False,
      curation_variant="outlier_l2",
      curation_threshold=3.0,
      curation_keep_ratio=1.0,
      self_influence_dot_threshold=0.0,
      late_flip_ratio=0.0,
      late_flip_start_step=None,
      late_flip_seed=123,
  )
  return DPOEvalBundle(
      actor_model=actor_model,
      reference_model=reference_model,
      tokenizer=tokenizer,
      actor_mesh=actor_mesh,
      reference_mesh=reference_mesh,
      training_config=training_config,
  )


def load_eval_dataset(
    *,
    module_spec: str,
    tokenizer: Any,
    batch_size: int,
    num_batches: int | None,
    dpo_config: Any,
):
  dataset = data_lib.get_dataset_from_module(module_spec, tokenizer)
  dataset = dpo_main_lib.filter_dpo_dataset(
      dataset,
      tokenizer=tokenizer,
      max_prompt_length=dpo_config.max_prompt_length,
      max_response_length=dpo_config.max_response_length,
      dataset_name="eval",
  )
  dataset = dataset.to_iter_dataset()
  return data_lib.post_init_dataset(
      dataset,
      tokenizer,
      batch_size=batch_size,
      num_batches=num_batches,
      max_prompt_length=None,
  )


def create_eval_trainer(
    bundle: DPOEvalBundle,
) -> tuple[dpo_trainer_lib.DPOTrainer, metrics_logger_lib.MetricsLogger]:
  metrics_logger = metrics_logger_lib.MetricsLogger()
  trainer = dpo_trainer_lib.DPOTrainer(
      model=bundle.actor_model,
      ref_model=bundle.reference_model,
      optimizer=optax.adamw(learning_rate=0.0),
      training_config=bundle.training_config,
      tokenizer=bundle.tokenizer,
  )
  trainer.metrics_logger = metrics_logger
  trainer.is_managed_externally = True
  return trainer, metrics_logger


def _resolve_metrics_prefix(trainer: dpo_trainer_lib.DPOTrainer) -> str:
  """Returns the eval metrics prefix across old/new trainer attribute layouts."""
  if hasattr(trainer, "metrics_prefix"):
    return trainer.metrics_prefix
  if hasattr(trainer, "config") and hasattr(trainer.config, "metrics_prefix"):
    return trainer.config.metrics_prefix
  if hasattr(trainer, "training_config") and hasattr(
      trainer.training_config, "metrics_prefix"
  ):
    return trainer.training_config.metrics_prefix
  raise AttributeError(
      "Unable to resolve metrics_prefix from the provided DPO trainer."
  )


def aggregate_eval_metrics(
    *,
    trainer: dpo_trainer_lib.DPOTrainer,
    metrics_logger: metrics_logger_lib.MetricsLogger,
    eval_dataset: Iterable[Any],
) -> dict[str, float]:
  metrics_prefix = _resolve_metrics_prefix(trainer)
  trainer.train([], eval_dataset)
  return {
      "loss": float(metrics_logger.get_metric(metrics_prefix, "loss", "eval")),
      "perplexity": float(
          metrics_logger.get_metric(metrics_prefix, "perplexity", "eval")
      ),
      "rewards_accuracy": float(
          metrics_logger.get_metric(metrics_prefix, "rewards/accuracy", "eval")
      ),
      "rewards_margin": float(
          metrics_logger.get_metric(metrics_prefix, "rewards/margin", "eval")
      ),
      "rewards_chosen": float(
          metrics_logger.get_metric(metrics_prefix, "rewards/chosen", "eval")
      ),
      "rewards_rejected": float(
          metrics_logger.get_metric(metrics_prefix, "rewards/rejected", "eval")
      ),
      "log_probs_chosen": float(
          metrics_logger.get_metric(metrics_prefix, "log_probs/chosen", "eval")
      ),
      "log_probs_rejected": float(
          metrics_logger.get_metric(metrics_prefix, "log_probs/rejected", "eval")
      ),
  }


def iter_preference_batch_metrics(
    *,
    trainer: dpo_trainer_lib.DPOTrainer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
) -> list[dict[str, float]]:
  if not prompts:
    return []
  training_input = {
      "prompts": list(prompts),
      "chosen_responses": list(chosen_responses),
      "rejected_responses": list(rejected_responses),
  }
  train_example = trainer._prepare_inputs(training_input)  # pylint: disable=protected-access
  losses, aux = dpo_trainer_lib.dpo_loss_fn_per_sample(
      trainer.model,
      train_example,
      algorithm=trainer.algorithm,
      beta=trainer.dpo_config.beta,
      lambda_orpo=trainer.dpo_config.lambda_orpo,
      label_smoothing=trainer.dpo_config.label_smoothing,
  )
  losses = np.asarray(jax.device_get(losses))
  aux = {key: np.asarray(jax.device_get(value)) for key, value in aux.items()}
  output = []
  for index in range(len(prompts)):
    output.append(
        {
            "loss": float(losses[index]),
            "rewards_accuracy": float(aux["rewards/accuracy"][index]),
            "rewards_margin": float(aux["rewards/margin"][index]),
            "rewards_chosen": float(aux["rewards/chosen"][index]),
            "rewards_rejected": float(aux["rewards/rejected"][index]),
            "log_probs_chosen": float(aux["log_probs/chosen"][index]),
            "log_probs_rejected": float(aux["log_probs/rejected"][index]),
        }
    )
  return output


def iter_response_reward_scores(
    *,
    trainer: dpo_trainer_lib.DPOTrainer,
    prompts: Sequence[str],
    responses: Sequence[str],
    anchor_responses: Sequence[str] | None = None,
) -> list[float]:
  """Score single responses with the DPO implicit reward.

  The chosen-side reward depends only on the prompt/response pair, so we can
  reuse the pairwise evaluator by pairing each response with a placeholder
  anchor response.
  """
  anchors = list(anchor_responses) if anchor_responses is not None else list(responses)
  batch_metrics = iter_preference_batch_metrics(
      trainer=trainer,
      prompts=prompts,
      chosen_responses=responses,
      rejected_responses=anchors,
  )
  return [float(row["rewards_chosen"]) for row in batch_metrics]


def close_eval_trainer(
    trainer: dpo_trainer_lib.DPOTrainer,
    metrics_logger: metrics_logger_lib.MetricsLogger,
) -> None:
  trainer.checkpoint_manager.close()
  metrics_logger.close()
  del trainer
  gc.collect()


def close_eval_bundle(bundle: DPOEvalBundle) -> None:
  try:
    del bundle.actor_model
    del bundle.reference_model
    gc.collect()
    jax.monitoring.clear_event_listeners()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to fully clear DPO eval bundle caches: %s", exc)


def iter_chunks(rows: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
  for start in range(0, len(rows), batch_size):
    yield rows[start : start + batch_size]
