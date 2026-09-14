"""On-policy prompt-gradient estimation in the actor's LoRA parameter space."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, TrainExample

from my_example.reward_rank_noise import (
    RewardRankNoiseConfig,
    apply_reward_rank_noise,
)
from my_example.rewards import MATCH_NUMBERS

from .artifacts import prompt_id
from .data_utils import Example, batch_examples
from .equivalence import (
    compare_learnalign_feature_paths,
    feature_execution_mode,
)
from .scoring import learnability


@dataclasses.dataclass(frozen=True)
class PromptGradientEstimate:
    """Projected prompt gradients and auditable selector reward data."""

    features: np.ndarray
    clean_binary_outcomes: np.ndarray
    selector_rewards: np.ndarray
    selector_advantages: np.ndarray
    mismatch_selected: np.ndarray
    mismatch_effective: np.ndarray


def exact_correctness(completions: Sequence[str], answers: Sequence[str]) -> np.ndarray:
    """Binary GSM8K correctness matching the repository's pass@1 evaluator."""
    outcomes = []
    for completion, answer in zip(completions, answers):
        match = MATCH_NUMBERS.search(completion)
        try:
            prediction = float(match.group(1).strip()) if match else None
            target = float(str(answer).strip())
            outcomes.append(float(prediction == target))
        except (TypeError, ValueError):
            outcomes.append(0.0)
    return np.asarray(outcomes, dtype=np.float32)


def _mix_u32(values: jax.Array, seed: int) -> jax.Array:
    """Deterministic integer mixing used by the sparse feature hash."""
    x = values.astype(jnp.uint32) + jnp.uint32(seed & 0xFFFFFFFF)
    x = (x ^ (x >> jnp.uint32(16))) * jnp.uint32(0x7FEB352D)
    x = (x ^ (x >> jnp.uint32(15))) * jnp.uint32(0x846CA68B)
    return x ^ (x >> jnp.uint32(16))


def project_gradient_tree(
    prompt_grads: Any,
    *,
    projection_dim: int,
    seed: int,
) -> jax.Array:
    """Applies a deterministic sparse JL/feature-hashing projection.

    The projection is accumulated leaf-by-leaf, so no dense P x D matrix and no
    flattened full-gradient copy is materialized.  This is an implementation
    adaptation of the papers' random projection for a single-node TPU setup.
    """
    leaves = jax.tree_util.tree_leaves(prompt_grads)
    if not leaves:
        raise ValueError("no differentiable gradient leaves were found")
    batch_size = leaves[0].shape[0]
    projected = jnp.zeros((batch_size, projection_dim), dtype=jnp.float32)
    coordinate_offset = 0
    for leaf_index, leaf in enumerate(leaves):
        flat = jnp.asarray(leaf, dtype=jnp.float32).reshape((batch_size, -1))
        size = flat.shape[1]
        coordinates = jnp.arange(size, dtype=jnp.uint32) + jnp.uint32(
            coordinate_offset
        )
        mixed = _mix_u32(coordinates, seed + 0x9E3779B9 * (leaf_index + 1))
        buckets = (mixed % jnp.uint32(projection_dim)).astype(jnp.int32)
        signs = jnp.where((mixed & jnp.uint32(1)) == 0, 1.0, -1.0)
        leaf_projection = jax.vmap(
            lambda row: jax.ops.segment_sum(
                row * signs, buckets, num_segments=projection_dim
            )
        )(flat)
        projected = projected + leaf_projection
        coordinate_offset += size
    return projected


class PromptGradientEstimator:
    """Generates binary rollouts and returns one feature per prompt."""

    def __init__(
        self,
        *,
        rl_cluster,
        training_algo_config: GRPOConfig,
        projection_dim: int,
        projection_seed: int,
        selection_micro_batch_size: int,
        noise_config: RewardRankNoiseConfig,
        grouped_feature_estimation: bool = False,
        equivalence_report_path: str | None = None,
        equivalence_selection_ratio: int = 4,
    ):
        self.rl_cluster = rl_cluster
        self.training_algo_config = training_algo_config
        self.projection_dim = projection_dim
        self.projection_seed = projection_seed
        self.selection_micro_batch_size = selection_micro_batch_size
        self.noise_config = noise_config
        self.grouped_feature_estimation = grouped_feature_estimation
        self.equivalence_report_path = equivalence_report_path
        self.equivalence_selection_ratio = equivalence_selection_ratio
        self._compiled: dict[int, Any] = {}
        self._grouped_compiled: dict[int, Any] = {}

    def _write_grouped_equivalence_report(
        self,
        *,
        examples: Sequence[Example],
        legacy_features: np.ndarray,
        grouped_features: np.ndarray,
        selector_rewards: np.ndarray,
        num_rollouts: int,
    ) -> None:
        """Writes one selection-oriented report over all requested prompts."""
        if not self.equivalence_report_path:
            return
        report_path = Path(self.equivalence_report_path)
        values = learnability(selector_rewards)
        report = compare_learnalign_feature_paths(
            legacy_features,
            grouped_features,
            selection_ratio=self.equivalence_selection_ratio,
            score_weights=values,
            acceptance_mode="selected-set",
        )
        source_indices = [
            int(example.get("index", index))
            for index, example in enumerate(examples)
        ]
        prompt_ids = [prompt_id(str(example["prompts"])) for example in examples]
        legacy_selected = report["legacy_selected_indices"]
        grouped_selected = report["grouped_selected_indices"]
        report.update(
            {
                "method": "learnalign",
                "legacy_feature_chunk": (
                    f"{self.selection_micro_batch_size}x{num_rollouts}"
                ),
                "grouped_feature_chunk": (
                    f"{self.selection_micro_batch_size}x{num_rollouts}"
                ),
                "grouped_gradient_count_per_chunk": (
                    self.selection_micro_batch_size
                ),
                "legacy_selected_source_indices": [
                    source_indices[index] for index in legacy_selected
                ],
                "grouped_selected_source_indices": [
                    source_indices[index] for index in grouped_selected
                ],
                "legacy_selected_prompt_ids": [
                    prompt_ids[index] for index in legacy_selected
                ],
                "grouped_selected_prompt_ids": [
                    prompt_ids[index] for index in grouped_selected
                ],
                "nonzero_learnability_prompts": int(np.count_nonzero(values)),
                "nonzero_legacy_feature_rows": int(
                    np.count_nonzero(np.linalg.norm(legacy_features, axis=1))
                ),
                "nonzero_grouped_feature_rows": int(
                    np.count_nonzero(np.linalg.norm(grouped_features, axis=1))
                ),
                "shared_tokens": True,
                "shared_model_parameters": True,
                "advantages": "shared_actual_selector_advantages",
                "score_weights": "actual_learnability_p(1-p)",
            }
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if not report["passed"]:
            raise RuntimeError(
                "LearnAlign grouped selected-set A/B check failed; see "
                f"{report_path}"
            )

    def _generate_binary_train_example(
        self,
        examples: Sequence[Example],
        *,
        num_rollouts: int,
        apply_mismatch: bool,
    ) -> tuple[
        TrainExample,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        batch = batch_examples(examples)
        repeated = {
            key: np.repeat(values, num_rollouts, axis=0)
            for key, values in batch.items()
        }
        prompts = [str(value) for value in repeated["prompts"]]
        answers = [str(value) for value in repeated["answer"]]
        rollout_output = self.rl_cluster.generate(
            prompts=prompts,
            mode=rl_cluster_lib.Mode.TRAIN,
            micro_batch_size=self.selection_micro_batch_size * num_rollouts,
        )

        pad_id = self.rl_cluster.rollout.pad_id()
        eos_id = self.rl_cluster.rollout.eos_id()
        completion_ids_np = np.asarray(rollout_output.tokens)
        prompt_ids = jnp.asarray(rollout_output.left_padded_prompt_tokens)
        completion_padding_mask = completion_ids_np != pad_id
        completion_mask = common.np_make_completion_mask(
            completion_ids_np, eos_tok=eos_id
        ) * completion_padding_mask
        clean_outcomes = exact_correctness(rollout_output.text, answers)
        if apply_mismatch and self.noise_config.enabled:
            noise_audit = apply_reward_rank_noise(
                rewards=clean_outcomes,
                prompts=prompts,
                num_generations=num_rollouts,
                config=self.noise_config,
            )
            selector_rewards = np.asarray(
                noise_audit.corrupted_rewards,
                dtype=np.float32,
            )
            mismatch_selected = noise_audit.selected_groups
            mismatch_effective = noise_audit.effective_groups
        else:
            selector_rewards = clean_outcomes.copy()
            mismatch_selected = np.zeros(len(examples), dtype=np.bool_)
            mismatch_effective = np.zeros(len(examples), dtype=np.bool_)
        advantage_fn = function_registry.get_advantage_estimator("grpo")
        advantages = advantage_fn(
            rewards=selector_rewards,
            num_generations=num_rollouts,
        )
        grouped_advantages = np.asarray(advantages, dtype=np.float32).reshape(
            (len(examples), num_rollouts)
        )
        return (
            TrainExample(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_ids != pad_id,
                completion_ids=jnp.asarray(completion_ids_np),
                completion_mask=jnp.asarray(completion_mask),
                ref_per_token_logps=None,
                advantages=jnp.asarray(advantages),
                old_per_token_logps=None,
                filter_random_values=None,
            ),
            clean_outcomes.reshape((len(examples), num_rollouts)),
            selector_rewards.reshape((len(examples), num_rollouts)),
            grouped_advantages,
            mismatch_selected,
            mismatch_effective,
        )

    def _feature_function(self, num_rollouts: int):
        if num_rollouts in self._compiled:
            return self._compiled[num_rollouts]

        score_config = dataclasses.replace(
            self.training_algo_config,
            beta=0.0,
            num_generations=num_rollouts,
        )
        policy_loss_fn = function_registry.get_policy_loss_fn(
            score_config.policy_loss_fn
        )
        pad_id = self.rl_cluster.rollout.pad_id()
        eos_id = self.rl_cluster.rollout.eos_id()
        projection_dim = self.projection_dim
        projection_seed = self.projection_seed

        def feature_fn(model, train_example):
            def one_loss(model, one_example):
                loss, _ = policy_loss_fn(
                    model,
                    one_example,
                    algo_config=score_config,
                    pad_id=pad_id,
                    eos_id=eos_id,
                )
                return loss

            wrt = (
                nnx.LoRAParam
                if self.rl_cluster.actor_trainer._lora_enabled  # pylint: disable=protected-access
                else nnx.Param
            )
            grad_fn = nnx.value_and_grad(
                one_loss,
                argnums=nnx.DiffState(0, wrt),
            )
            in_axes = jax.tree_util.tree_map(
                lambda value: None if value is None else 0,
                train_example,
                is_leaf=lambda value: value is None,
            )
            _, completion_grads = jax.vmap(
                grad_fn,
                in_axes=(None, in_axes),
            )(model, train_example)

            def group_mean(leaf):
                return leaf.reshape(
                    (-1, num_rollouts) + leaf.shape[1:]
                ).mean(axis=1)

            prompt_grads = jax.tree_util.tree_map(group_mean, completion_grads)
            return project_gradient_tree(
                prompt_grads,
                projection_dim=projection_dim,
                seed=projection_seed,
            )

        compiled = nnx.jit(feature_fn)
        self._compiled[num_rollouts] = compiled
        return compiled

    def _grouped_feature_function(self, num_rollouts: int):
        """Projects one gradient of the mean loss for each prompt group.

        This preserves the original full selector input shape while moving the
        rollout mean inside automatic differentiation.  It therefore produces
        only one gradient tree per prompt instead of one per completion.
        """
        if num_rollouts in self._grouped_compiled:
            return self._grouped_compiled[num_rollouts]

        score_config = dataclasses.replace(
            self.training_algo_config,
            beta=0.0,
            num_generations=num_rollouts,
        )
        policy_loss_fn = function_registry.get_policy_loss_fn(
            score_config.policy_loss_fn
        )
        pad_id = self.rl_cluster.rollout.pad_id()
        eos_id = self.rl_cluster.rollout.eos_id()
        projection_dim = self.projection_dim
        projection_seed = self.projection_seed

        def grouped_feature_fn(model, train_example):
            completion_count = train_example.advantages.shape[0]
            if completion_count % num_rollouts:
                raise ValueError(
                    "completion count must be divisible by num_rollouts; "
                    f"got {completion_count} and {num_rollouts}"
                )

            def group_completion_axis(value):
                if value is None:
                    return None
                return value.reshape(
                    (-1, num_rollouts) + value.shape[1:]
                )

            grouped_example = jax.tree_util.tree_map(
                group_completion_axis,
                train_example,
                is_leaf=lambda value: value is None,
            )

            def one_completion_loss(model, one_example):
                loss, _ = policy_loss_fn(
                    model,
                    one_example,
                    algo_config=score_config,
                    pad_id=pad_id,
                    eos_id=eos_id,
                )
                return loss

            def group_mean_loss(model, one_prompt_group):
                completion_in_axes = jax.tree_util.tree_map(
                    lambda value: None if value is None else 0,
                    one_prompt_group,
                    is_leaf=lambda value: value is None,
                )
                completion_losses = jax.vmap(
                    one_completion_loss,
                    in_axes=(None, completion_in_axes),
                )(model, one_prompt_group)
                return jnp.mean(completion_losses)

            wrt = (
                nnx.LoRAParam
                if self.rl_cluster.actor_trainer._lora_enabled  # pylint: disable=protected-access
                else nnx.Param
            )
            prompt_grad_fn = nnx.value_and_grad(
                group_mean_loss,
                argnums=nnx.DiffState(0, wrt),
            )
            prompt_in_axes = jax.tree_util.tree_map(
                lambda value: None if value is None else 0,
                grouped_example,
                is_leaf=lambda value: value is None,
            )
            _, prompt_grads = jax.vmap(
                prompt_grad_fn,
                in_axes=(None, prompt_in_axes),
            )(model, grouped_example)
            return project_gradient_tree(
                prompt_grads,
                projection_dim=projection_dim,
                seed=projection_seed,
            )

        compiled = nnx.jit(grouped_feature_fn)
        self._grouped_compiled[num_rollouts] = compiled
        return compiled

    def estimate(
        self,
        examples: Sequence[Example],
        *,
        num_rollouts: int,
        apply_mismatch: bool,
    ) -> PromptGradientEstimate:
        """Estimates prompt features using clean or rank-reversed binary rewards."""
        if not examples:
            raise ValueError("cannot estimate an empty example collection")
        all_features: list[np.ndarray] = []
        all_clean_outcomes: list[np.ndarray] = []
        all_selector_rewards: list[np.ndarray] = []
        all_advantages: list[np.ndarray] = []
        all_mismatch_selected: list[np.ndarray] = []
        all_mismatch_effective: list[np.ndarray] = []
        equivalence_legacy_features: list[np.ndarray] = []
        equivalence_grouped_features: list[np.ndarray] = []
        equivalence_enabled = bool(self.equivalence_report_path) and not Path(
            self.equivalence_report_path
        ).exists()
        execution_mode = feature_execution_mode(
            equivalence_enabled=equivalence_enabled,
            grouped_feature_estimation=self.grouped_feature_estimation,
        )
        legacy_feature_fn = self._feature_function(num_rollouts)
        grouped_feature_fn = None
        if self.grouped_feature_estimation or self.equivalence_report_path:
            grouped_feature_fn = self._grouped_feature_function(num_rollouts)
        for start in range(0, len(examples), self.selection_micro_batch_size):
            chunk = examples[start : start + self.selection_micro_batch_size]
            (
                train_example,
                clean_outcomes,
                selector_rewards,
                advantages,
                mismatch_selected,
                mismatch_effective,
            ) = self._generate_binary_train_example(
                chunk,
                num_rollouts=num_rollouts,
                apply_mismatch=apply_mismatch,
            )
            actor_mesh = self.rl_cluster.cluster_config.role_to_mesh[
                rl_cluster_lib.Role.ACTOR
            ]
            with actor_mesh, self.rl_cluster._get_logical_axis_rules_cm(  # pylint: disable=protected-access
                rl_cluster_lib.Role.ACTOR
            ):
                if execution_mode == "dual":
                    if grouped_feature_fn is None:
                        raise RuntimeError(
                            "grouped LearnAlign feature path is unavailable"
                        )
                    legacy_features = legacy_feature_fn(
                        self.rl_cluster.actor_trainer.model,
                        train_example,
                    )
                    grouped_features = grouped_feature_fn(
                        self.rl_cluster.actor_trainer.model,
                        train_example,
                    )
                    legacy_host_features = np.asarray(
                        jax.device_get(legacy_features)
                    )
                    host_features = np.asarray(jax.device_get(grouped_features))
                    equivalence_legacy_features.append(legacy_host_features)
                    equivalence_grouped_features.append(host_features)
                elif execution_mode == "grouped":
                    if grouped_feature_fn is None:
                        raise RuntimeError(
                            "grouped LearnAlign feature path is unavailable"
                        )
                    features = grouped_feature_fn(
                        self.rl_cluster.actor_trainer.model,
                        train_example,
                    )
                    host_features = np.asarray(jax.device_get(features))
                elif execution_mode == "legacy":
                    features = legacy_feature_fn(
                        self.rl_cluster.actor_trainer.model,
                        train_example,
                    )
                    host_features = np.asarray(jax.device_get(features))
                else:
                    raise AssertionError(
                        f"unexpected feature execution mode: {execution_mode}"
                    )
            all_features.append(host_features)
            all_clean_outcomes.append(clean_outcomes)
            all_selector_rewards.append(selector_rewards)
            all_advantages.append(advantages)
            all_mismatch_selected.append(mismatch_selected)
            all_mismatch_effective.append(mismatch_effective)
        estimate = PromptGradientEstimate(
            features=np.concatenate(all_features),
            clean_binary_outcomes=np.concatenate(all_clean_outcomes),
            selector_rewards=np.concatenate(all_selector_rewards),
            selector_advantages=np.concatenate(all_advantages),
            mismatch_selected=np.concatenate(all_mismatch_selected),
            mismatch_effective=np.concatenate(all_mismatch_effective),
        )
        if equivalence_enabled:
            self._write_grouped_equivalence_report(
                examples=examples,
                legacy_features=np.concatenate(equivalence_legacy_features),
                grouped_features=np.concatenate(equivalence_grouped_features),
                selector_rewards=estimate.selector_rewards,
                num_rollouts=num_rollouts,
            )
        return estimate
