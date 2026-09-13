"""On-policy prompt-gradient estimation in the actor's LoRA parameter space."""

from __future__ import annotations

import dataclasses
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

from .data_utils import Example, batch_examples


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
    ):
        self.rl_cluster = rl_cluster
        self.training_algo_config = training_algo_config
        self.projection_dim = projection_dim
        self.projection_seed = projection_seed
        self.selection_micro_batch_size = selection_micro_batch_size
        self.noise_config = noise_config
        self._compiled: dict[int, Any] = {}

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
        feature_fn = self._feature_function(num_rollouts)
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
                features = feature_fn(
                    self.rl_cluster.actor_trainer.model,
                    train_example,
                )
            all_features.append(np.asarray(jax.device_get(features)))
            all_clean_outcomes.append(clean_outcomes)
            all_selector_rewards.append(selector_rewards)
            all_advantages.append(advantages)
            all_mismatch_selected.append(mismatch_selected)
            all_mismatch_effective.append(mismatch_effective)
        return PromptGradientEstimate(
            features=np.concatenate(all_features),
            clean_binary_outcomes=np.concatenate(all_clean_outcomes),
            selector_rewards=np.concatenate(all_selector_rewards),
            selector_advantages=np.concatenate(all_advantages),
            mismatch_selected=np.concatenate(all_mismatch_selected),
            mismatch_effective=np.concatenate(all_mismatch_effective),
        )
