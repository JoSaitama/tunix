"""Fast host-side preflight for the 20% alignment-baseline mismatch path.

This test deliberately keeps some noisy prompts after selection.  It verifies
that mismatch is an input corruption rate, not a promise that a selector will
identify or remove every corrupted prompt.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from my_example.alignment_baselines.artifacts import (
    SelectionArtifactWriter,
    prompt_id,
)
from my_example.alignment_baselines.curriculum import (
    GradAlignCurriculum,
    LearnAlignCurriculum,
)


def _load_reward_rank_noise():
    """Loads the production host-side noise functions in lightweight envs."""
    if importlib.util.find_spec("flax") is not None:
        from my_example import reward_rank_noise

        return reward_rank_noise

    # reward_rank_noise.py keeps host utilities beside the GRPOLearner subclass.
    # These minimal import stubs let the exact production utilities run on a
    # laptop without installing JAX/Flax; TPU environments use the normal import.
    tunix_module = ModuleType("tunix")
    rl_module = ModuleType("tunix.rl")
    cluster_module = ModuleType("tunix.rl.rl_cluster")
    grpo_module = ModuleType("tunix.rl.grpo")
    learner_module = ModuleType("tunix.rl.grpo.grpo_learner")

    class _Mode:
        TRAIN = "train"
        EVAL = "eval"

    class _GRPOLearner:
        pass

    cluster_module.Mode = _Mode
    learner_module.GRPOLearner = _GRPOLearner
    rl_module.rl_cluster = cluster_module
    rl_module.grpo = grpo_module
    grpo_module.grpo_learner = learner_module
    tunix_module.rl = rl_module

    module_name = "_alignment_test_reward_rank_noise"
    source = Path(__file__).resolve().parents[2] / "my_example/reward_rank_noise.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward-rank noise module from {source}")
    module = importlib.util.module_from_spec(spec)
    stubs = {
        module_name: module,
        "tunix": tunix_module,
        "tunix.rl": rl_module,
        "tunix.rl.rl_cluster": cluster_module,
        "tunix.rl.grpo": grpo_module,
        "tunix.rl.grpo.grpo_learner": learner_module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


reward_rank_noise = _load_reward_rank_noise()
NOISE_CONFIG = reward_rank_noise.RewardRankNoiseConfig(fraction=0.2, seed=0)


class _Cluster:
    def __init__(self):
        self.logged = []

    def log_scalar_immediately(self, name, value, step):
        self.logged.append((name, value, step))


class _ProductionNoiseEstimator:
    """Small deterministic estimator using the production mismatch function."""

    def estimate(self, examples, *, num_rollouts, apply_mismatch):
        if num_rollouts != 4:
            raise AssertionError("this preflight expects four selector rollouts")
        clean = np.tile(
            np.asarray([1.0, 1.0, 0.0, 0.0]),
            (len(examples), 1),
        )
        prompts = [
            str(example["prompts"])
            for example in examples
            for _ in range(num_rollouts)
        ]
        if apply_mismatch:
            audit = reward_rank_noise.apply_reward_rank_noise(
                clean.reshape(-1),
                prompts,
                num_generations=num_rollouts,
                config=NOISE_CONFIG,
            )
            rewards = audit.corrupted_rewards.reshape(clean.shape)
            mismatch_selected = audit.selected_groups
            mismatch_effective = audit.effective_groups
        else:
            rewards = clean.copy()
            mismatch_selected = np.zeros(len(examples), dtype=np.bool_)
            mismatch_effective = np.zeros(len(examples), dtype=np.bool_)

        means = rewards.mean(axis=1, keepdims=True)
        stds = rewards.std(axis=1, ddof=1, keepdims=True)
        advantages = (rewards - means) / (stds + 1e-4)
        # Equal nonzero features make selection deterministic by source order.
        # This intentionally retains both clean and noisy candidates so the
        # downstream-training path can be asserted independently of score quality.
        features = np.tile(np.asarray([1.0, 0.0]), (len(examples), 1))
        return SimpleNamespace(
            features=features,
            clean_binary_outcomes=clean,
            selector_rewards=rewards,
            selector_advantages=advantages,
            mismatch_selected=mismatch_selected,
            mismatch_effective=mismatch_effective,
        )


def _example(prompt: str, index: int):
    return {"prompts": prompt, "answer": "0", "index": index}


def _exact_twenty_percent_pool():
    noisy = []
    clean = []
    cursor = 0
    while len(noisy) < 8 or len(clean) < 32:
        prompt = f"mismatch-preflight-prompt-{cursor}"
        target = (
            noisy
            if reward_rank_noise.selected_prompt(prompt, NOISE_CONFIG)
            else clean
        )
        target.append(prompt)
        cursor += 1

    # The selected prefix contains all 8 noisy prompts and 12 clean prompts.
    # Both methods use a stable top-k tie break, so this makes residual noisy
    # prompts reach training and exercises that part of the data path.
    ordered = []
    for index in range(8):
        ordered.extend((noisy[index], clean[index]))
    ordered.extend(clean[8:12])
    ordered.extend(clean[12:32])
    return [_example(prompt, index) for index, prompt in enumerate(ordered)]


def _read_records(path: str):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _training_prompts(batches):
    return [str(prompt) for batch in batches for prompt in batch["prompts"]]


def _assert_dense_update_noise(testcase, prompts):
    repeated_prompts = [prompt for prompt in prompts for _ in range(4)]
    clean_dense_rewards = np.tile(
        np.asarray([4.0, 0.0, 2.0, -1.0]), len(prompts)
    )
    audit = reward_rank_noise.apply_reward_rank_noise(
        clean_dense_rewards,
        repeated_prompts,
        num_generations=4,
        config=NOISE_CONFIG,
    )
    testcase.assertEqual(int(audit.selected_groups.sum()), 8)
    testcase.assertEqual(int(audit.effective_groups.sum()), 8)
    testcase.assertTrue(
        np.any(~np.isclose(audit.clean_rewards, audit.corrupted_rewards))
    )


class AlignmentMismatchFlowTest(unittest.TestCase):

    def test_learnalign_mixes_then_selects_noisy_and_clean_prompts(self):
        examples = _exact_twenty_percent_pool()
        with tempfile.TemporaryDirectory() as tmp:
            curriculum = LearnAlignCurriculum(
                training_examples=examples,
                train_batch_size=4,
                total_update_steps=6,
                estimator=_ProductionNoiseEstimator(),
                writer=SelectionArtifactWriter(tmp, "learnalign"),
                rl_cluster=_Cluster(),
                warmup_prompts=4,
                estimation_rollouts=4,
                selection_ratio=2,
                warmup_counts_toward_budget=True,
            )
            batches = list(curriculum)
            records = _read_records(f"{tmp}/learnalign_selection.jsonl")

        self.assertEqual(sum(r["mismatch_selected"] for r in records), 8)
        self.assertEqual(sum(not r["mismatch_selected"] for r in records), 32)
        noisy = next(r for r in records if r["mismatch_selected"])
        clean = next(r for r in records if not r["mismatch_selected"])
        self.assertEqual(noisy["clean_binary_outcomes"], [1, 1, 0, 0])
        self.assertEqual(noisy["selector_rewards"], [0, 0, 1, 1])
        self.assertEqual(clean["selector_rewards"], [1, 1, 0, 0])

        selected_ids = {r["prompt_id"] for r in records if r["selected"]}
        selected_training = _training_prompts(batches[1:])
        self.assertEqual(len(selected_training), 20)
        self.assertEqual(
            {prompt_id(prompt) for prompt in selected_training}, selected_ids
        )
        self.assertEqual(
            sum(
                reward_rank_noise.selected_prompt(prompt, NOISE_CONFIG)
                for prompt in selected_training
            ),
            8,
        )
        _assert_dense_update_noise(self, selected_training)

    def test_gradalign_keeps_validation_clean_and_mixes_candidates(self):
        examples = _exact_twenty_percent_pool()
        validation = [
            _example(f"clean-heldout-validation-{index}", index)
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            curriculum = GradAlignCurriculum(
                training_examples=examples,
                validation_examples=validation,
                train_batch_size=4,
                total_update_steps=5,
                estimator=_ProductionNoiseEstimator(),
                writer=SelectionArtifactWriter(tmp, "gradalign"),
                rl_cluster=_Cluster(),
                candidate_rollouts=4,
                validation_rollouts=4,
                selection_ratio=2,
                selection_interval=5,
            )
            batches = list(curriculum)
            records = _read_records(f"{tmp}/gradalign_selection.jsonl")

        validation_records = [
            record for record in records if record["record_type"] == "validation"
        ]
        candidate_records = [
            record for record in records if record["record_type"] == "candidate"
        ]
        self.assertTrue(validation_records)
        self.assertTrue(
            all(not record["mismatch_selected"] for record in validation_records)
        )
        self.assertTrue(
            all(
                record["selector_rewards"] == record["clean_binary_outcomes"]
                for record in validation_records
            )
        )
        self.assertEqual(
            sum(record["mismatch_selected"] for record in candidate_records), 8
        )
        self.assertEqual(
            sum(not record["mismatch_selected"] for record in candidate_records), 32
        )

        selected_training = _training_prompts(batches)
        self.assertEqual(len(selected_training), 20)
        selected_ids = {
            record["prompt_id"]
            for record in candidate_records
            if record["selected"]
        }
        self.assertEqual(
            {prompt_id(prompt) for prompt in selected_training}, selected_ids
        )
        self.assertEqual(
            sum(
                reward_rank_noise.selected_prompt(prompt, NOISE_CONFIG)
                for prompt in selected_training
            ),
            8,
        )
        _assert_dense_update_noise(self, selected_training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
