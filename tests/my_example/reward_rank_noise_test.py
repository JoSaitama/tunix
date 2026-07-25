"""Tests for deterministic prompt-group reward rank reversal."""

from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_example import reward_rank_noise
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOLearner


class RewardRankNoiseTest(unittest.TestCase):

    def test_unset_and_zero_fraction_preserve_clean_configuration(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(reward_rank_noise.config_from_env().enabled)

        with mock.patch.dict(
            os.environ,
            {
                reward_rank_noise.NOISE_FRACTION_ENV: "0",
                reward_rank_noise.NOISE_SEED_ENV: "7",
            },
            clear=True,
        ):
            config = reward_rank_noise.config_from_env()
            self.assertFalse(config.enabled)
            self.assertEqual(config.seed, 7)

    def test_invalid_configuration_fails_clearly(self):
        with mock.patch.dict(
            os.environ,
            {reward_rank_noise.NOISE_FRACTION_ENV: "1.1"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be in"):
                reward_rank_noise.config_from_env()

        with mock.patch.dict(
            os.environ,
            {
                reward_rank_noise.NOISE_FRACTION_ENV: "0.2",
                reward_rank_noise.NOISE_SEED_ENV: "-1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                reward_rank_noise.config_from_env()

    def test_prompt_selection_is_deterministic_and_nested(self):
        config_20 = reward_rank_noise.RewardRankNoiseConfig(0.2, 0)
        config_40 = reward_rank_noise.RewardRankNoiseConfig(0.4, 0)
        prompts = [f"question-{index}" for index in range(1000)]

        selected_20 = {
            prompt
            for prompt in prompts
            if reward_rank_noise.selected_prompt(prompt, config_20)
        }
        selected_40 = {
            prompt
            for prompt in prompts
            if reward_rank_noise.selected_prompt(prompt, config_40)
        }

        self.assertLessEqual(selected_20, selected_40)
        self.assertGreater(len(selected_20), 150)
        self.assertLess(len(selected_20), 250)
        self.assertGreater(len(selected_40), 350)
        self.assertLess(len(selected_40), 450)
        self.assertEqual(
            reward_rank_noise.stable_prompt_score("same", 3),
            reward_rank_noise.stable_prompt_score("same", 3),
        )

    def test_reverse_reward_ranks_preserves_multiset_mean_and_std(self):
        clean = np.asarray([4.0, 0.0, 2.0, -1.0])

        corrupted = reward_rank_noise.reverse_reward_ranks(clean)

        np.testing.assert_allclose(corrupted, [-1.0, 2.0, 0.0, 4.0])
        np.testing.assert_allclose(np.sort(corrupted), np.sort(clean))
        self.assertAlmostEqual(float(np.mean(corrupted)), float(np.mean(clean)))
        self.assertAlmostEqual(float(np.std(corrupted)), float(np.std(clean)))

    def test_apply_noise_selects_groups_and_reports_ties(self):
        config = reward_rank_noise.RewardRankNoiseConfig(1.0, 0)
        prompts = ["prompt-a"] * 4 + ["prompt-b"] * 4
        clean = np.asarray([4.0, 0.0, 2.0, -1.0, 1.0, 1.0, 1.0, 1.0])

        audit = reward_rank_noise.apply_reward_rank_noise(
            clean, prompts, num_generations=4, config=config
        )

        np.testing.assert_array_equal(audit.selected_groups, [True, True])
        np.testing.assert_array_equal(audit.effective_groups, [True, False])
        np.testing.assert_allclose(
            audit.corrupted_rewards,
            [-1.0, 2.0, 0.0, 4.0, 1.0, 1.0, 1.0, 1.0],
        )

    def test_apply_noise_rejects_mixed_prompt_group(self):
        with self.assertRaisesRegex(ValueError, "repeated copies"):
            reward_rank_noise.apply_reward_rank_noise(
                rewards=np.asarray([1.0, 2.0, 3.0, 4.0]),
                prompts=["a", "a", "b", "a"],
                num_generations=4,
                config=reward_rank_noise.RewardRankNoiseConfig(1.0, 0),
            )

    def test_learner_corrupts_train_but_not_eval_rewards(self):
        class _Cluster:

            def __init__(self):
                self.buffered = []

            def buffer_metrics(self, metrics, mode):
                self.buffered.append((metrics, mode))

            def buffer_metrics_async(self, metrics, mode, step):
                self.buffered.append((metrics, mode, step))

        learner = object.__new__(
            reward_rank_noise.RewardRankNoiseGRPOLearner
        )
        learner.noise_config = reward_rank_noise.RewardRankNoiseConfig(1.0, 0)
        learner.algo_config = SimpleNamespace(num_generations=4)
        learner.rl_cluster = _Cluster()
        clean = np.asarray([4.0, 0.0, 2.0, -1.0])
        prompts = ["prompt-a"] * 4
        completions = ["a", "b", "c", "d"]

        with mock.patch.object(
            GRPOLearner, "_compute_rewards", return_value=clean
        ):
            train_rewards = learner._compute_rewards(
                prompts,
                completions,
                mode=rl_cluster_lib.Mode.TRAIN,
            )
            eval_rewards = learner._compute_rewards(
                prompts,
                completions,
                mode=rl_cluster_lib.Mode.EVAL,
            )

        np.testing.assert_allclose(train_rewards, [-1.0, 2.0, 0.0, 4.0])
        np.testing.assert_allclose(eval_rewards, clean)
        self.assertTrue(learner.rl_cluster.buffered)


if __name__ == "__main__":
    unittest.main()
