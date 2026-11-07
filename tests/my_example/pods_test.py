"""Host-side PODS algorithm and preparation tests; no TPU runtime required."""
from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from my_example.pods import (
    pods_learner_class,
    select_max_variance_indices,
    selected_micro_batch_sizes,
    subset_advantages,
)
from tests.my_example.alignment_mismatch_flow_test import reward_rank_noise


@dataclasses.dataclass(frozen=True)
class Example:
    prompt_ids: object
    prompt_mask: object
    completion_ids: object
    completion_mask: object
    ref_per_token_logps: object
    old_per_token_logps: object
    advantages: object
    filter_random_values: object

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def tree_map(fn, value):
    return Example(**{
        field.name: None if getattr(value, field.name) is None else fn(getattr(value, field.name))
        for field in dataclasses.fields(value)
    })


class FakeBase:
    def __init__(self, rl_cluster, algo_config, **kwargs):
        self.rl_cluster, self.algo_config = rl_cluster, algo_config
        self.noise_config = kwargs.get("noise_config")

    def _compute_rewards(self, prompts, completions, mode, **kwargs):
        rewards = np.asarray(completions, dtype=float)
        if self.noise_config and mode == "train":
            rewards = reward_rank_noise.apply_reward_rank_noise(
                rewards, prompts, self.algo_config.num_generations, self.noise_config
            ).corrupted_rewards
        return rewards

    def _generate_and_compute_advantage(self, training_input, mode):
        rewards = self._compute_rewards(
            training_input["prompts"], training_input["rewards"], mode
        )
        ids = np.asarray(training_input["ids"])[:, None]
        return Example(ids, ids+100, ids+200, ids+300, ids+400, None,
                       subset_advantages(rewards.reshape(-1, 4)).reshape(-1), ids+500)

    def _process_accumulated_batches(self, batches, sizes, mode):
        return ["unchanged-eval"]


class PODSAlgorithmTest(unittest.TestCase):
    def test_hbm_batch_has_array_leaves_for_learner_repeat(self):
        from my_example.test_pods_hbm import make_hbm_batch

        batch = make_hbm_batch()
        for value in batch.values():
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.shape, (4,))
        repeated = {key: np.repeat(value, 4, axis=0) for key, value in batch.items()}
        self.assertEqual(repeated["prompts"].shape, (16,))
        for i in range(4):
            self.assertEqual(len(set(repeated["prompts"][i*4:(i+1)*4])), 1)

    def test_exhaustive_optimum_and_unique_indices(self):
        rng = np.random.default_rng(7)
        for n in range(2, 9):
            rewards = rng.integers(-12, 20, (15, n)) / 4
            for m in range(2, n+1):
                indices = select_max_variance_indices(rewards, m)
                actual = np.take_along_axis(rewards, indices, axis=1).var(axis=1)
                for row, ids, var in zip(rewards, indices, actual):
                    exact = max(np.var(row[list(subset)]) for subset in itertools.combinations(range(n), m))
                    self.assertAlmostEqual(var, exact)
                    self.assertEqual(len(set(ids)), m)

    def test_group_isolation_and_extremes(self):
        values = np.array([[0, .5, 3, 4], [100, 102, 101, 103]])
        chosen = np.take_along_axis(values, select_max_variance_indices(values, 2), axis=1)
        np.testing.assert_array_equal(chosen, [[4, 0], [103, 100]])

    def test_identity_order_and_zero_variance(self):
        values = np.array([[3, 0, 2, 1], [2, 2, 2, 2]])
        np.testing.assert_array_equal(select_max_variance_indices(values, 4), [[0,1,2,3]]*2)
        np.testing.assert_array_equal(select_max_variance_indices(values[1:], 2), [[0,1]])
        np.testing.assert_array_equal(subset_advantages([[2,2]]), [[0,0]])

    def test_subset_advantages_use_subset_not_full_group(self):
        values = np.array([[0., .5, 3., 4.]])
        indices = select_max_variance_indices(values, 2)
        selected = np.take_along_axis(values, indices, axis=1)
        actual = subset_advantages(selected)
        np.testing.assert_allclose(actual, [[2/(np.sqrt(8)+1e-4), -2/(np.sqrt(8)+1e-4)]])
        self.assertFalse(np.allclose(actual, np.take_along_axis(subset_advantages(values), indices, axis=1)))

    def test_invalid_inputs(self):
        for m in (0, 1, 5):
            with self.assertRaises(ValueError): select_max_variance_indices([[0,1,2,3]], m)
        for values in ([1,2], [[1,np.nan]], [[1,np.inf]]):
            with self.assertRaises(ValueError): select_max_variance_indices(values, 2)
        with self.assertRaises(ValueError): selected_micro_batch_sizes([15], 4, 2)
        self.assertEqual(selected_micro_batch_sizes([16, 8], 4, 2), [8,4])


class PODSPipelineTest(unittest.TestCase):
    def setUp(self):
        jax = ModuleType("jax")
        jax.tree = SimpleNamespace(map=tree_map)
        jax.device_put = np.asarray
        tunix, rl = ModuleType("tunix"), ModuleType("tunix.rl")
        cluster, utils = ModuleType("tunix.rl.rl_cluster"), ModuleType("tunix.rl.utils")
        cluster.Mode = SimpleNamespace(TRAIN="train", EVAL="eval")
        utils.merge_micro_batches = lambda batches: {
            key: np.concatenate([b[key] for b in batches]) for key in batches[0]
        }
        utils.get_batch_slice = lambda batch, sl: tree_map(lambda x: x[sl], batch)
        rl.rl_cluster, rl.utils, tunix.rl = cluster, utils, rl
        self.patch = mock.patch.dict(sys.modules, {
            "jax": jax, "tunix": tunix, "tunix.rl": rl,
            "tunix.rl.rl_cluster": cluster, "tunix.rl.utils": utils,
            "my_example.reward_rank_noise": reward_rank_noise,
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def learner(self, m=2, noise=None, dbc=False):
        cluster = SimpleNamespace(
            cluster_config=SimpleNamespace(training_config=SimpleNamespace(
                use_dynamic_batch_curation=dbc, max_steps=691)),
            global_steps=0, buffer_metrics=mock.Mock(),
        )
        return pods_learner_class(FakeBase)(
            rl_cluster=cluster, algo_config=SimpleNamespace(num_generations=4, beta=.08),
            pods_keep_count=m, pods_artifact_dir=self.tmp.name, noise_config=noise,
        )

    @staticmethod
    def batch(start=0, groups=4):
        return {"prompts": np.repeat([f"prompt-{start+i}" for i in range(groups)], 4),
                "rewards": np.tile([0., .5, 3., 4.], groups),
                "ids": np.arange(start*4, (start+groups)*4)}

    def test_real_mixin_gathers_all_fields_and_keeps_optional_none(self):
        learner = self.learner()
        selected = learner._generate_and_compute_advantage(self.batch(), "train")
        ids = np.array([3,0,7,4,11,8,15,12])[:,None]
        for name, offset in (("prompt_ids",0), ("prompt_mask",100), ("completion_ids",200),
                             ("completion_mask",300), ("ref_per_token_logps",400), ("filter_random_values",500)):
            np.testing.assert_array_equal(getattr(selected, name), ids+offset)
        self.assertIsNone(selected.old_per_token_logps)
        summary = json.loads((Path(self.tmp.name)/"pods_summary.json").read_text())
        self.assertEqual((summary["generated_completions"], summary["selected_completions"]), (16,8))
        metadata = json.loads((Path(self.tmp.name)/"run_metadata.json").read_text())
        self.assertEqual(metadata["beta"], .08)

    def test_mismatch_is_applied_on_n_before_selection(self):
        config = reward_rank_noise.RewardRankNoiseConfig(fraction=.2, seed=5)
        batch = self.batch(groups=100)
        expected = reward_rank_noise.apply_reward_rank_noise(
            batch["rewards"], batch["prompts"], 4, config
        )
        self.assertTrue(expected.selected_groups.any())
        self.assertTrue((~expected.selected_groups).any())
        learner = self.learner(noise=config)
        result = learner._generate_and_compute_advantage(batch, "train")
        records = [json.loads(x) for x in (Path(self.tmp.name)/"pods_selection.jsonl").read_text().splitlines()]
        np.testing.assert_array_equal([r["observed_rewards"] for r in records], expected.corrupted_rewards.reshape(-1,4))
        np.testing.assert_array_equal([r["mismatch_selected"] for r in records], expected.selected_groups)
        for i, record in enumerate(records):
            for j, local in enumerate(record["selected_indices"]):
                self.assertEqual(result.prompt_ids[i*2+j,0], i*4+local)

    def test_two_micro_batches_split_into_eight_each(self):
        learner = self.learner()
        chunks = learner._process_accumulated_batches([self.batch(), self.batch(4)], [16,16], "train")
        self.assertEqual([len(x.advantages) for x in chunks], [8,8])
        self.assertLess(chunks[0].prompt_ids.max(), 16)
        self.assertGreaterEqual(chunks[1].prompt_ids.min(), 16)

    def test_identity_and_eval_bypass(self):
        learner = self.learner(m=4)
        batch = self.batch()
        expected = FakeBase._generate_and_compute_advantage(learner, batch, "train")
        result = learner._generate_and_compute_advantage(batch, "train")
        for field in dataclasses.fields(result):
            np.testing.assert_equal(getattr(result, field.name), getattr(expected, field.name))
        eval_result = learner._generate_and_compute_advantage(batch, "eval")
        self.assertEqual(len(eval_result.advantages), 16)
        self.assertEqual(learner._process_accumulated_batches([], [], "eval"), ["unchanged-eval"])

    def test_reject_conflicting_filter_and_mixed_prompt_groups(self):
        with self.assertRaises(ValueError): self.learner(dbc=True)
        batch = self.batch()
        batch["prompts"][1] = "different"
        with self.assertRaises(ValueError): self.learner()._generate_and_compute_advantage(batch, "train")


if __name__ == "__main__":
    unittest.main()
