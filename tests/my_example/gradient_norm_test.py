"""SERVER CPU regressions; real NNX/JAX/Optax with tiny arrays only."""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class GradientNormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        import optax
        from my_example.rebuttal import gradient_norm_trainer as norm
        cls.jax, cls.jp, cls.nnx, cls.optax, cls.norm = jax, jnp, nnx, optax, norm

    def trainer(self, *, lora=False, aux=True, artifact=None):
        trainer = object.__new__(self.norm.GradientNormTop50Trainer)
        trainer.num_generations, trainer.artifact_dir = 4, artifact
        trainer._norm_seen = trainer._norm_kept = trainer._norm_logged_updates = 0
        trainer._train_steps, trainer.rl_metrics_to_log = 0, {}
        trainer._buffered_train_metrics = SimpleNamespace(additional_metrics={})
        trainer._lora_enabled, trainer._has_aux = lora, aux
        trainer.gen_model_input_fn = lambda x: x
        trainer._score_loss_has_aux = aux

        def total(model, policy, kl):
            value = model.weight[...] * (policy + kl)
            return (value, {"kl": model.weight[...] * kl}) if aux else value

        def policy_only(model, policy, kl):
            value = model.weight[...] * policy
            return (value, {"kl": self.jp.asarray(0.)}) if aux else value
        trainer.loss_fn, trainer._score_loss_fn = total, policy_only
        return trainer

    def model(self, lora=False):
        nnx, jp = self.nnx, self.jp
        class Toy(nnx.Module):
            def __init__(self):
                self.weight = (nnx.LoRAParam if lora else nnx.Param)(jp.asarray(2.))
                self.backbone = nnx.Param(jp.asarray(10.))
        return Toy()

    def test_norm_includes_all_leaves_and_promotes_reduction_only(self):
        tree = {"a": self.jp.array([[3., 4.], [0., 1.]], dtype=self.jp.bfloat16),
                "b": self.jp.array([12., 2.])}
        values = self.norm.squared_gradient_norms(tree)
        np.testing.assert_allclose(values, [169., 5.])
        self.assertEqual(values.dtype, self.jp.float32)
        self.assertEqual(tree["a"].dtype, self.jp.bfloat16)

    def test_exact_two_per_prompt_no_cross_prompt_ranking(self):
        values = self.jp.array([100., 90., 80., 70., 4., 3., 2., 1.])
        mask = self.norm.group_top50_mask(values)
        np.testing.assert_array_equal(mask, [True, True, False, False] * 2)

    def test_ties_and_all_zero_groups_are_deterministic(self):
        for compiled in (False, True):
            fn = self.jax.jit(self.norm.group_top50_mask) if compiled else self.norm.group_top50_mask
            result = fn(self.jp.array([1., 2., 2., 2., 0., 0., 0., 0.]))
            np.testing.assert_array_equal(result, [False, True, True, False, True, True, False, False])

    def test_strict_above_median_differs_on_ties(self):
        values = self.jp.ones(4)
        self.assertEqual(int(self.norm.group_top50_mask(values).sum()), 2)
        self.assertEqual(int((values > self.jp.median(values)).sum()), 0)

    def test_invalid_shapes_and_generation_counts_rejected(self):
        for values, g in ((self.jp.ones(3), 4), (self.jp.ones((1, 4)), 4),
                          (self.jp.ones(4), 2), (self.jp.ones(0), 4)):
            with self.assertRaises(ValueError):
                self.norm.group_top50_mask(values, g)
        for tree in ({}, {"a": self.jp.ones(4), "b": self.jp.ones(8)}):
            with self.assertRaises(ValueError):
                self.norm.squared_gradient_norms(tree)

    def test_nonfinite_aborts_eager_and_jit(self):
        for compiled in (False, True):
            fn = self.jax.jit(self.norm.group_top50_mask) if compiled else self.norm.group_top50_mask
            for value in (float("nan"), float("inf")):
                with self.subTest(compiled=compiled, value=value), \
                     mock.patch.object(logging.getLogger('jax._src.debugging'), 'disabled', True), \
                     self.assertRaises(Exception):
                    fn(self.jp.array([value, 1., 2., 3.])).block_until_ready()
                    self.jax.effects_barrier()

    def test_nonfinite_kl_gradient_does_not_silently_continue(self):
        trainer, model = self.trainer(), self.model()
        optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.Param)
        with mock.patch.object(logging.getLogger('jax._src.debugging'), 'disabled', True), \
             self.assertRaises(Exception):
            result = self.nnx.jit(trainer._train_step)(model, optimizer, dict(
                policy=self.jp.array([4., -3., 2., -1.]), kl=self.jp.array([float('inf'), 0., 0., 0.])))
            self.jax.block_until_ready(result)
            self.jax.effects_barrier()

    def test_sign_reversal_cannot_change_magnitude_ranking(self):
        gradients = {'w': self.jp.array([4., -3., 2., -1.])}
        original = self.norm.group_top50_mask(self.norm.squared_gradient_norms(gradients))
        reverse = self.norm.group_top50_mask(self.norm.squared_gradient_norms(
            self.jax.tree.map(lambda x: -x, gradients)))
        np.testing.assert_array_equal(original, reverse)

    def test_original_mismatch20_rewards_feed_norm_inputs_without_oracle(self):
        from my_example.reward_rank_noise import RewardRankNoiseConfig, apply_reward_rank_noise
        from tunix.rl.grpo.grpo_learner import compute_advantages
        prompts = [f'norm_fixture_prompt_{i}' for i in range(128) for _ in range(4)]
        rewards = np.tile([0., .5, 3., 4.], 128)
        noisy = apply_reward_rank_noise(rewards, prompts, 4, RewardRankNoiseConfig(.2, 5))
        self.assertTrue(noisy.selected_groups.any())
        self.assertTrue((~noisy.selected_groups).any())
        observed_a = compute_advantages(noisy.corrupted_rewards, 4).astype(np.float32)
        clean_a = compute_advantages(rewards, 4).astype(np.float32)
        selected = np.repeat(noisy.selected_groups, 4)
        np.testing.assert_array_equal(observed_a[~selected], clean_a[~selected])
        self.assertTrue(np.any(observed_a[selected] != clean_a[selected]))
        features = np.tile([1., 2., 3., 4.], 128)
        g = self.jp.asarray(-observed_a * features)
        mask = self.norm.group_top50_mask(self.norm.squared_gradient_norms({'w': g}))
        expected = np.argsort(-(observed_a * features).reshape(-1, 4) ** 2,
                              axis=1, kind='stable')[:, :2]
        expected_mask = np.zeros((128, 4), bool)
        np.put_along_axis(expected_mask, expected, True, axis=1)
        np.testing.assert_array_equal(np.asarray(mask).reshape(-1, 4), expected_mask)

    def test_policy_ranking_masks_full_gradient_including_kl(self):
        # Policy magnitudes select 0/1. Full-gradient magnitudes would select
        # 2/3 instead, so this separates the two algorithms unambiguously.
        for compiled in (False, True):
            trainer, model = self.trainer(), self.model()
            optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.Param)
            step = self.nnx.jit(trainer._train_step) if compiled else trainer._train_step
            loss, aux = step(model, optimizer, dict(policy=self.jp.array([4., -3., 2., -1.]),
                                                    kl=self.jp.array([1., 1., 100., -100.])))
            # mean selected total gradients (5,-2) =1.5, not original-N .75.
            np.testing.assert_allclose(model.weight[...], 1.85, atol=1e-6)
            np.testing.assert_allclose(loss, 3.)
            np.testing.assert_allclose(aux["kl"], 2.)
            self.assertEqual(float(aux["skipped_samples"]), 2.)
            self.assertEqual(float(aux["norm_top50_kept_fraction"]), .5)

    def test_lora_only_space_matches_dtv_not_backbone(self):
        trainer, model = self.trainer(lora=True), self.model(lora=True)
        trainer.loss_fn = lambda model, policy, kl: (
            model.weight[...] * (policy + kl) + model.backbone[...] * policy,
            {"kl": model.weight[...] * kl})
        trainer._score_loss_fn = lambda model, policy, kl: (
            model.weight[...] * policy + model.backbone[...] * (100. - policy), {"kl": self.jp.asarray(0.)})
        optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.LoRAParam)
        self.nnx.jit(trainer._train_step)(model, optimizer,
            dict(policy=self.jp.array([4., -3., 2., -1.]), kl=self.jp.zeros(4)))
        np.testing.assert_allclose(model.weight[...], 1.95)
        np.testing.assert_allclose(model.backbone[...], 10.)

    def test_original_advantages_and_full_batch_shape_preserved(self):
        trainer, model = self.trainer(), self.model()
        n = 16
        seen = []
        original = self.jp.tile(self.jp.array([-1.2, -.3, .4, 1.1]), 4)
        def score(model, advantages):
            seen.append(tuple(advantages.shape))  # scalar under per-unit vmap
            return model.weight[...] * advantages, {"kl": self.jp.asarray(0.)}
        trainer.loss_fn = trainer._score_loss_fn = score
        optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.Param)
        _, aux = trainer._train_step(model, optimizer, {"advantages": original})
        # Original [-1.2,+1.1] selected, no new two-answer [-.707,+.707] A.
        np.testing.assert_allclose(model.weight[...], 2.005, atol=1e-6)
        np.testing.assert_array_equal(original, self.jp.tile(self.jp.array([-1.2, -.3, .4, 1.1]), 4))
        self.assertEqual(float(aux["skipped_samples"]), n / 2)
        self.assertEqual(seen, [(), ()])

    def test_no_aux_path_and_missing_policy_hook(self):
        trainer, model = self.trainer(aux=False), self.model()
        optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.Param)
        _, aux = self.nnx.jit(trainer._train_step)(model, optimizer,
            dict(policy=self.jp.array([4., -3., 2., -1.]), kl=self.jp.zeros(4)))
        self.assertIsNone(aux)
        trainer._score_loss_fn = None
        with self.assertRaisesRegex(RuntimeError, "policy-score loss hook"):
            trainer._train_step(model, optimizer, {})

    def test_actual_grpo_loss_and_configured_kl_match_analytic_update(self):
        from dataclasses import replace
        from tunix.rl.grpo.grpo_learner import GRPOConfig, TrainExample, grpo_loss_fn
        trainer, model = self.trainer(), self.model()
        tokens = self.jp.repeat(self.jp.arange(1, 17)[:, None], 3, axis=1)
        features = np.asarray(tokens[:, 0], dtype=np.float32) / 100.
        advantages = np.tile([-1.2, -.3, .4, 1.1], 4).astype(np.float32)
        old = 2. * features[:, None] * np.ones((16, 3), dtype=np.float32)
        example = TrainExample(prompt_ids=self.jp.ones((16, 2), dtype=self.jp.int32),
            prompt_mask=self.jp.ones((16, 2)), completion_ids=tokens,
            completion_mask=self.jp.ones((16, 3)), advantages=self.jp.asarray(advantages),
            old_per_token_logps=self.jp.asarray(old), ref_per_token_logps=self.jp.zeros((16, 3)),
            filter_random_values=None)
        cfg = GRPOConfig(num_generations=4, beta=.08)
        trainer.loss_fn = lambda model, train_example: grpo_loss_fn(model, train_example, cfg, 0, 1)
        trainer._score_loss_fn = lambda model, train_example: grpo_loss_fn(
            model, train_example, replace(cfg, beta=0.), 0, 1)

        def logps(model, prompt_tokens, completion_tokens, **unused):
            return model.weight[...] * completion_tokens.astype(self.jp.float32) / 100.
        policy = -advantages * features
        total = policy + .08 * (1. - np.exp(-old[:, 0])) * features
        local_order = np.argsort(-policy.reshape(-1, 4) ** 2, axis=1, kind='stable')[:, :2]
        selected = (local_order + np.arange(4)[:, None] * 4).reshape(-1)
        expected = 2. - .1 * total[selected].mean()
        optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.Param)
        with mock.patch('tunix.rl.common.compute_per_token_logps', side_effect=logps):
            self.nnx.jit(trainer._train_step)(model, optimizer, {'train_example': example})
        np.testing.assert_allclose(model.weight[...], expected, atol=1e-6)
        np.testing.assert_array_equal(example.advantages, advantages)

    def test_records_and_session_counts_match_actual_mask(self):
        from tunix.rl.grpo.grpo_learner import TrainExample
        with tempfile.TemporaryDirectory() as directory:
            trainer, model = self.trainer(artifact=Path(directory)), self.model()
            example = TrainExample(prompt_ids=self.jp.tile(self.jp.array([[1, 2]]), (16, 1)),
                prompt_mask=self.jp.ones((16, 2)), completion_ids=self.jp.ones((16, 3)),
                completion_mask=self.jp.ones((16, 3)), advantages=self.jp.tile(self.jp.array([-1., -.3, .4, .9]), 4),
                old_per_token_logps=self.jp.zeros((16, 3)), ref_per_token_logps=self.jp.zeros((16, 3)),
                filter_random_values=None)
            trainer.loss_fn = trainer._score_loss_fn = lambda model, train_example: (
                model.weight[...] * train_example.advantages, {"kl": self.jp.asarray(0.)})
            optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.Param)
            for index in range(2):
                _, aux = self.nnx.jit(trainer._train_step)(model, optimizer, {"train_example": example})
                trainer._train_steps = index
                trainer._post_process_train_step(aux)
            rows = [json.loads(x) for x in (Path(directory) / 'norm_selection.jsonl').read_text().splitlines()]
            summary = json.loads((Path(directory) / 'norm_summary.json').read_text())
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["observed_completions"], 32)
            self.assertEqual(summary["retained_completions"], 16)
            for row in rows:
                self.assertFalse(row["advantages_recomputed"])
                for group in row["groups"]:
                    self.assertEqual(group["selected_local_indices"], [0, 3])
                    self.assertEqual(len(group["advantages"]), 4)

    def test_real_four_cpu_sharded_jit_keeps_half_every_group(self):
        self.assertEqual(self.jax.local_device_count(), 4,
                         "Run via test_gradient_norm.sh cpu (four virtual CPU devices)")
        mesh = self.jax.sharding.Mesh(np.asarray(self.jax.local_devices()), ("data",))
        sharding = self.jax.sharding.NamedSharding(mesh, self.jax.sharding.PartitionSpec("data"))
        trainer, model = self.trainer(), self.model()
        optimizer = self.nnx.Optimizer(model, self.optax.sgd(.1), wrt=self.nnx.Param)
        values = self.jax.device_put(self.jp.tile(self.jp.array([4., -3., 2., -1.]), 4), sharding)
        with mesh:
            step = self.nnx.jit(trainer._train_step)
            for _ in range(2):
                _, aux = step(model, optimizer, dict(policy=values, kl=self.jp.zeros(16)))
                self.assertEqual(float(aux["skipped_samples"]), 8.)
        np.testing.assert_allclose(model.weight[...], 1.9, atol=1e-6)

    def test_environment_rejects_combined_methods(self):
        for flags in ({"TUNIX_PODS_KEEP_COUNT": "2"}, {"TUNIX_DYNAMIC_SAMPLING": "1"},
                      {"TUNIX_GRPO_NUM_GENERATIONS": "8"}):
            with self.assertRaises(ValueError):
                self.norm.configuration_from_environment(flags)
        self.assertEqual(self.norm.configuration_from_environment({})["num_generations"], 4)

    def test_hbm_fixture_validation_rejects_wrong_selection_and_zero_stress(self):
        from my_example.rebuttal.test_gradient_norm_hbm import validate_records
        rows = [{"groups": [{"policy_gradient_squared_l2_norms": [4., 9., 1., 0.],
                             "policy_gradient_l2_norms": [2., 3., 1., 0.],
                             "selected_local_indices": [0, 1]} for _ in range(4)],
                 "advantages_recomputed": False, "score_objective": "policy", "mask_application": "full"}
                for _ in range(3)]
        validate_records(rows)
        rows[1]["groups"][0]["selected_local_indices"] = [2, 3]
        with self.assertRaises(RuntimeError):
            validate_records(rows)
        for group in rows[1]['groups']:
            group.update(policy_gradient_squared_l2_norms=[0.] * 4,
                         policy_gradient_l2_norms=[0.] * 4, selected_local_indices=[0, 1])
        with self.assertRaisesRegex(RuntimeError, 'nonzero policy gradients'):
            validate_records(rows)

    def test_cluster_opt_in_preserves_baseline_and_existing_dtv_routes(self):
        source = (ROOT / 'tunix/rl/rl_cluster.py').read_text()
        start = source.index('    actor_trainer_kwargs = {}')
        end = source.index('    if actor_config.use_dynamic_batch_curation and actor_trainer_cls is robust_trainer', start)
        block = ast.parse(textwrap.dedent(source[start:end]))
        for node in ast.walk(block):
            if isinstance(node, ast.If):
                node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
        # The real routing block logs actor_trainer_cls.__name__, so stand-ins
        # must be CLASSES, not strings. No actual trainer construction occurs.
        vanilla = type('Trainer', (), {})
        robust = type('RobustTrainer', (), {})
        dtv = type('SelfInfTrainer', (), {})
        dtv_policy = type('PolicySelfInfTrainer', (), {})
        loo = type('SelfInfLooTrainer', (), {})
        names = dict(gradient_norm_trainer=self.norm, os=os, logging=SimpleNamespace(info=lambda *a: None),
            robust_trainer=SimpleNamespace(RobustTrainer=robust), rl_trainer=SimpleNamespace(Trainer=vanilla),
            self_inf_trainer=SimpleNamespace(SelfInfTrainer=dtv),
            self_inf_policy_trainer=SimpleNamespace(PolicySelfInfTrainer=dtv_policy),
            self_inf_loo_trainer=SimpleNamespace(SelfInfLooTrainer=loo))
        for enabled, variant, flags, expected in (
                (False, 'gradient_norm_top50', {}, vanilla),
                (True, 'gradient_norm_top50', {}, self.norm.GradientNormTop50Trainer),
                (False, '', {}, vanilla), (True, '', {}, robust),
                (True, 'self_inf', {}, dtv), (True, 'self_inf', {'TUNIX_DBC_SELF_INF_POLICY': '1'}, dtv_policy),
                (True, 'self_inf', {'TUNIX_DBC_SELF_INF_LOO': '1'}, loo)):
            namespace = dict(names, actor_config=SimpleNamespace(use_dynamic_batch_curation=enabled))
            with mock.patch.dict(os.environ, dict(TUNIX_DBC_VARIANT=variant, **flags), clear=True):
                exec(compile(block, 'norm_route_test', 'exec'), namespace)
            self.assertEqual(namespace['actor_trainer_cls'], expected)


class GradientNormLauncherTest(unittest.TestCase):
    def run_shell(self, filename, *args, env=None):
        values = dict(os.environ, TUNIX_PODS_KEEP_COUNT='', TUNIX_DYNAMIC_SAMPLING='0',
                      TUNIX_NORM_RUN_ROOT='', TUNIX_MY_RESULT_DIR='')
        values.update(env or {})
        return subprocess.run(['bash', str(ROOT / 'my_example/rebuttal' / filename), *args],
                              env=values, text=True, capture_output=True)

    def test_baseline_dry_run_keeps_frozen_budget_and_single_worker(self):
        result = self.run_shell('run_gradient_norm_baseline.sh', '--dry-run', env={
            'TUNIX_EXPERIMENT_SEED': '5', 'TUNIX_REWARD_RANK_NOISE_FRACTION': '0.2',
            'TUNIX_REWARD_RANK_NOISE_SEED': '5'})
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in ('--num-generations 4', '--learning-rate 1e-6', '--max-train-examples 3072',
                     '--max-eval-examples 1319', 'TPU_PROCESS_BOUNDS=1', 'CLOUD_TPU_TASK_ID=0',
                     'TUNIX_DBC_VARIANT=gradient_norm_top50', 'gradient_norm_top50_seed5_mismatch0.2_'):
            self.assertIn(text, result.stdout)

    def test_rejects_generation_changes_and_method_mixing(self):
        for args in (('--num-generations', '8'), ('--use-dbc-self-inf-group',)):
            result = self.run_shell('run_gradient_norm_baseline.sh', '--dry-run', *args)
            self.assertNotEqual(result.returncode, 0)

    def test_suite_dry_run_has_requested_five_seeds_only(self):
        result = self.run_shell('run_gradient_norm_suite.sh', '--seeds', '5', '0', '13', '21', '42',
                                '--mismatch', '0.2', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 5)
        for line, seed in zip(lines, (5, 0, 13, 21, 42)):
            self.assertIn(f'TUNIX_EXPERIMENT_SEED={seed}', line)
            self.assertIn('TUNIX_REWARD_RANK_NOISE_FRACTION=0.2', line)


if __name__ == '__main__':
    unittest.main()
