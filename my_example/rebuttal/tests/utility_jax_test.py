"""Run on the SERVER with JAX_PLATFORMS=cpu; tiny arrays, no checkpoint/model."""

from dataclasses import replace
import unittest
from unittest import mock

import numpy as np


class ProductionGradientContractTest(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from tunix.rl.grpo.grpo_learner import GRPOConfig, TrainExample, grpo_loss_fn
        self.jax, self.jp, self.nnx = jax, jnp, nnx
        self.loss, self.config = grpo_loss_fn, GRPOConfig(num_generations=4, beta=.08)

        class Toy(nnx.Module):
            def __init__(self):
                self.weight = nnx.LoRAParam(jnp.array([.02, -.03], dtype=jnp.float32))
        self.model = Toy()
        self.features = np.arange(48).reshape(16, 3).astype(np.float32) / 100

        def logps(model, prompt_tokens, completion_tokens, **unused):
            return model.weight.value[0] * jnp.asarray(completion_tokens) + model.weight.value[1]
        self.logps = logps
        ids = jnp.asarray(self.features)
        old = logps(self.model, None, ids)
        self.example = TrainExample(
            prompt_ids=jnp.ones((16, 2)), prompt_mask=jnp.ones((16, 2)),
            completion_ids=ids, completion_mask=jnp.ones((16, 3)),
            advantages=jnp.asarray(np.tile([-1., -.2, .3, .9], 4)),
            old_per_token_logps=old, ref_per_token_logps=jnp.zeros((16, 3)),
            filter_random_values=None)

    def grads(self, beta):
        from my_example.rebuttal.utility_runtime import production_per_unit_gradient
        config = replace(self.config, beta=beta)
        fn = production_per_unit_gradient(lambda model, example: self.loss(
            model, example, config, 0, 1)[0])
        with mock.patch("tunix.rl.common.compute_per_token_logps", side_effect=self.logps):
            _, tree = fn(self.model, self.example)
            return np.asarray(self.jax.device_get(self.jax.tree.leaves(tree)[0]))

    def test_policy_vmap_gradient_matches_analytic_on_policy_surrogate(self):
        gradient = self.grads(0.)
        advantage = np.asarray(self.example.advantages)
        expected = -advantage[:, None] * np.column_stack([self.features.mean(axis=1), np.ones(16)])
        np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-6)

    def test_configured_kl_is_in_full_gradient_not_policy_score(self):
        policy, full = self.grads(0.), self.grads(.08)
        logps = np.asarray(self.example.old_per_token_logps)
        coefficient = .08 * (1 - np.exp(-logps))
        expected_kl = np.column_stack([(coefficient * self.features).mean(axis=1), coefficient.mean(axis=1)])
        np.testing.assert_allclose(full - policy, expected_kl, rtol=2e-4, atol=1e-6)
        self.assertGreater(np.max(np.abs(full - policy)), 0.)

    def test_original_rank_reversal_changes_advantages_not_clean_heldout(self):
        from my_example.reward_rank_noise import RewardRankNoiseConfig, apply_reward_rank_noise
        from tunix.rl.grpo.grpo_learner import compute_advantages
        rewards = np.tile([0., 1., 2., 3.], 4)
        prompts = [f"prompt_{i}" for i in range(4) for _ in range(4)]
        corrupted = apply_reward_rank_noise(rewards, prompts, 4, RewardRankNoiseConfig(1., 0))
        clean = apply_reward_rank_noise(rewards, prompts, 4, RewardRankNoiseConfig(0., 0))
        self.assertTrue(corrupted.selected_groups.all())
        self.assertTrue(corrupted.effective_groups.all())
        np.testing.assert_array_equal(clean.corrupted_rewards, rewards)
        np.testing.assert_array_equal(corrupted.corrupted_rewards.reshape(-1, 4),
                                      rewards.reshape(-1, 4)[:, ::-1])
        np.testing.assert_allclose(compute_advantages(corrupted.corrupted_rewards, 4),
                                   -compute_advantages(clean.corrupted_rewards, 4))

    def test_independent_clean_loss_change_matches_first_order_direction(self):
        from my_example.rebuttal.utility_runtime import production_mean_loss, production_per_unit_gradient
        config = replace(self.config, beta=0.)
        loss = lambda model, example: self.loss(model, example, config, 0, 1)[0]
        mean_loss, per_unit = production_mean_loss(loss), production_per_unit_gradient(loss)
        features = self.jp.asarray(self.features ** 2 + .1)
        heldout = self.example.replace(completion_ids=features,
                                       old_per_token_logps=self.logps(self.model, None, features))
        theta = self.model.weight.value
        with mock.patch("tunix.rl.common.compute_per_token_logps", side_effect=self.logps):
            _, candidate_grads = per_unit(self.model, self.example)
            _, heldout_grads = per_unit(self.model, heldout)
            direction = self.jax.tree.leaves(candidate_grads)[0].mean(axis=0)
            h = self.jax.tree.leaves(heldout_grads)[0].mean(axis=0)
            baseline = float(mean_loss(self.model, heldout))
            eta = .01
            try:
                self.model.weight.value = theta - eta * direction
                after = float(mean_loss(self.model, heldout))
            finally:
                self.model.weight.value = theta
        predicted = eta * float(self.jp.vdot(h, direction))
        self.assertGreater(predicted, 0.)
        self.assertAlmostEqual(baseline - after, predicted, delta=3e-7)
        np.testing.assert_array_equal(np.asarray(self.model.weight.value), np.asarray(theta))


if __name__ == "__main__":
    unittest.main()
