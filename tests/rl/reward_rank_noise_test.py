"""Tests for deterministic reward-rank mismatch."""

from absl.testing import absltest
import numpy as np
from tunix.rl import reward_rank_noise


class RewardRankNoiseTest(absltest.TestCase):

  def test_reverse_preserves_values_and_reverses_ranks(self):
    result = reward_rank_noise.reverse_reward_ranks(np.array([1, 0, 1, 0]))
    np.testing.assert_array_equal(result, np.array([0, 1, 0, 1]))

  def test_selected_groups_are_nested_by_fraction(self):
    prompts = [f"prompt-{index}" for index in range(100)]
    low = reward_rank_noise.RewardRankNoiseConfig(0.2, 21)
    high = reward_rank_noise.RewardRankNoiseConfig(0.4, 21)
    low_set = {p for p in prompts if reward_rank_noise.selected_prompt(p, low)}
    high_set = {p for p in prompts if reward_rank_noise.selected_prompt(p, high)}
    self.assertTrue(low_set <= high_set)

  def test_binary_group_mean_and_std_are_preserved(self):
    config = reward_rank_noise.RewardRankNoiseConfig(1.0, 0)
    audit = reward_rank_noise.apply_reward_rank_noise(
        np.array([1, 0, 1, 0, 0, 0, 0, 0]),
        ["a"] * 4 + ["b"] * 4,
        4,
        config,
    )
    clean = audit.clean_rewards.reshape(2, 4)
    corrupt = audit.corrupted_rewards.reshape(2, 4)
    np.testing.assert_allclose(clean.mean(1), corrupt.mean(1))
    np.testing.assert_allclose(clean.std(1), corrupt.std(1))
    self.assertTrue(audit.effective_groups[0])
    self.assertFalse(audit.effective_groups[1])


if __name__ == "__main__":
  absltest.main()
