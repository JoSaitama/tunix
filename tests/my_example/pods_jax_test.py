"""Real JAX/Flax TrainExample gather/split test; CPU backend is sufficient."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("jax") and importlib.util.find_spec("flax"),
                     "requires the training JAX/Flax environment")
class PODSJAXTest(unittest.TestCase):
    def test_real_struct_and_real_micro_batch_utilities(self):
        import jax.numpy as jnp
        from tunix.rl import rl_cluster
        from tunix.rl.grpo.grpo_learner import TrainExample
        from my_example.pods import pods_learner_class, subset_advantages

        class Base:
            def __init__(self, rl_cluster, algo_config):
                self.rl_cluster, self.algo_config = rl_cluster, algo_config

            def _compute_rewards(self, prompts, completions, mode, **kwargs):
                return np.tile([0., .5, 3., 4.], len(prompts)//4)

            def _generate_and_compute_advantage(self, data, mode):
                rewards = self._compute_rewards(data["prompts"], [], mode)
                ids = jnp.asarray(data["ids"])[:, None]
                return TrainExample(
                    prompt_ids=ids, prompt_mask=ids+100,
                    completion_ids=ids+200, completion_mask=ids+300,
                    ref_per_token_logps=ids+400, old_per_token_logps=None,
                    advantages=jnp.asarray(subset_advantages(rewards.reshape(-1,4)).reshape(-1)),
                    filter_random_values=jnp.concatenate([ids, ids], axis=1),
                )

        cluster = SimpleNamespace(
            cluster_config=SimpleNamespace(training_config=SimpleNamespace(
                use_dynamic_batch_curation=False, max_steps=691)),
            global_steps=0, buffer_metrics=lambda *args, **kwargs: None,
        )
        learner = pods_learner_class(Base)(
            rl_cluster=cluster, algo_config=SimpleNamespace(num_generations=4, beta=.08),
            pods_keep_count=2,
        )
        def batch(start):
            return {"prompts": np.repeat([f"p{i}" for i in range(start,start+4)],4),
                    "ids": np.arange(start*4,(start+4)*4)}
        chunks = learner._process_accumulated_batches(
            [batch(0),batch(4)], [16,16], rl_cluster.Mode.TRAIN)
        self.assertEqual([tuple(x.advantages.shape) for x in chunks], [(8,),(8,)])
        for offset, chunk in zip((0,16), chunks):
            expected = np.array([3,0,7,4,11,8,15,12])[:,None]+offset
            np.testing.assert_array_equal(chunk.prompt_ids, expected)
            np.testing.assert_array_equal(chunk.ref_per_token_logps, expected+400)
            self.assertIsNone(chunk.old_per_token_logps)
            self.assertEqual(chunk.filter_random_values.shape, (8,2))


if __name__ == "__main__":
    unittest.main()
