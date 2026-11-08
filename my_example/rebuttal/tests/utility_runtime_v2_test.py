"""Server CPU regression of real diagnostic methods with tiny arrays only."""

from contextlib import nullcontext
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


class RuntimeV2Test(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from my_example.audit_grpo_checkpoint_restore import parameter_summary
        from my_example.rebuttal.parameter_snapshot import ParameterSnapshot
        from my_example.rebuttal.utility_runtime import UtilityRuntime

        class Toy(nnx.Module):
            def __init__(self):
                self.weight = nnx.LoRAParam(jnp.array([.25, -.5], dtype=jnp.float32))
        self.jax, self.jp, self.nnx = jax, jnp, nnx
        runtime = object.__new__(UtilityRuntime)
        self.runtime = runtime
        runtime.jax, runtime.jnp, runtime.nnx = jax, jnp, nnx
        runtime.actor = Toy()
        runtime._snapshot = ParameterSnapshot(nnx.state(runtime.actor, nnx.LoRAParam))
        runtime.parameter_summary = parameter_summary
        runtime.initial_summary = parameter_summary(jax, runtime.snapshot)
        runtime.args = SimpleNamespace(probe_max_grad_norm=.1, numerical_rtol=.1, numerical_atol=1e-7,
                                       zero_epsilon=1e-12, probe_step_sizes=[.001, .002], epsilon=.2)
        runtime.mesh, runtime.monitor = nullcontext(), lambda _: nullcontext()
        runtime.snapshot_memory = lambda _: None
        runtime.average = lambda tree, mask: jax.tree.map(
            lambda x: jnp.sum(x * mask[:, None], axis=0) / mask.sum(), tree)
        runtime.add = lambda a, b: jax.tree.map(lambda x, y: x + y, a, b)
        runtime.scale = lambda a, b: jax.tree.map(lambda x: x * b, a)
        runtime.dot = lambda a, b: sum(jnp.sum(x * y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

        def perturb(initial, direction, eta, maximum):
            norm = jnp.sqrt(runtime.dot(direction, direction))
            clip = jnp.minimum(1., maximum / jnp.maximum(norm, 1e-30)) if maximum > 0 else 1.
            changed = jax.tree.map(lambda p, g: p - eta * clip * g, initial, direction)
            delta = jax.tree.map(lambda p, q: q - p, initial, changed)
            count = sum(jnp.count_nonzero(x) for x in jax.tree.leaves(delta))
            return changed, delta, norm, jnp.asarray(clip), count
        runtime.perturb = perturb
        runtime.clean_value = lambda _: float(runtime.dot(nnx.state(runtime.actor, nnx.LoRAParam),
                                                          nnx.state(runtime.actor, nnx.LoRAParam)))
        runtime.as_example = lambda host: host
        runtime.unit_logps = lambda model, example: self.jp.asarray(example["old_per_token_logps"])

    def test_heldout_split_weighted_mean_with_unequal_half_counts(self):
        runtime = self.runtime
        runtime.gradients = lambda host: {"w": self.jp.full((16, 2), host["constant"])}
        hosts = [{"constant": x, "advantages": np.ones(16)} for x in (1., 2., 6.)]
        h, report = runtime.heldout_gradient(hosts)
        np.testing.assert_allclose(h["w"], [3., 3.])
        np.testing.assert_allclose(runtime._h_halves[0]["w"], [3.5, 3.5])
        np.testing.assert_allclose(runtime._h_halves[1]["w"], [2., 2.])
        self.assertEqual(report["split_half"]["prompts"], [8, 4])
        self.assertTrue(report["split_half"]["positive_alignment"])

    def test_one_batch_does_not_claim_split_stability(self):
        runtime = self.runtime
        runtime.gradients = lambda host: {"w": self.jp.ones((16, 2))}
        _, report = runtime.heldout_gradient([{"advantages": np.zeros(16)}])
        self.assertFalse(report["split_half"]["available"])
        self.assertIsNone(report["split_half"]["cosine"])
        self.assertEqual(report["zero_advantage_units"], 16)

    def heldout(self):
        return [{"completion_mask": np.ones((16, 2)),
                 "old_per_token_logps": np.zeros((16, 2), dtype=np.float32)}]

    def test_real_numerical_preflight_quadratic_pass_and_exact_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.runtime
            runtime.output = Path(tmp)
            h = runtime.scale(runtime.snapshot, 2.)
            baseline = runtime.clean_value(None)
            report = runtime.numerical_preflight(self.heldout(), h, baseline, baseline)
            self.assertTrue(report["passed"])
            self.assertEqual(report["validated_step_sizes"], [.001, .002])
            runtime.assert_original_parameters("test_end")

    def test_numerical_failure_is_reported_without_rewriting_old_logps(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.runtime
            runtime.output = Path(tmp)
            h = runtime.scale(runtime.snapshot, 2.)
            baseline = runtime.clean_value(None)
            runtime.clean_value = lambda _: baseline if runtime._snapshot.matches(
                self.nnx.state(runtime.actor, self.nnx.LoRAParam)) else baseline + .1
            host = self.heldout()
            old = host[0]["old_per_token_logps"].copy()
            report = runtime.numerical_preflight(host, h, baseline, baseline)
            self.assertFalse(report["passed"])
            self.assertEqual(report["validated_step_sizes"], [])
            np.testing.assert_array_equal(old, host[0]["old_per_token_logps"])
            runtime.assert_original_parameters("test_end")

    def archive_host(self):
        from tunix.rl.grpo.grpo_learner import compute_advantages
        prompts = np.repeat([f"prompt{i}" for i in range(4)], 4)
        rewards = np.tile([0., 1., 2., 3.], 4)
        return {"prompt_ids": np.ones((16, 2), dtype=np.int32),
                "completion_ids": np.ones((16, 3), dtype=np.int32),
                "completion_mask": np.ones((16, 3), dtype=np.int32),
                "old_per_token_logps": np.zeros((16, 3), dtype=np.float32),
                "ref_per_token_logps": np.zeros((16, 3), dtype=np.float32),
                "advantages": compute_advantages(rewards, 4).astype(np.float32),
                "clean_rewards": rewards, "observed_rewards": rewards,
                "correct": np.zeros(16, dtype=bool), "noise_selected": np.zeros(4, dtype=bool),
                "noise_effective": np.zeros(4, dtype=bool), "text": np.array(["x"] * 16),
                "answers": np.array(["1"] * 16), "prompt_strings": prompts,
                "prompt_ids_hash": np.array([hashlib.sha256(p.encode()).hexdigest()[:16] for p in prompts]),
                "source_indices": np.repeat(np.arange(4), 4), "rollout_seed": np.array(0)}

    def test_replay_reapplies_candidate_noise_and_keeps_heldout_clean(self):
        from my_example.rebuttal.utility_data import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.runtime
            runtime.output = Path(tmp) / "out"
            (runtime.output / "fixed_rollouts").mkdir(parents=True)
            runtime.args.candidate_noise, runtime.args.noise_seed = 1., 0
            path = Path(tmp) / "original.npz"
            original = self.archive_host()
            np.savez_compressed(path, **original)
            digest = sha256_file(path)
            candidate = runtime.replay_prepare(path, list(range(4)), role="candidate", batch_id=0, original_args={})
            heldout = runtime.replay_prepare(path, list(range(4)), role="heldout", batch_id=0, original_args={})
            np.testing.assert_array_equal(candidate["observed_rewards"], np.tile([3., 2., 1., 0.], 4))
            np.testing.assert_allclose(candidate["advantages"], -original["advantages"])
            np.testing.assert_array_equal(heldout["observed_rewards"], original["clean_rewards"])
            self.assertFalse(heldout["noise_selected"].any())
            self.assertEqual(digest, sha256_file(path))

    def test_legacy_replay_rejects_new_noise_and_accepts_same_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.runtime
            runtime.output = Path(tmp) / "out"
            (runtime.output / "fixed_rollouts").mkdir(parents=True)
            runtime.args.candidate_noise, runtime.args.noise_seed = 1., 0
            host = self.archive_host()
            host.pop("prompt_strings")
            path = Path(tmp) / "legacy.npz"
            np.savez_compressed(path, **host)
            original_args = {"candidate_noise": 0., "noise_seed": 0}
            with self.assertRaisesRegex(ValueError, "legacy replay lacks"):
                runtime.replay_prepare(path, list(range(4)), role="candidate", batch_id=0, original_args=original_args)
            runtime.args.candidate_noise = 0.
            result = runtime.replay_prepare(path, list(range(4)), role="candidate", batch_id=0, original_args=original_args)
            np.testing.assert_array_equal(result["advantages"], host["advantages"])


if __name__ == "__main__":
    unittest.main()
