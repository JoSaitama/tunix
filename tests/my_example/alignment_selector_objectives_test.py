"""CPU mathematical regression tests; no TPU/JAX/Flax dependency required."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from my_example.alignment_baselines.selector_objectives import (
    selector_loss_from_logps, selector_metadata,
)
from my_example.alignment_baselines.projection_hash import bucket_and_sign


class SelectorObjectivesTest(unittest.TestCase):
    def loss(self, logps, mask, advantage=0.7, **kwargs):
        return selector_loss_from_logps(logps, mask, advantage,
                                        array_module=np, **kwargs)

    def test_method_specific_beta_is_config_driven(self):
        for beta in (0.0, 0.08, 0.123):
            self.assertEqual(selector_metadata("learnalign", beta)["selector_beta"], beta)
            self.assertEqual(selector_metadata("gradalign", beta)["selector_beta"], 0.0)

    def test_gradalign_sums_tokens_and_ignores_kl(self):
        a = self.loss([-2.0], [1], method="gradalign", beta=0.08)
        b = self.loss([-2.0, -2.0], [1, 1], method="gradalign", beta=0.08)
        self.assertAlmostEqual(float(b), 2 * float(a))
        self.assertEqual(float(b), float(self.loss([-2.0, -2.0], [1, 1],
                                                method="gradalign", beta=0.0)))

    def test_learnalign_averages_tokens(self):
        a = self.loss([-2.0], [1], method="learnalign", beta=0.0)
        b = self.loss([-2.0, -2.0], [1, 1], method="learnalign", beta=0.0)
        self.assertAlmostEqual(float(a), float(b))

    def test_learnalign_eq7_kl_gradient_coefficient(self):
        logps = np.asarray([-2.0, -1.0], dtype=np.float32)
        ref = np.asarray([-1.8, -1.4], dtype=np.float32)
        beta, advantage = 0.08, 0.7
        numerical = []
        for i in range(2):
            plus, minus = logps.copy(), logps.copy()
            plus[i] += 0.002
            minus[i] -= 0.002
            kwargs = dict(method="learnalign", beta=beta, reference_logps=ref)
            numerical.append((self.loss(plus, [1, 1], **kwargs)
                              - self.loss(minus, [1, 1], **kwargs)) / 0.004)
        expected = -(advantage + beta * (np.exp(ref - logps) - 1)) / 2
        np.testing.assert_allclose(numerical, expected, atol=5e-5)

    def test_nonzero_learnalign_beta_requires_reference(self):
        with self.assertRaisesRegex(ValueError, "requires reference"):
            self.loss([-2.0], [1], method="learnalign", beta=0.08)

    def test_padding_does_not_affect_either_objective(self):
        for method in ("learnalign", "gradalign"):
            beta = 0.08
            a = self.loss([-2.0], [1], method=method, beta=beta, reference_logps=[-1.8])
            b = self.loss([-2.0, -10000.0], [1, 0], method=method,
                          beta=beta, reference_logps=[-1.8, 10000.0])
            self.assertAlmostEqual(float(a), float(b))
            self.assertEqual(float(self.loss([-2.0], [0], method=method,
                                            beta=beta, reference_logps=[-1.8])), 0.0)

    def test_sign_is_fixed_but_not_bucket_parity(self):
        coords = np.arange(65536, dtype=np.uint32)
        kwargs = dict(leaf_index=0, dimension=4096, seed=42, xp=np)
        with np.errstate(over="ignore"):
            bucket, sign = bucket_and_sign(coords, **kwargs)
            bucket2, sign2 = bucket_and_sign(coords, **kwargs)
        np.testing.assert_array_equal(bucket, bucket2)
        np.testing.assert_array_equal(sign, sign2)
        both_sign_buckets = sum(np.unique(sign[bucket == b]).size == 2 for b in range(4096))
        self.assertGreater(both_sign_buckets, 4000)
        parity = np.where(bucket % 2 == 0, 1.0, -1.0)
        self.assertLess(abs(float(np.mean(sign == parity)) - 0.5), 0.02)

    def test_projection_remains_linear_for_rollout_subbatch_mean(self):
        rng = np.random.default_rng(123)
        gradients = rng.normal(size=(4, 8, 100)).astype(np.float32)
        with np.errstate(over="ignore"):
            bucket, sign = bucket_and_sign(np.arange(100, dtype=np.uint32),
                                          leaf_index=0, dimension=32, seed=5, xp=np)
        def project(rows):
            return np.asarray([np.bincount(bucket, weights=row * sign,
                                          minlength=32) for row in rows])
        whole = project(gradients.mean(axis=1))
        split = (project(gradients[:, :4].mean(axis=1))
                 + project(gradients[:, 4:].mean(axis=1))) / 2
        np.testing.assert_allclose(whole, split, atol=3e-7)

    def test_reference_forward_is_bounded_and_method_specific(self):
        # Execute the production helper with host stand-ins; importing the full
        # TPU stack is not needed to check scheduling/argument wiring.
        source = Path(__file__).resolve().parents[2] / "my_example/alignment_baselines/gradient_features.py"
        tree = ast.parse(source.read_text())
        estimator_class = next(node for node in tree.body
                               if isinstance(node, ast.ClassDef) and node.name == "PromptGradientEstimator")
        helper = next(node for node in estimator_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_with_reference_logps")
        helper.returns = None
        for argument in helper.args.args:
            argument.annotation = None
        namespace = {"jnp": np, "jax": SimpleNamespace(block_until_ready=lambda x: x,
                     lax=SimpleNamespace(stop_gradient=lambda x: x))}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        calls = []
        def reference(**kwargs):
            calls.append(kwargs)
            return np.zeros_like(kwargs["completion_tokens"], dtype=np.float32)
        rollout = SimpleNamespace(pad_id=lambda: 0, eos_id=lambda: 1)
        cluster = SimpleNamespace(rollout=rollout, get_ref_per_token_logps=reference)
        example = SimpleNamespace(prompt_ids=np.ones((16, 3)), completion_ids=np.ones((16, 5)))
        example.replace = lambda **kwargs: kwargs
        for method in ("gradalign", "learnalign"):
            instance = SimpleNamespace(definition=selector_metadata(method, 0.08),
                                       rl_cluster=cluster, selection_micro_batch_size=4)
            result = namespace["_with_reference_logps"](instance, example)
            if method == "gradalign":
                self.assertIs(result, example)
                self.assertFalse(calls)
            else:
                self.assertEqual(len(calls), 4)
                self.assertEqual(calls[-1]["micro_batch_size"], 4)
                self.assertTrue(all(call["prompt_tokens"].shape[0] <= 4 for call in calls))
                self.assertEqual(result["ref_per_token_logps"].shape, (16, 5))


@unittest.skipUnless(importlib.util.find_spec("jax") is not None, "JAX unavailable locally; run on server")
class SelectorAutodiffTest(unittest.TestCase):
    def test_actual_jax_derivatives_match_paper_coefficients(self):
        import jax
        import jax.numpy as jnp
        with jax.default_device(jax.devices("cpu")[0]):
            logps = jnp.asarray([-2.0, -1.0, -3.0])
            ref = jnp.asarray([-1.8, -1.4, -2.0])
            mask = jnp.asarray([1.0, 1.0, 0.0])
            for method in ("learnalign", "gradalign"):
                derivative = jax.grad(lambda x: selector_loss_from_logps(
                    x, mask, 0.7, method=method, beta=0.08, reference_logps=ref))(logps)
                expected = -(0.7 + 0.08 * (jnp.exp(ref - logps) - 1)) * mask / 2
                if method == "gradalign":
                    expected = -0.7 * mask
                np.testing.assert_allclose(derivative, expected, rtol=1e-6, atol=1e-7)

    def test_jax_signed_hash_matches_numpy_and_is_linear(self):
        import jax
        import jax.numpy as jnp
        with jax.default_device(jax.devices("cpu")[0]):
            coords = np.arange(128, dtype=np.uint32)
            with np.errstate(over="ignore"):
                buckets, signs = bucket_and_sign(coords, leaf_index=2, dimension=32, seed=42, xp=np)
            jax_buckets, jax_signs = bucket_and_sign(jnp.asarray(coords), leaf_index=2,
                                                   dimension=32, seed=42, xp=jnp)
            np.testing.assert_array_equal(jax_buckets, buckets)
            np.testing.assert_array_equal(jax_signs, signs)
            # Exercise the actual production tree projection, not a mirror.
            from my_example.alignment_baselines.gradient_features import project_gradient_tree
            a = {"one": jnp.arange(4 * 128, dtype=jnp.float32).reshape(4, 128) / 100}
            b = {"one": jnp.sin(a["one"])}
            kwargs = dict(projection_dim=32, seed=42)
            combined = project_gradient_tree({"one": a["one"] + b["one"]}, **kwargs)
            parts = project_gradient_tree(a, **kwargs) + project_gradient_tree(b, **kwargs)
            np.testing.assert_allclose(combined, parts, atol=1e-5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
