"""SERVER CPU tests: tiny real JAX/NNX arrays, no Gemma/TPU/checkpoint download."""

from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from my_example.rebuttal.utility_fp32_control import (
    check_forward_precision, clean_host, configure_fp32, control_gradient,
    logp_difference, parse_args, run_pairs, traced_float_dtypes,
)


class FP32ArgumentsTest(unittest.TestCase):
    def arguments(self):
        return ["--input-dir", "/unused/source", "--output-dir", "/unused/out",
                "--reward-mode", "accuracy", "--replay-rollouts", "/unused/replay",
                "--backbone-path", "/unused/backbone"]

    def parse(self, extra=()):
        partition = {"candidate_indices": list(range(4)), "heldout_indices": list(range(2764, 2772))}
        with mock.patch("my_example.rebuttal.audit_grpo_utility.replay_metadata",
                        return_value=(None, {"data_partition": partition})):
            return parse_args(self.arguments() + list(extra))

    def test_control_is_bounded_and_predeclared(self):
        args = self.parse()
        self.assertEqual((args.heldout_prompts, args.candidate_prompts), (8, 4))
        self.assertEqual(args.probe_step_sizes, [1e-6, 1e-4])
        self.assertEqual(args.mode, "numerics")

    def test_rejects_large_scan_precision_gate_relaxation_and_step_search(self):
        for extra in (["--mode", "score"], ["--probe-step-sizes", ".001"],
                      ["--numerical-rtol", ".2"], ["--min-headroom-fraction", ".01"],
                      ["--heldout-reward-mode", "binary"], ["--probe-max-grad-norm", "0"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.parse(extra)

    def test_logp_difference_checks_actual_fp32_dtype_and_valid_mask(self):
        old = np.zeros((16, 2), np.float32)
        new = old.copy()
        new[0, 0] = 1.
        mask = np.ones_like(old, dtype=bool)
        mask[0, 0] = False
        report = logp_difference(new, old, mask, .2)
        self.assertEqual(report["max_abs"], 0.)
        self.assertEqual(report["valid_tokens"], 31)
        with self.assertRaises(ValueError):
            logp_difference(new.astype(np.float16), old, mask, .2)
        np.testing.assert_array_equal(old, np.zeros_like(old))


class FP32RuntimeTest(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from my_example.audit_grpo_checkpoint_restore import parameter_summary
        from my_example.rebuttal.parameter_snapshot import ParameterSnapshot
        from my_example.rebuttal.utility_runtime import UtilityRuntime
        from tunix.rl.grpo.grpo_learner import GRPOConfig, TrainExample

        @dataclass(frozen=True)
        class Config:
            param_dtype: object = jnp.bfloat16

        class Toy(nnx.Module):
            def __init__(self):
                self.base = nnx.Param(jnp.array([.25, -.125], dtype=jnp.bfloat16))
                self.weight = nnx.LoRAParam(jnp.array([.02, -.03], dtype=jnp.float32))
                self.counter = nnx.Variable(jnp.asarray(3, dtype=jnp.int32))
                self.config = Config()

        runtime = object.__new__(UtilityRuntime)
        self.runtime = runtime
        self.jax, self.jp, self.nnx = jax, jnp, nnx
        runtime.jax, runtime.jnp, runtime.nnx = jax, jnp, nnx
        runtime.mesh, runtime.monitor = nullcontext(), lambda _: nullcontext()
        runtime.actor = Toy()
        runtime.TrainExample, runtime.pad_id, runtime.eos_id = TrainExample, 0, 1
        runtime.algo = GRPOConfig(num_generations=4, beta=.08)
        runtime.args = SimpleNamespace(zero_epsilon=1e-12)
        runtime.parameter_summary = parameter_summary
        runtime._snapshot = ParameterSnapshot(nnx.state(runtime.actor, nnx.LoRAParam))
        runtime.initial_summary = parameter_summary(jax, runtime.snapshot)
        runtime.snapshot_memory = lambda _: None
        runtime.dot = lambda a, b: sum(jnp.sum(x * y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
        runtime.add = lambda a, b: jax.tree.map(lambda x, y: x + y, a, b)
        runtime.scale = lambda a, scale: jax.tree.map(lambda x: x * scale, a)
        self.tokens = jnp.arange(48).reshape(16, 3)

        def logps(model, prompt_tokens, completion_tokens, **unused):
            return model.weight.value[0] * completion_tokens.astype(jnp.float32) / 100 + model.weight.value[1]
        self.logps = logps
        self.host = {"prompt_ids": np.ones((16, 2), np.int32),
                     "completion_ids": np.asarray(self.tokens), "completion_mask": np.ones((16, 3)),
                     "advantages": np.tile([-1., -.3, .4, .9], 4).astype(np.float32),
                     "old_per_token_logps": np.asarray(logps(runtime.actor, None, self.tokens))}

    def test_promotion_is_value_preserving_and_does_not_change_integer_state(self):
        before = np.asarray(self.runtime.actor.base.value, dtype=np.float32)
        report = configure_fp32(self.runtime)
        self.assertEqual(self.runtime.actor.base.value.dtype, self.jp.float32)
        np.testing.assert_array_equal(self.runtime.actor.base.value, before)
        self.assertEqual(self.runtime.actor.counter.value.dtype, self.jp.int32)
        self.assertEqual(self.runtime.actor.config.param_dtype, self.jp.float32)
        self.assertEqual(report["source_restored_lora"], report["fp32_control_lora"])
        self.runtime.assert_original_parameters("tiny_fp32_promotion")

    def test_mean_scalar_gradient_matches_analytic_and_finite_difference(self):
        configure_fp32(self.runtime)
        old = self.host["old_per_token_logps"].copy()
        with mock.patch("tunix.rl.common.compute_per_token_logps", side_effect=self.logps):
            h, info = control_gradient(self.runtime, [self.host, self.host])
            gradient = self.jax.tree.leaves(h)[0]
            expected = -np.mean(self.host["advantages"][:, None] * np.column_stack([
                np.asarray(self.tokens).mean(1) / 100., np.ones(16)]), axis=0)
            np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-7)
            baseline = self.runtime.clean_value([self.host])
            theta = self.runtime.actor.weight.value
            try:
                self.runtime.actor.weight.value = theta - .001 * gradient
                actual = baseline - self.runtime.clean_value([self.host])
            finally:
                self.runtime.actor.weight.value = theta
            self.assertAlmostEqual(actual, .001 * float(self.jp.vdot(gradient, gradient)), delta=1e-7)
        self.assertEqual(info["units"], 32)
        np.testing.assert_array_equal(old, self.host["old_per_token_logps"])
        self.runtime.assert_original_parameters("tiny_fp32_end")

    def test_actual_mean_loss_jaxpr_and_hidden_bf16_detection(self):
        configure_fp32(self.runtime)
        with mock.patch("tunix.rl.common.compute_per_token_logps", side_effect=self.logps):
            report = check_forward_precision(self.runtime, self.host)
        self.assertEqual(report["floating_intermediate_dtypes"], ["float32"])
        traced = self.jax.make_jaxpr(lambda x: x.astype(self.jp.bfloat16).astype(self.jp.float32))(
            self.jp.ones(2, self.jp.float32))
        self.assertIn("bfloat16", traced_float_dtypes(traced))

    def test_clean_archive_validation_rejects_pollution_and_does_not_rewrite(self):
        from tunix.rl.grpo.grpo_learner import compute_advantages
        host = dict(self.host, clean_rewards=np.tile([0., 1., 2., 3.], 4),
                    observed_rewards=np.tile([0., 1., 2., 3.], 4),
                    noise_selected=np.zeros(4, bool), noise_effective=np.zeros(4, bool))
        host["advantages"] = compute_advantages(host["clean_rewards"], 4).astype(np.float32)
        result = clean_host(host)
        self.assertIsNot(result, host)
        with self.assertRaisesRegex(ValueError, "corrupted"):
            clean_host(dict(host, noise_selected=np.ones(4, bool)))
        with self.assertRaisesRegex(ValueError, "advantages"):
            clean_host(dict(host, advantages=np.zeros(16, np.float32)))


class FP32GateTest(unittest.TestCase):
    def test_failed_gate_never_reports_method_benefit_or_production_validation(self):
        # Explicit NumPy-only report plumbing fixture, not model/HBM evidence.
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            runtime = SimpleNamespace(output=Path(tmp), args=SimpleNamespace(
                numerical_atol=1e-7, probe_step_sizes=[1e-6, 1e-4]))
            runtime.clean_value = lambda _: 0.

            def probe(direction, h, heldout, baseline, eta):
                return {"changed_parameter_elements": int(eta != 0),
                        "utility": -.001 if eta else 0.,
                        "predicted_first_order_utility": eta * .1}
            runtime.probe = probe
            result = run_pairs(runtime, [], None, 0., 0.)
            self.assertFalse(result["passed"])
            self.assertFalse(result["production_path_validated"])
            self.assertTrue(result["not_method_evidence"])
            self.assertEqual(len(result["pairs"]), 2)
            self.assertEqual(json.loads((Path(tmp) / "numerical_report.json").read_text())["validated_step_sizes"], [])


if __name__ == "__main__":
    unittest.main()
