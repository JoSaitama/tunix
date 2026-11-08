"""Server CPU regression for NNX snapshot aliases; no large model or TPU."""

from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


class ParameterSnapshotTest(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from my_example.rebuttal.parameter_snapshot import ParameterSnapshot
        self.jax, self.jp, self.nnx, self.Snapshot = jax, jnp, nnx, ParameterSnapshot

        class Toy(nnx.Module):
            def __init__(self):
                self.weight = nnx.LoRAParam(jnp.array([.25, -.5], dtype=jnp.float32))
                self.zeros = nnx.LoRAParam(jnp.array([-0., 0.], dtype=jnp.float32))
        self.model = Toy()
        self.guard = ParameterSnapshot(nnx.state(self.model, nnx.LoRAParam))

    def test_backup_survives_live_model_mutation(self):
        original = [np.asarray(x).copy() for x in self.jax.tree.leaves(self.guard.materialize())]
        temporary = self.jax.tree.map(lambda x: x + 1., self.nnx.state(self.model, self.nnx.LoRAParam))
        self.nnx.update(self.model, temporary)
        self.assertFalse(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))
        for before, after in zip(original, self.jax.tree.leaves(self.guard.materialize())):
            self.assertEqual(before.tobytes(), np.asarray(after).tobytes())
        self.nnx.update(self.model, self.guard.materialize())
        self.assertTrue(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))

    def test_materialized_wrappers_are_disposable(self):
        temporary_wrapper = self.guard.materialize()
        self.nnx.update(self.model, temporary_wrapper)
        self.model.weight.value = self.model.weight.value + .3
        self.assertFalse(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))
        self.nnx.update(self.model, self.guard.materialize())
        self.assertTrue(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))

    def test_signed_zero_and_tiny_changes_not_hidden_by_numeric_tolerance(self):
        self.model.zeros.value = self.jp.array([0., 0.], dtype=self.jp.float32)
        self.assertFalse(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))
        details = self.guard.differences(self.nnx.state(self.model, self.nnx.LoRAParam))
        self.assertEqual(details["changed_elements_bitwise"], 1)
        self.assertEqual(details["examples"][0]["max_abs_error"], 0.)
        self.nnx.update(self.model, self.guard.materialize())
        self.model.weight.value = self.model.weight.value.at[0].set(
            self.jp.nextafter(self.model.weight.value[0], self.jp.float32(1.)))
        self.assertFalse(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))

    def fake_runtime(self, output):
        from my_example.audit_grpo_checkpoint_restore import parameter_summary
        from my_example.rebuttal.utility_runtime import UtilityRuntime
        runtime = object.__new__(UtilityRuntime)
        runtime.args = SimpleNamespace(probe_max_grad_norm=.1)
        runtime.actor, runtime._snapshot, runtime.output = self.model, self.guard, Path(output)
        runtime.jax, runtime.jnp, runtime.nnx = self.jax, self.jp, self.nnx
        runtime.mesh, runtime.monitor = nullcontext(), lambda _: nullcontext()
        runtime.parameter_summary = parameter_summary
        runtime.initial_summary = parameter_summary(self.jax, runtime.snapshot)
        runtime.snapshot_memory = lambda _: None
        runtime.dot = lambda a, b: sum(self.jp.sum(x * y) for x, y in zip(
            self.jax.tree.leaves(a), self.jax.tree.leaves(b)))

        # Inject only a tiny deterministic perturbation; exercise the REAL
        # UtilityRuntime.probe precondition/finally/strict restore code below.
        def perturb(initial, direction, eta, unused_max):
            changed = self.jax.tree.map(lambda p, g: p - eta * g, initial, direction)
            delta = self.jax.tree.map(lambda p, q: q - p, initial, changed)
            count = sum(self.jp.count_nonzero(x) for x in self.jax.tree.leaves(delta))
            return changed, delta, self.jp.sqrt(runtime.dot(direction, direction)), self.jp.array(1.), count
        runtime.perturb = perturb
        runtime.clean_value = lambda _: float(runtime.dot(
            self.nnx.state(self.model, self.nnx.LoRAParam),
            self.nnx.state(self.model, self.nnx.LoRAParam)))
        return runtime

    def test_real_probe_restores_on_success_and_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.fake_runtime(tmp)
            direction = self.jax.tree.map(self.jp.ones_like, runtime.snapshot)
            baseline = runtime.clean_value(None)
            for eta in (.01, -.01, .03):
                result = runtime.probe(direction, direction, None, baseline, eta)
                self.assertTrue(result["parameter_delta_informative"])
                self.assertTrue(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))
            def fail(_):
                raise ValueError("injected clean evaluation failure")
            runtime.clean_value = fail
            with self.assertRaisesRegex(ValueError, "injected"):
                runtime.probe(direction, direction, None, baseline, .01)
            self.assertTrue(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))

    def test_real_probe_rejects_a_dirty_initial_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.fake_runtime(tmp)
            direction = self.jax.tree.map(self.jp.ones_like, runtime.snapshot)
            self.model.weight.value = self.model.weight.value + .1
            with self.assertRaisesRegex(RuntimeError, "before_probe"):
                runtime.probe(direction, direction, None, 0., .01)
            self.assertTrue((Path(tmp) / "parameter_restore_failure.json").is_file())

    def test_roundtrip_uses_real_runtime_and_unchanged_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.fake_runtime(tmp)
            result = runtime.roundtrip_check()
            self.assertTrue(result["passed"])
            self.assertEqual(result["roundtrips"], 2)
            self.assertFalse(result["utility_audit_completed"])
            self.assertTrue(self.guard.matches(self.nnx.state(self.model, self.nnx.LoRAParam)))


if __name__ == "__main__":
    unittest.main()
