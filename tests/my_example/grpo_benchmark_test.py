"""Offline tests for the opt-in benchmark (no model or TPU)."""
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

from my_example.rebuttal import grpo_benchmark as bench


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        self.bench = bench.Benchmark("vanilla", "timing", "off", 1, 2, self.path, "v5p-16")
        self.bench.metadata = {}
        self.bench.learner_entry = time.perf_counter()

    def test_validation_rejects_memory_with_io(self):
        with self.assertRaisesRegex(ValueError, "diagnostics off"):
            bench.Benchmark("dtv", "memory", "on", 1, 2, self.path, "v5p-16")

    def test_timing_stop_only_after_two_measured_iterations(self):
        b = self.bench
        b.actor = SimpleNamespace(train_steps=0)
        b.cluster = SimpleNamespace(global_steps=0)
        b._drain = Mock()
        b._optimizer_steps = lambda: b.actor.train_steps
        for i in range(1, 4):
            b.before_iteration()
            b.note_prompts(["a", "b", "c", "d"])
            b.note_rollout(16, 20 * i)
            b.actor.train_steps = b.cluster.global_steps = i
            self.assertEqual(b.after_iteration(), i == 3)
        self.assertEqual(b._drain.call_count, 2)
        self.assertEqual(b.end_optimizer_step - b.start_optimizer_step, 2)
        self.assertAlmostEqual(sum(x["seconds"] for x in b.rows[1:]), b.window_seconds)
        self.assertEqual(bench.summarize(b.rows[1:], b.window_seconds)["generated_tokens"], 100)

    def test_wrong_batch_and_duplicate_input_are_rejected(self):
        b = self.bench
        b.before_iteration()
        b.note_prompts(["a"])
        with self.assertRaisesRegex(ValueError, "one input batch"):
            b.note_prompts(["a"])
        with self.assertRaisesRegex(ValueError, "full nonempty"):
            b.after_iteration()

    def test_compilation_attributed_but_not_conflated_with_cache_hits(self):
        b = self.bench
        b.active_iteration = 16
        handler = bench._CompileLog(b)
        with b.phase("rollout"):
            for msg in ("Persistent compilation cache hit", "Finished XLA compilation of jit(f)"):
                handler.emit(logging.LogRecord("jax.test", 30, "", 1, msg, (), None))
        self.assertEqual([e["iteration"] for e in b.compiler_events], [16, 16])
        self.assertEqual([e["phase"] for e in b.compiler_events], ["rollout", "rollout"])
        self.assertEqual([e["xla_compile_completed_log"] for e in b.compiler_events], [False, True])

    def test_unchanged_peak_is_never_a_window_peak(self):
        def stats(peak, used):
            return {"stats": {"peak_bytes_in_use": peak, "bytes_in_use": used}}

        result = bench.memory_report({
            "window_start": {"0": stats(12, 7), "1": stats(13, 8), "2": {"stats": None}},
            "window_end": {"0": stats(12, 8), "1": stats(17, 9), "2": {"stats": None}},
        }, [])
        self.assertIsNone(result["devices"]["0"]["new_allocator_high_water_within_window_bytes"])
        self.assertEqual(result["devices"]["1"]["new_allocator_high_water_within_window_bytes"], 17)
        self.assertIsNone(result["max_device_allocator_lifetime_peak_bytes"])
        self.assertIsNone(result["verified_max_device_window_peak_hbm_bytes"])

    def _fixture(self, mode="timing"):
        b = self.bench
        b.mode = mode
        b.metadata = {
            "schedule_optimizer_steps": 691, "actor_trainer": "Trainer", "valuation": False,
            "commit_sha": "test", "topology": {"devices": [{"id": i} for i in range(4)]},
        }
        b.batching = {"full": 4}
        b.warmup_seconds = 2.5
        b.teardown_seconds = 0.1
        b.start_optimizer_step = 1
        b.end_optimizer_step = 3
        b.window_seconds = 5
        b.rows = [
            {
                "iteration": i, "window": "warmup" if i == 1 else "measure",
                "seconds": 2.5, "prompts": 4, "completions": 16,
                "generated_tokens": 100, "prompt_order_sha256": "f" * 64,
            }
            for i in range(1, 4)
        ]
        if mode == "memory":
            b.memory = {
                label: {str(i): {"stats": None} for i in range(4)}
                for label in ("window_start", "window_end")
            }
            (self.path / "trace").mkdir()
            (self.path / "trace/test.xplane.pb").touch()
        b._write_reports()
        (self.path / "exit_code").write_text("0\n")
        return b

    def test_checks_counts_and_absence_of_production_output(self):
        self._fixture()
        summary, _, _ = bench.check_run(self.path)
        self.assertEqual(summary["measurement"]["optimizer_updates"], 2)
        (self.path / "checkpoints").mkdir()
        with self.assertRaisesRegex(ValueError, "unexpected checkpoints"):
            bench.check_run(self.path)

    def test_xprof_qualification_requires_all_devices_and_resident_evidence(self):
        self._fixture("memory")
        evidence = self.path / "evidence.json"
        payload = {
            "collector": "XProf Memory Profile", "peak_scope": "measurement_window",
            "resident_allocations_verified": True, "verification_notes": "synthetic fixture",
            "iterations_inclusive": [2, 3],
            "devices": {str(i): {"peak_hbm_bytes": 100 + i} for i in range(4)},
        }
        evidence.write_text(json.dumps(payload))
        self.assertEqual(bench.verified_xprof_peaks(self.path, evidence)["max_device_peak_hbm_bytes"], 103)
        del payload["devices"]["3"]
        evidence.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "every mesh device"):
            bench.verified_xprof_peaks(self.path, evidence)

    def test_comparable_rejects_different_initial_model_or_order(self):
        keys = (
            "commit_sha", "versions", "topology", "config", "schedule_optimizer_steps",
            "base_model_revision", "initial_lora_sha256", "batching", "eos_tokens",
            "actor_parameter_dtypes", "jax_default_matmul_precision", "jax_enable_x64",
            "jax_enable_compilation_cache",
        )
        meta = dict.fromkeys(keys, "same")
        baseline = ({}, meta, ["first", "second"])
        bench.check_comparable([baseline, ({}, meta, ["first"])])
        with self.assertRaisesRegex(ValueError, "Prompt order differs"):
            bench.check_comparable([baseline, ({}, meta, ["second"])])
        with self.assertRaisesRegex(ValueError, "initial_lora_sha256"):
            bench.check_comparable([baseline, ({}, {**meta, "initial_lora_sha256": "bad"}, ["first"])])

    @unittest.skipUnless(importlib.util.find_spec("flax"), "Flax is optional for offline checks")
    def test_real_nnx_optax_counter_and_barrier_on_cpu(self):
        from flax import nnx
        import jax
        import jax.numpy as jnp
        import optax

        with jax.default_device(jax.devices("cpu")[0]):
            model = nnx.Linear(1, 1, rngs=nnx.Rngs(13))
            optimizer = nnx.Optimizer(
                model, optax.MultiSteps(optax.adamw(1e-6), 1), wrt=nnx.Param,
            )
            b = self.bench
            b.jax, b.nnx = jax, nnx
            b.actor = SimpleNamespace(model=model, optimizer=optimizer)
            b.cluster = SimpleNamespace(rollout=SimpleNamespace(model=lambda: model))
            self.assertEqual(b._optimizer_steps(), 0)
            grads = nnx.grad(lambda m: m(jnp.ones((1, 1))).sum())(model)
            optimizer.update(model, grads)
            b._drain()
            self.assertEqual(b._optimizer_steps(), 1)


if __name__ == "__main__":
    unittest.main()
