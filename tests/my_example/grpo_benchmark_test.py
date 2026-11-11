"""Offline tests for the opt-in benchmark (no model or TPU)."""
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

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

    def test_ds_counts_rejected_rollouts_and_one_update(self):
        b = self.bench
        b.method = "grpo-ds"
        b.actor = SimpleNamespace(train_steps=1)
        b.cluster = SimpleNamespace(global_steps=1)
        b.before_iteration()
        b.note_rollout(16, 30)
        b.note_ds_candidate(["a", "b", "c", "d"], eligible=3, selected=3)
        b.note_rollout(16, 20)
        b.note_ds_candidate(["e", "f", "g", "h"], eligible=4, selected=1)
        self.assertFalse(b.after_iteration())
        row = b.rows[0]
        self.assertEqual((row["generated_tokens"], row["update_completions"]), (50, 16))
        self.assertEqual((row["ds_zero_variance_groups"], row["ds_excess_eligible_groups"]), (1, 3))

    def test_pods_requires_eight_actor_rows(self):
        self.bench.method = "pods"
        self.bench.check_actor_input([SimpleNamespace(completion_ids=SimpleNamespace(shape=(8, 4)))])
        with self.assertRaisesRegex(ValueError, "expected 8 actor rows"):
            self.bench.check_actor_input([SimpleNamespace(completion_ids=SimpleNamespace(shape=(16, 4)))])

    def test_gradalign_cycle_validation_and_first_selector_memory(self):
        with self.assertRaisesRegex(ValueError, "complete ten-update cycles"):
            bench.Benchmark("gradalign", "timing", "off", 15, 50, self.path, "v5p-16")
        b = bench.Benchmark("gradalign", "memory", "off", 0, 1, self.path, "v5p-16")
        self.assertEqual(b.first_batch_size, 4)
        b.metadata = {}
        b.learner_entry = time.perf_counter()
        b.actor = SimpleNamespace(train_steps=0)
        b.cluster = SimpleNamespace(global_steps=0)
        b._drain = Mock()
        b._optimizer_steps = lambda: b.actor.train_steps
        b._memory_snapshot = Mock()
        b._start_trace = Mock()
        b._stop_trace = Mock()
        with self.assertRaisesRegex(ValueError, "outside"):
            with b.selector_round():
                pass
        b.before_iteration()
        b._start_trace.assert_called_once()
        with b.selector_round():
            b.note_selector_rollout(120, 200, candidate=False)
            b.note_selector_rollout(640, 800, candidate=True)
        b.note_prompts(["a", "b", "c", "d"])
        b.note_rollout(16, 100)
        b.actor.train_steps = b.cluster.global_steps = 1
        self.assertTrue(b.after_iteration())
        result = bench.summarize(b.rows, 2)
        self.assertEqual((result["selector_rounds"], result["selector_completions"]), (1, 760))
        self.assertEqual((result["training_rollout_tokens"], result["generated_tokens"]), (100, 1100))
        self.assertEqual(result["generated_tokens_per_wall_second"], 550)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy not installed")
    def test_gradalign_curriculum_preserves_lazy_cycles_without_artifacts(self):
        import numpy as np
        from my_example.alignment_baselines.curriculum import GradAlignCurriculum

        b = bench.Benchmark("gradalign", "timing", "off", 10, 10, self.path, "v5p-16")
        calls = []

        def estimate(examples, *, num_rollouts, apply_mismatch):
            calls.append((b.active_iteration, len(examples), apply_mismatch))
            count = len(examples) * num_rollouts
            b.note_selector_rollout(count, count, candidate=apply_mismatch)
            return SimpleNamespace(features=np.ones((len(examples), 2)))

        prompts = [{"prompts": str(i)} for i in range(190)]
        curriculum = GradAlignCurriculum(
            training_examples=prompts[:160], validation_examples=prompts[160:],
            train_batch_size=4, total_update_steps=20, estimator=SimpleNamespace(estimate=estimate),
            writer=None, rl_cluster=SimpleNamespace(_training_benchmark=b), candidate_rollouts=4,
            validation_rollouts=4, selection_ratio=4, selection_interval=10,
        )
        b.metadata, b.learner_entry = {}, time.perf_counter()
        b.actor = SimpleNamespace(train_steps=0)
        b.cluster = SimpleNamespace(global_steps=0)
        b._drain = Mock()
        b._optimizer_steps = lambda: b.actor.train_steps
        batches = iter(curriculum)
        self.assertEqual(calls, [])
        for i in range(1, 21):
            b.before_iteration()
            b.note_prompts(next(batches)["prompts"])
            b.note_rollout(16, 100)
            b.actor.train_steps = b.cluster.global_steps = i
            b.after_iteration()
        self.assertEqual(calls, [(1, 30, False), (1, 160, True), (11, 30, False), (11, 160, True)])
        self.assertEqual(bench.summarize(b.rows[10:], b.window_seconds)["selector_rounds"], 1)
        self.assertEqual(list(self.path.iterdir()), [])

    def test_launch_routes_and_budgets(self):
        launcher = Path(bench.__file__).with_name("run_grpo_benchmark.sh")
        markers = {"pods": "pods_keep=2", "grpo-ds": "ds=1", "norm-top50": "actor_variant=gradient_norm_top50",
                   "pcgrad": "conflict_method=pcgrad", "cagrad": "conflict_method=cagrad",
                   "gradalign": "-m my_example.alignment_main"}
        for method, marker in markers.items():
            for mode in ("timing", "memory"):
                warmup, steps = ((20, 50) if mode == "timing" else (0, 1)) if method == "gradalign" else (
                    (15, 50) if mode == "timing" else (1, 2))
                cmd = ["bash", str(launcher), method, mode, str(self.path / "never-create"),
                       str(warmup), str(steps), "off", "--dry-run"]
                with self.subTest(method=method, mode=mode):
                    plan = subprocess.check_output(cmd, text=True)
                    self.assertIn(marker, plan)
                    self.assertIn("--skip-eval-before --skip-eval-after", plan)
        self.assertFalse((self.path / "never-create").exists())

    def test_disk_guard_rejects_small_volume(self):
        with patch.object(bench.shutil, "disk_usage", return_value=SimpleNamespace(free=5 * 2**30)):
            with self.assertRaisesRegex(ValueError, "need 6 GiB"):
                bench.check_disk(self.path, 6)

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
            trace = self.path / "trace/plugins/profile/test/test.xplane.pb"
            trace.parent.mkdir(parents=True)
            trace.touch()
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

    def test_observed_xprof_extracts_stack_heap_without_certifying_residency(self):
        self._fixture("memory")
        profiles = {str(i): {"profileSummary": {"memoryCapacity": "1000", "peakStats": {
            "peakBytesInUse": str(100 + i), "stackReservedBytes": "90",
            "heapAllocatedBytes": str(10 + i)}}} for i in range(4)}
        convert = SimpleNamespace(raw_to_tool_data=SimpleNamespace(xspace_to_tool_data=lambda *_: (
            json.dumps({"memoryProfilePerAllocator": profiles}), "json")))
        with patch.dict(sys.modules, {"xprof.convert": convert}):
            observed = bench.observed_xprof_peaks(self.path)
            self.assertEqual(observed["devices"]["3"]["peak_hbm_bytes"], 103)
            self.assertFalse(observed["resident_allocations_verified"])
            del profiles["3"]
            with self.assertRaisesRegex(ValueError, "Incomplete XProf coverage"):
                bench.observed_xprof_peaks(self.path)

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
