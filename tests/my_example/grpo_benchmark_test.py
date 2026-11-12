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
from contextlib import nullcontext
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
                   "gradalign": "-m my_example.alignment_main", "learnalign": "--learnalign-warmup-prompts 40",
                   "random10": "fixed_filter=random scope=group ratio=0.10",
                   "reward10": "fixed_filter=reward scope=group ratio=0.10"}
        for method, marker in markers.items():
            for mode in ("timing", "memory"):
                warmup, steps = (15, 50) if mode == "timing" else (1, 2)
                if method == "gradalign":
                    warmup, steps = (20, 50) if mode == "timing" else (0, 1)
                elif method == "learnalign" and mode == "timing":
                    warmup, steps = (10, 50)
                cmd = ["bash", str(launcher), method, mode, str(self.path / "never-create"),
                       str(warmup), str(steps), "off", "--dry-run"]
                with self.subTest(method=method, mode=mode):
                    plan = subprocess.check_output(cmd, text=True)
                    self.assertIn(marker, plan)
                    self.assertIn("--skip-eval-before --skip-eval-after", plan)
        base = ["bash", str(launcher), "learnalign"]
        out = str(self.path / "never-create")
        plan = subprocess.check_output(
            [*base, "memory", out, "10", "2", "off", "--dry-run"], text=True,
        )
        self.assertIn("warmup=10 steps=2", plan)
        rejected = subprocess.run(
            [*base, "timing", out, "75", "50", "off", "--dry-run"], check=False,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertFalse((self.path / "never-create").exists())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy not installed")
    def test_learnalign_keeps_full_selector_outside_training_windows(self):
        import numpy as np
        from my_example.alignment_baselines.curriculum import LearnAlignCurriculum

        for mode, warmup, steps in (("timing", 10, 50), ("memory", 1, 2), ("memory", 10, 2)):
            with self.subTest(mode=mode, warmup=warmup, steps=steps):
                out = self.path / f"{mode}-{warmup}"
                out.mkdir()
                b = bench.Benchmark("learnalign", mode, "off", warmup, steps, out, "v5p-16")
                b.metadata = {"schedule_optimizer_steps": 691, "actor_trainer": "Trainer", "valuation": True}
                b.batching, b.learner_entry = {}, 0.0
                b.actor = SimpleNamespace(train_steps=0)
                b.cluster = SimpleNamespace(global_steps=0, _training_benchmark=b)
                b._optimizer_steps = lambda: b.actor.train_steps
                b._drain = Mock()
                b._memory_snapshot = Mock(return_value={str(i): {"stats": None} for i in range(4)})
                b._start_trace = Mock()
                selector_calls = []
                clock = [0.0]

                def estimate(examples, *, num_rollouts, apply_mismatch):
                    self.assertEqual(b.active_iteration, 11)
                    b._start_trace.assert_not_called()
                    self.assertEqual((len(examples), num_rollouts), (2764, 8))
                    selector_calls.append(True)
                    clock[0] += 120
                    b.compiler_events.append({"iteration": 11, "phase": "learnalign_selector",
                                              "xla_compile_completed_log": True})
                    b.note_selector_rollout(len(examples) * num_rollouts, 40000, candidate=apply_mismatch)
                    return SimpleNamespace(features=np.ones((len(examples), 2)),
                                           selector_rewards=np.tile([0, 1] * 4, (len(examples), 1)))

                data = LearnAlignCurriculum(
                    training_examples=[{"prompts": str(i)} for i in range(2764)],
                    train_batch_size=4, total_update_steps=691, estimator=SimpleNamespace(estimate=estimate),
                    writer=None, rl_cluster=b.cluster, warmup_prompts=40, estimation_rollouts=8,
                    selection_ratio=2, warmup_counts_toward_budget=True,
                )
                batches = iter(data)
                with patch.object(bench.time, "perf_counter", side_effect=lambda: clock[0]):
                    for i in range(1, warmup + steps + 1):
                        b.before_iteration()
                        b.note_prompts(next(batches)["prompts"])
                        if mode == "memory" and i == warmup + 1:
                            b._start_trace.assert_called_once()
                        b.note_rollout(16, 100)
                        clock[0] += 2
                        b.actor.train_steps = b.cluster.global_steps = i
                        self.assertEqual(b.after_iteration(), i == warmup + steps)
                self.assertEqual(selector_calls, [] if warmup == 1 else [True])
                self.assertEqual(b.window_seconds, 2 * steps)
                self.assertEqual(b.rows[warmup]["seconds"], 2)
                b.teardown_seconds = 0.0
                if mode == "memory":
                    (out / "trace").mkdir()
                    (out / "trace/test.xplane.pb").touch()
                b._write_reports()
                (out / "exit_code").write_text("0\n")
                report, _, _ = bench.check_run(out)
                self.assertEqual(report["measurement"]["generated_tokens"], 100 * steps)
                self.assertEqual(report["measured_xla_compile_steps"], [])
                if mode == "timing":
                    self.assertEqual(report["learnalign_prelearn"]["mean_seconds_per_iteration"], 2)
                if warmup == 10:
                    self.assertEqual(report["learnalign_selector"]["elapsed_seconds"], 120)
                    self.assertEqual(report["learnalign_selector"]["generated_tokens"], 40000)

    def test_fixed_filter_count_is_materialized_after_window(self):
        b = self._fixture()
        b.method = "random10"
        b.metadata.update(actor_trainer="FixedFilterTrainer", valuation=True)
        device_get = Mock(side_effect=lambda values: values)
        b.jax = SimpleNamespace(device_get=device_get)
        for i, count in enumerate((0, 1, 3), 1):
            b.active_iteration = i
            b.note_fixed_filter(count)
        device_get.assert_not_called()
        b._write_reports()
        device_get.assert_called_once_with((0, 1, 3))
        report, _, _ = bench.check_run(self.path)
        self.assertEqual(report["measurement"]["filtered_completions"], 4)
        self.assertEqual(report["measurement"]["actual_filtered_fraction"], 4 / 32)

    def test_learnalign_hbm_opt_in_requires_postselect_memory(self):
        env = {
            "TUNIX_BENCH_METHOD": "learnalign", "TUNIX_GRPO_BENCHMARK": "memory",
            "TUNIX_BENCH_DIAGNOSTICS": "off", "TUNIX_BENCH_WARMUP": "10",
            "TUNIX_BENCH_STEPS": "2", "TUNIX_BENCH_OUT": str(self.path),
            "TUNIX_BENCH_RESOURCE": "v5p-16", bench.LEARNALIGN_XPROF_ENV: sys.executable,
        }
        with patch.dict(bench.os.environ, env, clear=True):
            self.assertEqual(bench.Benchmark.from_env().learnalign_xprof_python, Path(sys.executable))
            bench.os.environ["TUNIX_BENCH_WARMUP"] = "1"
            with self.assertRaisesRegex(ValueError, "requires LearnAlign postselect memory mode"):
                bench.Benchmark.from_env()

    def _learnalign_hbm_fixture(self):
        b = bench.Benchmark("learnalign", "memory", "off", 10, 2, self.path, "v5p-16")
        b.learnalign_xprof_python = Path(sys.executable)
        b.active_iteration = 11
        b.selector_rounds = b.selector_completions = 0
        b.selector_reference_tokens = b.selector_candidate_tokens = 0
        b.selector_seconds = 0.0
        b.metadata = {"commit_sha": "test"}
        b._drain = Mock()
        b._start_trace = Mock(side_effect=lambda _: setattr(b, "trace_active", True))
        b._stop_trace = Mock(side_effect=lambda: setattr(b, "trace_active", False))
        b.phase = lambda _: nullcontext()
        b._open_measurement_window = Mock()
        return b

    def test_learnalign_hbm_samples_full_pool_without_overlapping_postselect(self):
        b = self._learnalign_hbm_fixture()
        b._convert_xprof_trace = Mock(side_effect=[
            {"max_device_peak_hbm_gib": value, "devices": {str(i): {} for i in range(4)}}
            for value in (12, 14, 13)
        ])
        seen = []
        with b.selector_round():
            for index in range(1, 692):
                with b.learnalign_selector_chunk(index=index, total=691, prompt_count=4, num_rollouts=8):
                    b.note_selector_rollout(32, 100, candidate=True)
                    if b.trace_active:
                        seen.append(index)
                self.assertFalse(b.trace_active)
            b._open_measurement_window.assert_not_called()
        self.assertEqual(seen, [1, 346, 691])
        self.assertEqual(b._convert_xprof_trace.call_count, 3)
        b._open_measurement_window.assert_called_once()
        self.assertEqual(b.learnalign_selection["completions"], 22112)
        self.assertEqual(b.selector_completions, 0)
        report = json.loads((self.path / "selector_hbm.json").read_text())
        self.assertEqual(report["sampled_chunks"], seen)
        self.assertEqual(report["max_device_peak_hbm_gib"], 14)
        self.assertEqual(sum(s["completions"] for s in report["samples"]), 96)
        self.assertFalse(report["resident_allocations_verified"])

    def test_learnalign_hbm_invalid_trace_stops_before_postselect(self):
        b = self._learnalign_hbm_fixture()
        b._convert_xprof_trace = Mock(side_effect=subprocess.CalledProcessError(1, "xprof"))
        with self.assertRaisesRegex(RuntimeError, "failed four-device extraction"):
            with b.selector_round(), b.learnalign_selector_chunk(
                index=1, total=691, prompt_count=4, num_rollouts=8,
            ):
                b.note_selector_rollout(32, 100, candidate=True)
        self.assertFalse(b.trace_active)
        b._open_measurement_window.assert_not_called()
        self.assertFalse((self.path / "selector_hbm.json").exists())

    def test_disk_guard_rejects_small_volume(self):
        with patch.object(bench.shutil, "disk_usage", return_value=SimpleNamespace(free=5 * 2**30)):
            with self.assertRaisesRegex(ValueError, "need 6 GiB"):
                bench.check_disk(self.path, 6)

    def test_ds_diagnostic_is_opt_in_and_memory_only(self):
        env = {
            "TUNIX_BENCH_METHOD": "grpo-ds", "TUNIX_GRPO_BENCHMARK": "memory",
            "TUNIX_BENCH_DIAGNOSTICS": "off", "TUNIX_BENCH_WARMUP": "1",
            "TUNIX_BENCH_STEPS": "2", "TUNIX_BENCH_OUT": str(self.path),
            "TUNIX_BENCH_RESOURCE": "v5p-16",
        }
        with patch.dict(bench.os.environ, env, clear=True):
            self.assertIsNone(bench.Benchmark.from_env().ds_xprof_python)
            bench.os.environ[bench.DS_XPROF_ENV] = sys.executable
            self.assertEqual(bench.Benchmark.from_env().ds_xprof_python, Path(sys.executable))
            bench.os.environ["TUNIX_GRPO_BENCHMARK"] = "timing"
            with self.assertRaisesRegex(ValueError, r"requires GRPO\+DS memory mode"):
                bench.Benchmark.from_env()

    def test_ds_profiler_options_do_not_read_legacy_getter(self):
        class Options:
            @property
            def advanced_configuration(self):
                raise TypeError("cannot read protobuf map")

            @advanced_configuration.setter
            def advanced_configuration(self, config):
                self.config = config

        b = self.bench
        b.ds_xprof_python = Path(sys.executable)
        b.devices = tuple(range(4))
        profiler = SimpleNamespace(ProfileOptions=Options, start_trace=Mock(), stop_trace=Mock())
        b.jax = SimpleNamespace(profiler=profiler)
        with patch.object(bench, "check_disk"):
            b._start_trace()
            b._stop_trace()
        kwargs = profiler.start_trace.call_args.kwargs
        self.assertEqual(kwargs["profiler_options"].config, {
            "tpu_trace_mode": "TRACE_ONLY_XLA", "tpu_num_chips_to_profile_per_task": 4,
        })
        self.assertNotIn("device_tracer_level", b.metadata["ds_profiler_requested_options"])
        profiler.stop_trace.assert_called_once()
        self.assertFalse(b.trace_active)

    def test_ds_preflight_failure_prevents_training(self):
        b = self.bench
        b.ds_xprof_python = Path(sys.executable)
        learner = Mock()
        with (
            patch.object(b, "_run_ds_profiler_preflight", side_effect=RuntimeError("no TPU peaks")),
            self.assertRaisesRegex(RuntimeError, "no TPU peaks"),
        ):
            b._run_with_compile_log(learner, None, None)
        learner.train.assert_not_called()

    def test_ds_memory_window_covers_refills_and_actor_updates(self):
        b = bench.Benchmark("grpo-ds", "memory", "off", 1, 2, self.path, "v5p-16")
        b.metadata, b.learner_entry = {}, time.perf_counter()
        b.actor = SimpleNamespace(train_steps=0)
        b.cluster = SimpleNamespace(global_steps=0)
        b._optimizer_steps = lambda: b.actor.train_steps
        b._drain = Mock()
        b._memory_snapshot = Mock()
        b._start_trace = Mock(side_effect=lambda: setattr(b, "trace_active", True))
        b._stop_trace = Mock(side_effect=lambda: setattr(b, "trace_active", False))
        b.jax = SimpleNamespace(profiler=SimpleNamespace(TraceAnnotation=lambda *a, **k: nullcontext()))
        for i in (1, 2, 3):
            b.before_iteration()
            for prompts in (["a", "b", "c", "d"], ["e", "f", "g", "h"]):
                self.assertEqual(b.trace_active, i >= 2)
                b.note_rollout(16, 100)
                b.note_ds_candidate(prompts, eligible=2, selected=2)
            self.assertEqual(b.trace_active, i >= 2)
            b.actor.train_steps = b.cluster.global_steps = i
            self.assertEqual(b.after_iteration(), i == 3)
        b._start_trace.assert_called_once()
        b._stop_trace.assert_called_once()
        self.assertFalse(b.trace_active)
        self.assertEqual(b.end_optimizer_step - b.start_optimizer_step, 2)
        self.assertEqual(bench.summarize(b.rows[1:], b.window_seconds)["candidate_batches"], 4)

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
            probe = bench.xprof_window(self.path / "trace", [0, 1, 2, 3])
            self.assertEqual(probe["devices"], observed["devices"])
            self.assertNotIn("iterations_inclusive", probe)
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
