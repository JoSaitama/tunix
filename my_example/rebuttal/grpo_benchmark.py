"""Opt-in clean GSM8K benchmark attached to the production learner.

This module imports no JAX at import time; report checks run on CPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import dataclasses
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time

METHODS = {
    "vanilla": "Trainer",
    "dtv": "PolicySelfInfTrainer",
    "dtv-loo": "PolicySelfInfLooTrainer",
    "pods": "Trainer",
    "grpo-ds": "Trainer",
    "norm-top50": "GradientNormTop50Trainer",
    "pcgrad": "GradientConflictTrainer",
    "cagrad": "GradientConflictTrainer",
    "gradalign": "Trainer",
    "learnalign": "Trainer",
    "random10": "FixedFilterTrainer",
    "reward10": "FixedFilterTrainer",
}
GRADALIGN_INTERVAL = 10
LEARNALIGN_PRE_UPDATES = 10
LEARNALIGN_PRE_SKIP = 5
LEARNALIGN_SELECTOR_COMPLETIONS = 691 * 4 * 8
FIXED_FILTERS = {"random10": "random", "reward10": "reward"}
ALIGNMENT_METHODS = {"gradalign", "learnalign"}
VALUATION_ACTORS = {"dtv", "dtv-loo", "norm-top50", "pcgrad", "cagrad", *FIXED_FILTERS}
STEP_FIELDS = (
    "iteration", "window", "seconds", "prompts", "completions",
    "generated_tokens", "update_completions", "candidate_batches",
    "ds_zero_variance_groups", "ds_excess_eligible_groups", "selector_rounds",
    "selector_seconds", "selector_completions", "selector_tokens",
    "selector_reference_tokens", "selector_candidate_tokens",
    "filtered_completions", "prompt_order_sha256", "compile_event",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash_parts(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def summarize(rows, elapsed):
    _require(bool(rows) and elapsed > 0, "Empty measurement window")
    tokens = sum(r["generated_tokens"] for r in rows)
    selector_tokens = sum(r.get("selector_tokens", 0) for r in rows)
    result = {
        "iterations": len(rows),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_iteration": elapsed / len(rows),
        "median_iteration_boundary_seconds": statistics.median(r["seconds"] for r in rows),
        "generated_tokens": tokens + selector_tokens,
        "generated_tokens_per_wall_second": (tokens + selector_tokens) / elapsed,
        "training_rollout_tokens": tokens,
        "selector_tokens": selector_tokens,
        "all_generated_tokens": tokens + selector_tokens,
        "training_rollout_tokens_per_wall_second": tokens / elapsed,
    }
    for key in ("completions", "update_completions", "candidate_batches",
                "ds_zero_variance_groups", "ds_excess_eligible_groups", "selector_rounds",
                "selector_seconds", "selector_completions", "selector_reference_tokens",
                "selector_candidate_tokens"):
        result[key] = sum(r.get(key, 0) for r in rows)
    if all("filtered_completions" in row for row in rows):
        result["filtered_completions"] = sum(row["filtered_completions"] for row in rows)
        result["actual_filtered_fraction"] = result["filtered_completions"] / result["completions"]
    return result


def _train_compile_steps(events, method):
    return {
        e["iteration"] for e in events
        if e["xla_compile_completed_log"] and e["iteration"] is not None
        and not (method == "learnalign" and e.get("phase") == "learnalign_selector")
    }


class _CompileLog(logging.Handler):
    def __init__(self, benchmark):
        super().__init__()
        self.benchmark = benchmark

    def emit(self, record):
        message = record.getMessage()
        lower = message.lower()
        if any(x in lower for x in (
            "compiling ", "compilation", "finished tracing", "cache hit", "cache miss",
        )):
            b = self.benchmark
            b.compiler_events.append({
                "iteration": b.active_iteration,
                "phase": getattr(b.thread, "phase", None),
                "elapsed_since_main_seconds": time.perf_counter() - b.main_start,
                "logger": record.name,
                "message": message,
                # A compile request or tracing message alone is not proof of XLA execution.
                "xla_compile_completed_log": "finished xla compilation" in lower,
            })
        else:
            logging.getLogger().handle(record)


class Benchmark:
    def __init__(self, method, mode, diagnostics, warmup, steps, out, resource):
        _require(method in METHODS, f"Unsupported method: {method}")
        _require(mode in ("timing", "memory"), f"Unsupported mode: {mode}")
        _require(diagnostics in ("on", "off"), "Diagnostics must be on or off")
        _require((warmup >= 1 or (method == "gradalign" and mode == "memory" and warmup == 0))
                 and steps >= 1, "Invalid warmup/measurement step count")
        _require(mode != "memory" or diagnostics == "off", "Memory profile requires diagnostics off")
        _require(diagnostics == "off" or method in {"vanilla", "dtv", "dtv-loo"},
                 "New baseline benchmarks require diagnostics off")
        if method == "gradalign":
            if mode == "timing":
                _require(warmup % GRADALIGN_INTERVAL == 0 and steps % GRADALIGN_INTERVAL == 0,
                         "GradAlign timing requires complete ten-update cycles")
            else:
                _require((warmup, steps) == (0, 1),
                         "GradAlign memory captures the first selector and update (0 + 1)")
        self.memory_phase = None
        if method == "learnalign":
            if mode == "timing":
                _require((warmup, steps) == (LEARNALIGN_PRE_UPDATES, 50),
                         "LearnAlign timing uses 10 + 50 updates")
            else:
                _require((warmup, steps) in {(1, 2), (LEARNALIGN_PRE_UPDATES, 2)},
                         "LearnAlign memory uses 1 + 2 (prelearn) or 10 + 2 (postselect)")
                self.memory_phase = "prelearn" if warmup == 1 else "postselect"
        self.method, self.mode, self.diagnostics = method, mode, diagnostics
        self.quiet = diagnostics == "off"
        self.warmup, self.steps = warmup, steps
        self.out, self.resource = Path(out).resolve(), resource
        self.main_start = time.perf_counter()
        self.active_iteration = None
        self.rows = []
        self.compiler_events = []
        self.thread = threading.local()
        self.trace_active = False
        self.memory = {}
        self.learnalign_selection = None
        self.filter_counts = []
        # The ordinary learner peeks at item 1 to infer its batch size. That
        # would execute GradAlign's first selector before timing/tracing starts.
        self.first_batch_size = 4 if method == "gradalign" else None

    @classmethod
    def from_env(cls):
        return cls(
            os.environ["TUNIX_BENCH_METHOD"], os.environ["TUNIX_GRPO_BENCHMARK"],
            os.environ["TUNIX_BENCH_DIAGNOSTICS"], int(os.environ["TUNIX_BENCH_WARMUP"]),
            int(os.environ["TUNIX_BENCH_STEPS"]), os.environ["TUNIX_BENCH_OUT"],
            os.environ["TUNIX_BENCH_RESOURCE"],
        )

    def check_config(self, cfg, max_steps):
        _require(os.environ.get("TUNIX_EXPERIMENT_SEED") == "13", "Expected seed 13")
        _require(float(os.environ.get("TUNIX_REWARD_RANK_NOISE_FRACTION", "0")) == 0,
                 "Benchmark requires clean rewards")
        _require(os.environ.get("TUNIX_REWARD_MODE") == "accuracy", "Expected formal accuracy reward")
        _require(not any((
            cfg.runtime.use_wandb, cfg.runtime.eval_before_train, cfg.runtime.eval_after_train,
        )), "WandB and pre/post evaluation must be disabled")
        _require((max_steps, cfg.data.train_micro_batch_size, cfg.grpo.num_generations,
                  cfg.grpo.num_iterations) == (691, 4, 4, 1), "Expected the formal 691-step 4x4 schedule")
        _require(self.warmup + self.steps <= max_steps, "Requested window exceeds training budget")
        _require(cfg.model.model_id == "google/gemma-3-1b-it", "Unexpected checkpoint family")
        _require((cfg.grpo.max_prompt_length, cfg.grpo.total_generation_steps) == (256, 768),
                 "Expected the formal GSM8K generation shape")
        _require((cfg.data.source, cfg.data.max_train_examples, cfg.data.train_fraction,
                  cfg.data.num_epochs, cfg.lora.rank, cfg.lora.alpha) ==
                 ("tfds", 3072, 0.9, 1, 64, 64), "Unexpected data or LoRA configuration")
        _require((cfg.training.learning_rate, cfg.training.warmup_fraction, cfg.grpo.beta) ==
                 (1e-6, 0.1, 0.08), "Unexpected LR schedule or KL coefficient")
        _require(not any(os.environ.get(x) for x in (
            "TUNIX_ENABLE_ACTOR_PHASE_TIMING", "TUNIX_ENABLE_ACTOR_HEARTBEAT",
            "TUNIX_ENABLE_ACTOR_PROGRESS_METRICS",
        )), "Disable legacy optional actor probes in the benchmark")
        _require(os.environ.get("TUNIX_DISABLE_TFLOPS_MEASUREMENT") == "1",
                 "Benchmark must disable optional TFLOPs compilation")

    def check_topology(self, jax, mesh):
        devices = tuple(mesh.devices.flat)
        _require(
            (jax.process_count(), jax.process_index(), jax.device_count(),
             jax.local_device_count(), tuple(mesh.devices.shape)) == (1, 0, 4, 4, (4, 1))
            and len({d.id for d in devices}) == 4
            and all(d.platform == "tpu" and d.process_index == 0 for d in devices),
            "Expected one logical process with four local TPU devices in a (4,1) mesh",
        )
        self.devices = devices
        self.topology = {
            "declared_tpu_resource": self.resource,
            "declared_physical_worker": os.environ.get("TUNIX_BENCH_PHYSICAL_WORKER"),
            "physical_hostname": os.uname().nodename,
            "participating_workers": 1,
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
            "global_device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "mesh_shape": list(mesh.devices.shape),
            "mesh_axis_names": list(mesh.axis_names),
            "devices": [{"id": d.id, "kind": d.device_kind, "device": str(d)} for d in devices],
        }

    def configure_loop(self, learner, full, mini, micro, service, accumulation):
        _require((full, mini, micro, service, accumulation) == (4, 4, 4, 4, 1),
                 "Unexpected learner batching")
        _require((learner._rollout_micro_batch_size, learner._compute_logps_micro_batch_size)
                 == (4, 4), "Unexpected service batching")
        _require(not learner.can_enable_async_rollout, "Expected the formal colocated mesh")
        self.batching = {
            "full": full, "mini": mini, "micro": micro, "service": service,
            "grad_accumulation": accumulation, "rollout_micro": learner._rollout_micro_batch_size,
            "logps_micro": learner._compute_logps_micro_batch_size,
            "async_rollout": learner.can_enable_async_rollout,
            "sync_weights": learner.should_sync_weights,
        }

    def _drain(self):
        # Wait on existing arrays/effects; do not dispatch a new device kernel.
        states = (
            self.nnx.state(self.actor.model, self.nnx.LoRAParam),
            self.nnx.state(self.actor.optimizer),
            self.nnx.state(self.cluster.rollout.model()),
        )
        self.jax.block_until_ready(states)
        self.jax.effects_barrier()

    def _optimizer_steps(self):
        optimizer = self.actor.optimizer
        _require(hasattr(optimizer.opt_state, "gradient_step"),
                 "Expected Optax MultiSteps; cannot verify real optimizer updates")

        def scalar(value):
            raw = value.get_value() if hasattr(value, "get_value") else getattr(value, "value", value)
            return int(self.jax.device_get(raw))

        applied = scalar(optimizer.opt_state.gradient_step)
        _require(applied == scalar(optimizer.step), "Optax and NNX optimizer counters disagree")
        return applied

    def _memory_snapshot(self):
        result = {}
        for d in self.devices:
            try:
                stats = d.memory_stats()
                result[str(d.id)] = {
                    "stats": None if stats is None else {
                        k: None if v is None else int(v) for k, v in stats.items()
                    },
                }
            except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
                result[str(d.id)] = {"stats": None, "error": str(exc)}
        return result

    def _start_trace(self):
        check_disk(self.out, float(os.environ.get("TUNIX_BENCH_MIN_FREE_GIB", "6")))
        options_type = getattr(self.jax.profiler, "ProfileOptions", None)
        options = options_type() if options_type else None
        kwargs = {}
        if options is not None:
            if hasattr(options, "tpu_num_chips_to_profile_per_task"):
                options.tpu_num_chips_to_profile_per_task = len(self.devices)
            options.host_tracer_level = 2
            options.python_tracer_level = 0
            kwargs["profiler_options"] = options
        self.jax.profiler.start_trace(str(self.out / "trace"), **kwargs)
        self.trace_active = True

    def _stop_trace(self):
        if self.trace_active:
            self.trace_active = False
            self.jax.profiler.stop_trace()

    def before_iteration(self):
        step = len(self.rows) + 1
        if step == 1:
            self.previous_boundary = self.warmup_start = time.perf_counter()
            self.metadata["learner_setup_seconds"] = self.warmup_start - self.learner_entry
        if self.method == "learnalign" and self.mode == "timing" and step == LEARNALIGN_PRE_SKIP + 1:
            self._drain()
            self.previous_boundary = time.perf_counter()
        if step == self.warmup + 1:
            self._drain()
            warmup_end = time.perf_counter()
            self.warmup_seconds = warmup_end - self.warmup_start
            self.start_optimizer_step = self._optimizer_steps()
            self.start_actor_step = self.actor.train_steps
            _require(self.start_optimizer_step == self.start_actor_step == self.warmup,
                     "Warmup actor and optimizer counters disagree")
            self.warmup_end = warmup_end
            if self.method != "learnalign" or self.memory_phase == "prelearn":
                self._open_measurement_window()
        self.active_iteration = step
        self.prompt_hash = None
        self.prompt_count = 0
        self.rollouts = 0
        self.tokens = 0
        self.ds_batches = self.ds_zero_variance = self.ds_excess_eligible = 0
        self.selector_rounds = self.selector_completions = 0
        self.selector_reference_tokens = self.selector_candidate_tokens = 0
        self.selector_seconds = 0.0

    def _open_measurement_window(self):
        if self.mode == "memory":
            self.memory["window_start"] = self._memory_snapshot()
            self._start_trace()
        self.previous_boundary = self.window_start = time.perf_counter()
        setup = self.window_start - self.warmup_end
        if self.learnalign_selection is not None:
            setup -= self.learnalign_selection["elapsed_seconds"]
        self.metadata["post_warmup_setup_seconds"] = setup

    def note_prompts(self, prompts):
        _require(self.prompt_hash is None, "Expected one input batch per iteration")
        self.prompt_count = len(prompts)
        self.prompt_hash = _hash_parts(str(p).encode("utf-8") for p in prompts)

    def note_rollout(self, count, tokens):
        self.rollouts += count
        self.tokens += tokens

    def note_fixed_filter(self, count):
        _require(self.method in FIXED_FILTERS and self.active_iteration is not None,
                 "Unexpected fixed-filter count")
        # Keep the scalar device result, without forcing a host sync per update.
        self.filter_counts.append((self.active_iteration, count))

    def check_actor_input(self, examples):
        count = sum(example.completion_ids.shape[0] for example in examples)
        expected = 8 if self.method == "pods" else 16
        _require(count == expected, f"{self.method}: expected {expected} actor rows, got {count}")

    def note_ds_candidate(self, prompts, eligible, selected):
        _require(self.method == "grpo-ds" and len(prompts) == 4
                 and 0 <= selected <= eligible <= 4, "Unexpected DS candidate batch")
        if self.ds_batches == 0:
            self.note_prompts(prompts)
        self.ds_batches += 1
        self.ds_zero_variance += 4 - eligible
        self.ds_excess_eligible += eligible - selected

    @contextmanager
    def selector_round(self):
        _require(self.method in ALIGNMENT_METHODS and self.active_iteration is not None,
                 "Selector ran outside a measured or warmup iteration")
        is_learnalign = self.method == "learnalign"
        if is_learnalign:
            _require(self.active_iteration == LEARNALIGN_PRE_UPDATES + 1
                     and self.learnalign_selection is None and self.memory_phase != "prelearn"
                     and not self.trace_active, "Unexpected LearnAlign selector boundary")
            print("LearnAlign benchmark: starting the full selector after update 10.", flush=True)
        start = time.perf_counter()
        with self.phase(f"{self.method}_selector"):
            yield
        self.selector_seconds += time.perf_counter() - start
        self.selector_rounds += 1
        if is_learnalign:
            _require(self.selector_completions == LEARNALIGN_SELECTOR_COMPLETIONS
                     and self.selector_reference_tokens == 0 and self.selector_candidate_tokens > 0,
                     "Expected a complete LearnAlign full-pool selector")
            self.learnalign_selection = {
                "after_optimizer_updates": LEARNALIGN_PRE_UPDATES,
                "elapsed_seconds": self.selector_seconds,
                "completions": self.selector_completions,
                "generated_tokens": self.selector_candidate_tokens,
            }
            # The standalone selector is not part of step 11's timing or tokens.
            self.selector_rounds = self.selector_completions = self.selector_candidate_tokens = 0
            self.selector_seconds = 0.0
            print("LearnAlign benchmark: full selector complete; starting post-select window.", flush=True)
            self._open_measurement_window()

    def note_selector_rollout(self, count, tokens, *, candidate):
        _require(self.method in ALIGNMENT_METHODS and self.active_iteration is not None,
                 "Selector rollout ran outside an iteration")
        self.selector_completions += count
        if candidate:
            self.selector_candidate_tokens += tokens
        else:
            self.selector_reference_tokens += tokens

    @contextmanager
    def phase(self, label):
        prior = getattr(self.thread, "phase", None)
        self.thread.phase = label
        try:
            if self.trace_active:
                with self.jax.profiler.TraceAnnotation(
                    f"benchmark/{label}", iteration=self.active_iteration
                ):
                    yield
            else:
                yield
        finally:
            self.thread.phase = prior

    def after_iteration(self):
        step = self.active_iteration
        candidate_batches = self.ds_batches if self.method == "grpo-ds" else 1
        _require(self.prompt_count == 4 and candidate_batches > 0
                 and self.rollouts == 16 * candidate_batches and self.tokens > 0,
                 f"Iteration {step} did not produce a full nonempty 4x4 rollout")
        if self.method == "gradalign":
            expected = int((step - 1) % GRADALIGN_INTERVAL == 0)
            # The last round can be partial at the unchanged 691-step budget.
            candidates = 16 * min(GRADALIGN_INTERVAL, 692 - step)
            _require(self.selector_rounds == expected
                     and self.selector_completions == expected * 4 * (30 + candidates),
                     "Unexpected GradAlign selector cycle or rollout count")
        if self.method == "learnalign" and step > LEARNALIGN_PRE_UPDATES:
            _require(self.learnalign_selection is not None, "Post-select update without full selector")
        _require(self.actor.train_steps == self.cluster.global_steps == step,
                 f"Iteration {step} actor/global counters disagree")
        final = step == self.warmup + self.steps
        if final:
            with self.phase("window_drain"):
                self._drain()
        elif self.method == "learnalign" and self.mode == "timing" and step == LEARNALIGN_PRE_UPDATES:
            self._drain()
        end = time.perf_counter()
        self.rows.append({
            "iteration": step, "window": "warmup" if step <= self.warmup else "measure",
            "seconds": end - self.previous_boundary, "prompts": self.prompt_count,
            "completions": self.rollouts, "generated_tokens": self.tokens,
            "update_completions": 8 if self.method == "pods" else 16,
            "candidate_batches": candidate_batches,
            "ds_zero_variance_groups": self.ds_zero_variance,
            "ds_excess_eligible_groups": self.ds_excess_eligible,
            "selector_rounds": self.selector_rounds, "selector_seconds": self.selector_seconds,
            "selector_completions": self.selector_completions,
            "selector_reference_tokens": self.selector_reference_tokens,
            "selector_candidate_tokens": self.selector_candidate_tokens,
            "selector_tokens": self.selector_reference_tokens + self.selector_candidate_tokens,
            "prompt_order_sha256": self.prompt_hash,
        })
        self.previous_boundary = end
        self.active_iteration = None
        if final:
            self.window_seconds = end - self.window_start
            if self.mode == "memory":
                self.memory["window_end"] = self._memory_snapshot()
                self._stop_trace()
            self.end_optimizer_step = self._optimizer_steps()
            _require(self.end_optimizer_step == self.actor.train_steps == step,
                     "Final actor and optimizer counters disagree")
        return final

    def run(self, jax, nnx, mesh, learner, dataset, cfg, model_path, eos_tokens, load_start,
            *, selector_config=None):
        import numpy as np

        self.jax, self.nnx = jax, nnx
        self.cluster = learner.rl_cluster
        self.actor = self.cluster.actor_trainer
        self.cluster._training_benchmark = self
        self.actor._benchmark_quiet = self.quiet
        actor_name = type(self.actor).__name__
        learner_name = {"pods": "PODSLearner", "grpo-ds": "DynamicSamplingLearner"}.get(
            self.method, "GRPOLearner"
        )
        _require(type(learner).__name__ == learner_name, f"Expected {learner_name}")
        _require(actor_name == METHODS[self.method], f"Unexpected actor trainer: {actor_name}")
        _require(self.actor.train_steps == 0, "A fresh run is required")
        _require(self.actor.config.checkpoint_root_directory is None, "Checkpointing must be disabled")
        if self.method in FIXED_FILTERS:
            _require((self.actor.method, self.actor.scope, self.actor.filter_ratio,
                      self.actor.num_generations, self.actor.decisions_path)
                     == (FIXED_FILTERS[self.method], "group", 0.1, 4, None),
                     "Expected group fixed filtering at 10% without decisions logs")
            self.actor._benchmark_aux_sink = self.note_fixed_filter
        if self.method in {"dtv", "dtv-loo"}:
            _require(self.actor.scope == "group" and self.actor._score_loss_fn is not None,
                     "Expected group-scoped policy valuation")
        if self.method == "norm-top50":
            _require(self.actor.num_generations == 4 and self.actor._score_loss_fn is not None
                     and self.actor.artifact_dir is None, "Unexpected norm top50 route")
        if self.method in {"pcgrad", "cagrad"}:
            _require(self.actor.method == self.method and not self.actor.include_kl
                     and self.actor.c == 0.5, "Unexpected gradient conflict route")
        if self.method == "pods":
            _require(learner.pods_keep_count == 2 and learner._pods_dir is None,
                     "Expected PODS keep-two without artifacts")
        if self.method == "grpo-ds":
            _require(learner._ds_cap == 20 and learner._ds_dir is None,
                     "Expected DS cap twenty without artifacts")
        _require((selector_config is not None) == (self.method in ALIGNMENT_METHODS),
                 "Unexpected selector configuration")
        if self.method == "dtv-loo":
            _require(self.actor.min_keep_fraction == 0.25, "Expected formal DTV-Loo retention floor")
            path = self.actor.decisions_path
            _require(not path if self.quiet else
                     bool(path) and Path(path).resolve() == self.out / "selection.jsonl",
                     "Unexpected DTV-Loo selection path")
        self._drain()
        load_seconds = time.perf_counter() - load_start
        if self.mode == "memory":
            self.memory["post_initialization"] = self._memory_snapshot()
        _require(self._optimizer_steps() == 0, "Optimizer was not freshly initialized")

        def adapter_parts():
            params = nnx.state(self.actor.model, nnx.LoRAParam)
            leaves, _ = jax.tree_util.tree_flatten_with_path(params)
            _require(bool(leaves), "Expected LoRA parameters")
            for path, value in leaves:
                array = np.asarray(jax.device_get(value))
                yield repr((path, array.shape, str(array.dtype))).encode()
                yield array.tobytes()

        config = dataclasses.asdict(cfg)
        for key in ("checkpoint_root_directory", "metrics_log_dir", "use_dynamic_batch_curation"):
            config["training"].pop(key, None)
        config["runtime"].pop("output_dir", None)
        self.metadata = {
            "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "versions": {p: _package_version(p) for p in (
                "google-tunix", "jax", "jaxlib", "flax", "optax", "google-metrax", "libtpu", "qwix",
            )},
            "topology": self.topology, "config": config, "schedule_optimizer_steps": 691,
            "seed": 13, "data_shuffle_seed": 55, "rollout_seed": 13, "eos_tokens": eos_tokens,
            "base_model_snapshot": str(model_path),
            "base_model_revision": Path(model_path).name,
            "initial_lora_sha256": _hash_parts(adapter_parts()),
            "actor_parameter_dtypes": sorted({
                str(x.dtype) for x in jax.tree.leaves(nnx.state(self.actor.model))
                if hasattr(x, "dtype")
            }),
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "jax_enable_x64": jax.config.jax_enable_x64,
            "jax_enable_compilation_cache": jax.config.jax_enable_compilation_cache,
            "actor_trainer": actor_name,
            "learner": learner_name,
            "selector_config": selector_config,
            "diagnostics": self.diagnostics,
            "valuation": self.method != "vanilla",
            "phases": {"actor_update": (
                "fused scoring/manipulation + update" if self.method in VALUATION_ACTORS
                else "optimizer and batch GRPO gradient"
            )},
        }
        self.metadata["main_setup_seconds"] = time.perf_counter() - self.main_start
        self.metadata["model_tokenizer_cluster_load_seconds"] = load_seconds
        self.metadata["launcher_to_main_seconds"] = (
            self.main_start - float(os.environ["TUNIX_BENCH_LAUNCH_MONOTONIC"])
            if "TUNIX_BENCH_LAUNCH_MONOTONIC" in os.environ else None
        )
        self._run_with_compile_log(learner, dataset, mesh)
        _require(len(self.rows) == self.warmup + self.steps, "Training ended before the requested window")
        self.teardown_seconds = time.perf_counter() - self.previous_boundary
        self._write_reports()

    def _run_with_compile_log(self, learner, dataset, mesh):
        logger = logging.getLogger("jax")
        handler = _CompileLog(self)
        old_propagate = logger.propagate
        old_compile = self.jax.config.jax_log_compiles
        logger.addHandler(handler)
        logger.propagate = False
        # Process-global so the rollout ThreadPoolExecutor also emits compile logs.
        self.jax.config.update("jax_log_compiles", True)
        try:
            with mesh:
                self.learner_entry = time.perf_counter()
                learner.train(dataset, None)
        finally:
            try:
                self._stop_trace()
            finally:
                self.jax.config.update("jax_log_compiles", old_compile)
                logger.propagate = old_propagate
                logger.removeHandler(handler)

    def _write_reports(self):
        if self.method in FIXED_FILTERS:
            _require([step for step, _ in self.filter_counts] == list(range(1, len(self.rows) + 1)),
                     "Expected one fixed-filter count per update")
            counts = self.jax.device_get(tuple(count for _, count in self.filter_counts))
            for row, count in zip(self.rows, counts):
                _require(float(count) in range(5), "Invalid group fixed-filter count")
                row["filtered_completions"] = int(count)
        compile_steps = _train_compile_steps(self.compiler_events, self.method)
        for row in self.rows:
            row["compile_event"] = row["iteration"] in compile_steps
        measured = self.rows[self.warmup:]
        sensitivity = [r for r in measured if not r["compile_event"]]
        elapsed = sum(r["seconds"] for r in sensitivity)
        stats = summarize(measured, self.window_seconds)
        stats["optimizer_updates"] = self.end_optimizer_step - self.start_optimizer_step
        if self.method == "gradalign":
            stats["window_seconds_outside_selector"] = self.window_seconds - stats["selector_seconds"]
        if self.method == "grpo-ds":
            stats["rejected_groups"] = stats["ds_zero_variance_groups"] + stats["ds_excess_eligible_groups"]
        with (self.out / "iterations.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, STEP_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        _dump(self.out / "compiler_events.json", self.compiler_events)
        _dump(self.out / "metadata.json", {**self.metadata, "batching": self.batching})
        if self.mode == "memory":
            memory = memory_report(self.memory, list(self.out.glob("trace/**/*.xplane.pb")))
            if self.memory_phase is not None:
                memory["memory_phase"] = self.memory_phase
            _dump(self.out / "memory.json", memory)
        report = {
            "method": self.method, "mode": self.mode, "diagnostics": self.diagnostics,
            "warmup_iterations": self.warmup, "warmup_seconds": self.warmup_seconds,
            "post_window_teardown_seconds": self.teardown_seconds,
            "measurement": stats, "actor_optimizer_start": self.start_optimizer_step,
            "actor_optimizer_end": self.end_optimizer_step,
            "measured_xla_compile_steps": sorted(compile_steps & {r["iteration"] for r in measured}),
            "compile_attribution": "JAX completion log inside outer iteration; unassigned events kept raw",
            "excluding_logged_compile_steps": (
                summarize(sensitivity, elapsed) if sensitivity and elapsed > 0 else None
            ),
            "token_definition": "non-padding completion tokens, excluding stop EOS; includes rejected DS rollouts and GradAlign selector tokens in all_generated_tokens",
            "timing_scope": "full outer iterations; only window boundaries explicitly drained",
            "reviewer_timing_eligible": self.mode == "timing" and self.quiet
            and self.steps == 50 and self.warmup >= 10,
        }
        if self.method == "learnalign":
            report["learnalign_selector"] = self.learnalign_selection
            if self.mode == "timing":
                pre = self.rows[LEARNALIGN_PRE_SKIP:LEARNALIGN_PRE_UPDATES]
                report["learnalign_prelearn"] = {
                    **summarize(pre, sum(row["seconds"] for row in pre)),
                    "iterations_inclusive": [LEARNALIGN_PRE_SKIP + 1, LEARNALIGN_PRE_UPDATES],
                    "xla_compile_steps": sorted(compile_steps & {r["iteration"] for r in pre}),
                }
                report["timing_scope"] = "post-select updates only; pre-learning and full selector reported separately"
        if self.memory_phase is not None:
            report["memory_phase"] = self.memory_phase
        _dump(self.out / "summary.json", report)


def memory_report(snapshots, traces):
    before, after = snapshots["window_start"], snapshots["window_end"]
    devices = {}
    for device, final in after.items():
        start_stats = before[device].get("stats") or {}
        end_stats = final.get("stats") or {}
        start_peak = start_stats.get("peak_bytes_in_use")
        end_peak = end_stats.get("peak_bytes_in_use")
        grew = start_peak is not None and end_peak is not None and end_peak > start_peak
        devices[device] = {
            "allocator_lifetime_peak_bytes": end_peak,
            "new_allocator_high_water_within_window_bytes": end_peak if grew else None,
            "verified_window_peak_hbm_bytes": None,
        }
    lifetime = [x["allocator_lifetime_peak_bytes"] for x in devices.values()]
    return {
        "collector": "JAX TPU device.memory_stats + JAX/XProf XPlane trace",
        "units": "bytes",
        "snapshots": snapshots,
        "devices": devices,
        "max_device_allocator_lifetime_peak_bytes": (
            max(lifetime) if lifetime and all(v is not None for v in lifetime) else None
        ),
        "verified_max_device_window_peak_hbm_bytes": None,
        "trace_files": [str(p) for p in sorted(traces)],
        "qualification": (
            "Counters are allocator-defined; XProf needs manual validation of all-device "
            "coverage, resident allocations and window peak. A flat or absent TPU trace is "
            "not accepted as total runtime HBM. Phase is unknown until validated."
        ),
    }


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_disk(path, minimum_gib):
    _require(0 < minimum_gib < float("inf"), "Disk threshold must be finite and positive")
    free = shutil.disk_usage(path).free / 2**30
    _require(free >= minimum_gib, f"{path}: {free:.2f} GiB free; need {minimum_gib:g} GiB before trace")
    return free


def check_run(path):
    """Offline smoke/report invariants. Never opens JAX or initializes TPU."""
    path = Path(path)
    _require((path / "exit_code").read_text().strip() == "0", f"{path}: launch failed")
    summary = _load(path / "summary.json")
    metadata = _load(path / "metadata.json")
    compiler = _load(path / "compiler_events.json")
    method = summary["method"]
    with (path / "iterations.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    warmup = summary["warmup_iterations"]
    measured = summary["measurement"]
    steps = measured["iterations"]
    _require(len(rows) == warmup + steps, f"{path}: incorrect row count")
    _require(summary["actor_optimizer_start"] == warmup and
             summary["actor_optimizer_end"] == warmup + steps and
             measured["optimizer_updates"] == steps, f"{path}: incorrect optimizer update count")
    _require(metadata["schedule_optimizer_steps"] == 691, f"{path}: wrong LR schedule")
    _require(metadata["actor_trainer"] == METHODS[summary["method"]], f"{path}: wrong actor")
    _require(metadata["valuation"] == (summary["method"] != "vanilla"),
             f"{path}: wrong valuation route")
    if method == "learnalign":
        phase = summary.get("memory_phase")
        if summary["mode"] == "timing":
            _require((warmup, steps) == (LEARNALIGN_PRE_UPDATES, 50), f"{path}: wrong LearnAlign timing window")
            pre = summary["learnalign_prelearn"]
            _require(pre["iterations_inclusive"] == [LEARNALIGN_PRE_SKIP + 1, LEARNALIGN_PRE_UPDATES]
                     and pre["iterations"] == LEARNALIGN_PRE_UPDATES - LEARNALIGN_PRE_SKIP
                     and abs(pre["elapsed_seconds"] - sum(float(r["seconds"]) for r in
                             rows[LEARNALIGN_PRE_SKIP:LEARNALIGN_PRE_UPDATES])) < 1e-6,
                     f"{path}: wrong LearnAlign pre-learning report")
        else:
            _require((phase, warmup, steps) in {("prelearn", 1, 2), ("postselect", LEARNALIGN_PRE_UPDATES, 2)},
                     f"{path}: wrong LearnAlign memory phase/window")
        selection = summary["learnalign_selector"]
        if phase == "prelearn":
            _require(selection is None, f"{path}: pre-learning unexpectedly ran the selector")
        else:
            _require(selection is not None and selection["after_optimizer_updates"] == LEARNALIGN_PRE_UPDATES
                     and selection["completions"] == LEARNALIGN_SELECTOR_COMPLETIONS
                     and selection["elapsed_seconds"] > 0 and selection["generated_tokens"] > 0,
                     f"{path}: missing full LearnAlign selector")
    for i, row in enumerate(rows, 1):
        candidate_batches = int(row.get("candidate_batches") or 1)
        _require(
            (int(row["iteration"]), int(row["prompts"]), int(row["completions"]))
            == (i, 4, 16 * candidate_batches)
            and (candidate_batches >= 1 if method == "grpo-ds" else candidate_batches == 1)
            and row["window"] == ("warmup" if i <= warmup else "measure")
            and int(row["generated_tokens"]) > 0
            and float(row["seconds"]) > 0
            and len(row["prompt_order_sha256"]) == 64,
            f"{path}: bad iteration {i}",
        )
        if method == "grpo-ds":
            rejected = int(row["ds_zero_variance_groups"]) + int(row["ds_excess_eligible_groups"])
            _require(rejected == 4 * (candidate_batches - 1), f"{path}: bad DS accounting")
        if method == "pods":
            _require(int(row["update_completions"]) == 8, f"{path}: bad PODS count")
        if method in FIXED_FILTERS:
            _require(int(row["filtered_completions"]) in range(5), f"{path}: bad fixed-filter count")
        if method == "learnalign":
            _require(all(float(row[key]) == 0 for key in
                         ("selector_rounds", "selector_completions", "selector_seconds", "selector_tokens")),
                     f"{path}: standalone selector leaked into update accounting")
        if method == "gradalign":
            expected = int((i - 1) % GRADALIGN_INTERVAL == 0)
            candidates = 16 * min(GRADALIGN_INTERVAL, 692 - i)
            _require(int(row["selector_rounds"]) == expected
                     and int(row["selector_completions"]) == expected * 4 * (30 + candidates)
                     and int(row["selector_tokens"]) == int(row["selector_reference_tokens"])
                     + int(row["selector_candidate_tokens"])
                     and (float(row["selector_seconds"]) > 0) == bool(expected),
                     f"{path}: bad GradAlign selector accounting")
    _require(sum(int(row["generated_tokens"]) + int(row.get("selector_tokens") or 0)
                 for row in rows[warmup:]) ==
             measured["generated_tokens"], f"{path}: inconsistent token count")
    _require(abs(sum(float(row["seconds"]) for row in rows[warmup:]) -
                 measured["elapsed_seconds"]) < 1e-6, f"{path}: inconsistent window timing")
    if method in FIXED_FILTERS:
        _require(sum(int(r["filtered_completions"]) for r in rows[warmup:]) == measured["filtered_completions"],
                 f"{path}: inconsistent filtered count")
    _require(sorted(s for s in _train_compile_steps(compiler, method) if warmup < s <= warmup + steps)
             == summary["measured_xla_compile_steps"], f"{path}: inconsistent compilation flags")
    for unused in ("checkpoints", "model"):
        _require(not (path / unused).exists(), f"{path}: unexpected {unused} output")
    if summary["diagnostics"] == "off":
        for unused in ("tensorboard", "selection.jsonl", "selection"):
            _require(not (path / unused).exists(), f"{path}: unexpected {unused} output")
    else:
        _require(any((path / "tensorboard").rglob("events.out.tfevents*")),
                 f"{path}: missing original TensorBoard output")
        _require((path / "selection.jsonl").is_file() == (summary["method"] == "dtv-loo"),
                 f"{path}: unexpected selection log state")
    if summary["mode"] == "memory":
        memory = _load(path / "memory.json")
        _require(memory.get("memory_phase") == summary.get("memory_phase"), f"{path}: memory phase mismatch")
        _require(len(memory["devices"]) == 4, f"{path}: incomplete device inventory")
        _require(any((path / "trace").rglob("*.xplane.pb")), f"{path}: no XPlane trace")
    return summary, metadata, [r["prompt_order_sha256"] for r in rows]


def check_comparable(runs):
    _, reference, _ = runs[0]
    compare_keys = (
        "commit_sha", "versions", "topology", "config", "schedule_optimizer_steps",
        "base_model_revision", "initial_lora_sha256", "batching", "eos_tokens",
        "actor_parameter_dtypes", "jax_default_matmul_precision", "jax_enable_x64",
        "jax_enable_compilation_cache",
    )
    for _, meta, _ in runs[1:]:
        different = [k for k in compare_keys if meta[k] != reference[k]]
        _require(not different, f"Incompatible benchmark metadata: {different}")
    by_stream = {}
    for summary, meta, prompts in runs:
        method = summary.get("method")
        key = method if method in ALIGNMENT_METHODS | {"grpo-ds"} else "fixed_stream"
        if key in by_stream:
            reference_meta, reference_prompts = by_stream[key]
            _require(meta.get("selector_config") == reference_meta.get("selector_config"),
                     "Selector budget differs")
            common = min(len(reference_prompts), len(prompts))
            _require(reference_prompts[:common] == prompts[:common], "Prompt order differs")
        if key not in by_stream or len(prompts) > len(by_stream[key][1]):
            by_stream[key] = meta, prompts


def observed_xprof_peaks(run):
    """Extract XProf's window peaks without certifying resident-buffer coverage."""
    summary, metadata, _ = check_run(run)
    _require(summary["mode"] == "memory", "Expected a memory run")
    traces = sorted(Path(run).glob("trace/plugins/profile/*/*.xplane.pb"))
    _require(len(traces) == 1, "Expected exactly one local-worker XPlane file")
    from xprof.convert import raw_to_tool_data

    raw, _ = raw_to_tool_data.xspace_to_tool_data(
        [str(traces[0])], "memory_profile", {"use_saved_result": False}
    )
    data = json.loads(raw) if raw else {}
    profiles = data.get("memoryProfilePerAllocator", {})
    ids = {str(device["id"]) for device in metadata["topology"]["devices"]}
    _require(len(ids) == 4 and ids <= set(profiles),
             f"Incomplete XProf coverage; memory IDs: {sorted(profiles)}")
    devices = {}
    for device in sorted(ids):
        profile = profiles[device]["profileSummary"]
        peak = profile["peakStats"]
        used = int(peak["peakBytesInUse"])
        stack = int(peak.get("stackReservedBytes", 0))
        heap = int(peak.get("heapAllocatedBytes", 0))
        capacity = int(profile["memoryCapacity"])
        _require(0 < used <= capacity and used == stack + heap,
                 f"Invalid stack + heap XProf peak on TPU {device}")
        devices[device] = {"peak_hbm_bytes": used, "peak_hbm_gib": used / 2**30,
                           "stack_reserved_bytes": stack, "heap_allocated_bytes": heap,
                           "capacity_bytes": capacity,
                           "peak_time_ps": int(profile.get("peakStatsTimePs", 0))}
    return {
        "collector": "XProf Memory Profile", "method": summary["method"],
        "source_run": str(Path(run).resolve()), "trace_file": str(traces[0]),
        "commit_sha": metadata["commit_sha"], "devices": devices,
        "memory_phase": summary.get("memory_phase"),
        "iterations_inclusive": [summary["warmup_iterations"] + 1, summary["actor_optimizer_end"]],
        "max_device_peak_hbm_gib": max(d["peak_hbm_gib"] for d in devices.values()),
        "resident_allocations_verified": False,
        "qualification": "Observed window stack + heap; resident buffers not independently verified.",
    }


def verified_xprof_peaks(run, evidence):
    """Validate the analyst's XProf transcription, not the raw TPU trace itself."""
    summary, metadata, _ = check_run(run)
    _require(summary["mode"] == "memory", "Expected a memory run")
    report = _load(Path(evidence))
    _require(report["collector"] == "XProf Memory Profile" and
             report["peak_scope"] == "measurement_window" and
             report["resident_allocations_verified"] is True and
             bool(report["verification_notes"].strip()), "Missing runtime/resident HBM evidence")
    ids = {str(d["id"]) for d in metadata["topology"]["devices"]}
    values = report["devices"]
    _require(set(values) == ids, "XProf evidence must cover every mesh device")
    expected_window = [summary["warmup_iterations"] + 1, summary["actor_optimizer_end"]]
    _require(report["iterations_inclusive"] == expected_window, "XProf evidence window mismatch")
    for value in values.values():
        _require(type(value["peak_hbm_bytes"]) is int and value["peak_hbm_bytes"] > 0,
                 "Provide measured HBM bytes for every device")
    maximum = max(value["peak_hbm_bytes"] for value in values.values())
    return {
        **report,
        "memory_phase": summary.get("memory_phase"),
        "max_device_peak_hbm_bytes": maximum,
        "max_device_peak_hbm_gib": maximum / 1024**3,
        "source_run": str(Path(run).resolve()),
        "commit_sha": metadata["commit_sha"],
        "source": "Operator-verified XProf runtime profile; not derived from lifetime allocator counters",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Validate smoke/report artifacts and compare run metadata")
    check.add_argument("runs", nargs="+", type=Path)
    check.add_argument("--expect-warmup", type=int)
    check.add_argument("--expect-steps", type=int)
    xprof = sub.add_parser("xprof-peaks", help="Print a per-device/maximum report from verified XProf evidence")
    xprof.add_argument("run", type=Path)
    xprof.add_argument("evidence", type=Path)
    observed = sub.add_parser("xprof-observed", help="Extract all four observed peaks with XProf")
    observed.add_argument("run", type=Path)
    disk = sub.add_parser("disk-check", help="Refuse capture with inadequate free space")
    disk.add_argument("path", type=Path)
    disk.add_argument("--min-free-gib", type=float, default=6)
    args = parser.parse_args()
    if args.command == "xprof-peaks":
        print(json.dumps(verified_xprof_peaks(args.run, args.evidence), indent=2, allow_nan=False))
        return
    if args.command == "xprof-observed":
        print(json.dumps(observed_xprof_peaks(args.run), indent=2, allow_nan=False))
        return
    if args.command == "disk-check":
        print(f"Free space before capture: {check_disk(args.path, args.min_free_gib):.2f} GiB")
        return
    runs = [check_run(path) for path in args.runs]
    check_comparable(runs)
    for path, (summary, _, _) in zip(args.runs, runs):
        if args.expect_warmup is not None:
            _require(summary["warmup_iterations"] == args.expect_warmup, f"{path}: unexpected warmup")
        if args.expect_steps is not None:
            _require(summary["measurement"]["iterations"] == args.expect_steps, f"{path}: unexpected window")
        print(json.dumps({"status": "PASS", "run": str(path), **summary}, allow_nan=False))


if __name__ == "__main__":
    main()
