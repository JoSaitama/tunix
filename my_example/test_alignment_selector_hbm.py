"""Independent TPU selector stress test; no original checkpoints are changed.

First runs two disposable real updates and real rollouts to initialize caches, then tests
full configured prompt/completion lengths and a two-prompt reference tail. This
checks actual execution, not a guarantee of every future allocator peak.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("learnalign", "gradalign"), required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-program-gib", type=float, default=0.0,
                        help="Optional compiled-program budget, not total device HBM.")
    parser.add_argument("--min-headroom-fraction", type=float, default=0.1,
                        help="Check peak/limit if the device exposes both counters.")
    args, forwarded = parser.parse_known_args()
    if args.repeats < 1 or args.max_program_gib < 0 or not 0 <= args.min_headroom_fraction < 1:
        parser.error("repeats must be positive and budget nonnegative")
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    output = Path(args.report_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "method": args.method, "phases": [],
              "scope": "selector_after_two_disposable_real_updates_no_model_exports",
              "caveat": "Execution PASS is not a guaranteed worst-case training HBM bound."}

    def save():
        (output / "hbm_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    try:
        import jax
        import jax.numpy as jnp
        from tunix.generate import tokenizer_adapter as tokenizer_lib
        from tunix.rl import rl_cluster as cluster_lib
        from .alignment_baselines.gradient_features import PromptGradientEstimator
        from .config import config_from_args
        from .deps import assert_dependencies
        from .model import (apply_lora, download_model, load_eos_tokens,
                            load_model, resolve_model_config)
        from .reward_rank_noise import config_from_env
        from .seeding import experiment_seed
        from .sharding import make_mesh
        from .train import build_cluster_config, build_grpo_config, build_optimizer, build_trainer
        from .alignment_baselines.data_utils import batch_examples
        from .alignment_baselines.scoring import cosine_scores, learnability, learnalign_scores

        assert_dependencies()
        cfg = config_from_args([
            "--source", "tfds", "--train-micro-batch-size", "4",
            "--num-generations", "4", "--learning-rate", "1e-6",
            "--mesh-counts", "4,1", "--no-wandb", "--skip-eval-before",
            "--skip-eval-after", "--metrics-log-dir", str(output / "tensorboard"),
            "--checkpoint-root", str(output / "unused_checkpoints"),
        ] + forwarded + ["--metrics-log-dir", str(output / "tensorboard"),
                         "--checkpoint-root", str(output / "unused_checkpoints")])
        if cfg.data.train_micro_batch_size != 4 or cfg.grpo.num_generations != 4:
            raise ValueError("HBM test must retain frozen 4 prompts x 4 training rollouts")
        if cfg.training.use_dynamic_batch_curation:
            raise ValueError("do not combine alignment selector test with DTV/DBC")
        if jax.process_count() != 1 or jax.local_device_count() != 4:
            raise ValueError("this test requires one process with four local TPU devices")
        if any(device.platform != "tpu" for device in jax.local_devices()):
            raise ValueError("HBM test must run on TPU, not a CPU backend")
        path = download_model(cfg.model)
        mesh, _ = make_mesh(cfg.runtime.mesh_counts)
        base = load_model(path, resolve_model_config(cfg.model.model_id), mesh)
        actor = apply_lora(base, cfg.lora, mesh=mesh)
        tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=cfg.model.tokenizer_path)
        eos = load_eos_tokens(path)
        if tokenizer.eos_id() not in eos:
            eos.append(tokenizer.eos_id())
        cluster = cluster_lib.RLCluster(
            actor=actor, reference=base, tokenizer=tokenizer,
            cluster_config=build_cluster_config(
                mesh, cfg.grpo, cfg.eval, cfg.training,
                build_optimizer(cfg.training, 691), 691, 4, eos, False),
        )
        rollouts = 8 if args.method == "learnalign" else 4
        estimator = PromptGradientEstimator(
            rl_cluster=cluster, training_algo_config=build_grpo_config(cfg.grpo),
            projection_dim=4096, projection_seed=experiment_seed() or 0,
            selection_micro_batch_size=4, noise_config=config_from_env(),
            method=args.method, grouped_feature_estimation=True,
            grouped_rollout_subbatch_size=4,
        )
        report["selector_definition"] = estimator.definition
        report["configuration"] = {
            "seed": experiment_seed(), "mismatch": estimator.noise_config.fraction,
            "prompt_length": cfg.grpo.max_prompt_length,
            "completion_length": cfg.grpo.total_generation_steps,
            "selector_rollouts": rollouts, "backward_completion_limit": 16,
            "projection_dim": 4096, "training_beta": cfg.grpo.beta,
            "learning_rate": cfg.training.learning_rate, "lora": vars(cfg.lora),
        }

        def memory_snapshot():
            snapshots = []
            for device in jax.local_devices():
                try:
                    stats = device.memory_stats()
                except (NotImplementedError, RuntimeError):
                    stats = None
                snapshots.append({"device": str(device), "stats": (
                    None if stats is None else {key: int(value) for key, value in stats.items()}
                )})
            return snapshots

        # Optional best-effort compilation report. Keep the SAME compiled
        # executable in the estimator cache; never compare against legacy 32 AD.
        original_feature_function = estimator._feature_function
        measured = set()
        def measured_feature_function(count):
            function = original_feature_function(count)
            def invoke(model, example):
                key = (count, tuple(example.prompt_ids.shape), tuple(example.completion_ids.shape))
                if key not in measured:
                    measured.add(key)
                    entry = {"rollouts": count, "prompt_shape": key[1], "completion_shape": key[2]}
                    try:
                        executable = function.lower(model, example).compile()
                        analysis = executable.memory_analysis()
                        if analysis is None:
                            raise RuntimeError("memory analysis unavailable")
                        fields = ("argument_size_in_bytes", "output_size_in_bytes",
                                  "temp_size_in_bytes", "alias_size_in_bytes")
                        entry["compiler_bytes"] = {field: int(getattr(analysis, field)) for field in fields}
                        values = entry["compiler_bytes"]
                        entry["program_estimate_gib"] = (
                            values["argument_size_in_bytes"] + values["output_size_in_bytes"]
                            + values["temp_size_in_bytes"] - values["alias_size_in_bytes"]
                        ) / 1024**3
                    except Exception as exc:
                        # Compilation OOM must fail, not be silently retried.
                        message = str(exc).lower()
                        if any(marker in message for marker in (
                            "resource_exhausted", "out of memory", "failed to allocate",
                            "program hbm requirement", "out_of_memory",
                        )):
                            raise
                        entry["compiler_memory_unavailable"] = str(exc)
                    report.setdefault("programs", []).append(entry)
                    save()
                    if args.max_program_gib:
                        if "program_estimate_gib" not in entry:
                            raise RuntimeError("requested compiler budget cannot be verified on this runtime")
                        if entry["program_estimate_gib"] > args.max_program_gib:
                            raise RuntimeError("compiled selector exceeds --max-program-gib")
                return function(model, example)
            return invoke
        estimator._feature_function = measured_feature_function

        examples = [{"prompts": f"What is {i + 1} + 1? Give the final answer.",
                     "answer": str(i + 2), "index": i} for i in range(4)]
        def phase(name, batch, *, mismatch):
            print(f"HBM phase {name}: {len(batch)} prompts x {rollouts} rollouts", flush=True)
            start = time.monotonic()
            estimate = estimator.estimate(batch, num_rollouts=rollouts, apply_mismatch=mismatch)
            if estimate.features.shape != (len(batch), 4096) or not np.isfinite(estimate.features).all():
                raise RuntimeError("invalid selector features")
            if name.startswith("full_length") and not np.any(estimate.selector_advantages != 0):
                raise RuntimeError("synthetic HBM input must exercise nonzero advantages")
            if args.method == "learnalign":
                scores = learnalign_scores(estimate.features, learnability(estimate.selector_rewards))
            else:
                scores = cosine_scores(estimate.features, estimate.features.mean(axis=0))
            if not np.isfinite(scores).all():
                raise RuntimeError("invalid selector scores")
            snapshots = memory_snapshot()
            headrooms = []
            for snapshot in snapshots:
                stats = snapshot["stats"] or {}
                if stats.get("bytes_limit", 0) and "peak_bytes_in_use" in stats:
                    headrooms.append(1 - stats["peak_bytes_in_use"] / stats["bytes_limit"])
            report["phases"].append({"name": name, "seconds": time.monotonic() - start,
                                     "feature_shape": list(estimate.features.shape),
                                     "device_memory": snapshots,
                                     "peak_headroom_fraction": headrooms or None})
            save()
            if headrooms and min(headrooms) < args.min_headroom_fraction:
                raise RuntimeError("measured peak device HBM leaves insufficient requested headroom")

        # Keep the actual train executable and Adam state resident when testing
        # selection, as at the LearnAlign warmup/selection boundary. These two
        # updates affect only this newly loaded disposable actor, never a run.
        print("Preparing HBM context: two disposable frozen-parameter GRPO updates", flush=True)
        trainer = build_trainer(cluster, cfg.grpo)
        trainer.train([batch_examples(examples), batch_examples(examples)])
        report["disposable_training_updates"] = 2
        phase("real_rollouts_initialize_generation_cache", examples, mismatch=True)
        valid_tokens = np.asarray([i for i in range(10, 100)
                                   if i not in eos and i != cluster.rollout.pad_id()], dtype=np.int32)
        token = int(valid_tokens[0])
        def full_length_rollout(*, prompts, **unused):
            size = len(prompts)
            # Distinct responses avoid artificial all-response Jacobian
            # equality/cancellation while keeping every position valid.
            positions = np.arange(cfg.grpo.total_generation_steps)[None, :]
            completions = valid_tokens[(positions + np.arange(size)[:, None]) % len(valid_tokens)]
            completions[:, -1] = tokenizer.eos_id()
            prompt_tokens = np.full((size, cfg.grpo.max_prompt_length), token, dtype=np.int32)
            text = [f"<answer>{int(examples[i // rollouts]['answer']) if i % 2 == 0 else 999999}</answer>"
                    for i in range(size)]
            return SimpleNamespace(tokens=jnp.asarray(completions),
                                   left_padded_prompt_tokens=jnp.asarray(prompt_tokens), text=text)
        with mock.patch.object(cluster, "generate", side_effect=full_length_rollout):
            for repeat in range(args.repeats):
                phase(f"full_length_candidates_{repeat}", examples, mismatch=True)
            # GradAlign's 30 references end with a two-prompt microbatch;
            # check that shape as well as clean reference scoring.
            phase("full_length_clean_reference", examples, mismatch=False)
            phase("full_length_two_prompt_tail", examples[:2], mismatch=False)
        report["passed"] = True
        report["headroom_verified"] = all(
            len(phase["peak_headroom_fraction"] or []) == len(jax.local_devices())
            for phase in report["phases"])
        save()
        print(f"PASS: {output / 'hbm_report.json'}", flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
