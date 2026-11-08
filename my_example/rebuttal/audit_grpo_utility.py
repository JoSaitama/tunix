"""Fixed-checkpoint utility audit for R1/R2, not a training entry point.

All model execution must run on an idle TPU worker. CPU unit tests only import
the argument/statistics layer. No optimizer state is loaded or invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .utility_math import (
    assumption_evidence, cluster_interval, direction_summary, fraction, matched_random,
    reference_summary, scores_from_gram, selection_masks, unit_summary,
)
from .utility_data import (
    load_rollout, partition_indices, read_partition, replay_metadata, sha256_file,
    validate_partition, validate_replay_contract,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--step", type=int, default=500)
    p.add_argument("--expected-commit", default="6fba11c8d2e4cb00ee19e8ed45e196521d3f93d0")
    p.add_argument("--mode", choices=("preflight", "roundtrip", "numerics", "smoke", "hbm", "score", "probe"), default="preflight")
    p.add_argument("--reward-mode", choices=("accuracy", "legacy"), required=True,
                   help="Explicit source dense reward branch; never infer it from checkpoint weights.")
    p.add_argument("--heldout-reward-mode", choices=("same", "binary"), default="same")
    p.add_argument("--checkpoint-training-condition", default="clean",
                   help="Provenance, independent of candidate-noise; not inferred automatically.")
    p.add_argument("--historical-source-config-confirmed", action="store_true")
    p.add_argument("--model-id", default="google/gemma-3-1b-it")
    p.add_argument("--backbone-path")
    p.add_argument("--tokenizer-path", default="gs://gemma-data/tokenizers/tokenizer_gemma3.model")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=float, default=64.)
    p.add_argument("--train-data-dir", default="./data/train")
    p.add_argument("--pool-prompts", type=int, default=3072)
    p.add_argument("--train-fraction", type=float, default=.9)
    p.add_argument("--dataset-shuffle-seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--rollout-seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--candidate-noise", type=float, default=0.)
    p.add_argument("--candidate-prompts", type=int)
    p.add_argument("--heldout-prompts", type=int)
    p.add_argument("--partition-file", help="Reuse saved data_partition.json or audit_manifest.json IDs")
    p.add_argument("--replay-rollouts", help="Read-only previous audit RUN directory; no generation/TFDS loading")
    p.add_argument("--prompt-length", type=int, default=256)
    p.add_argument("--completion-length", type=int, default=768)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--beta", type=float, default=.08)
    p.add_argument("--epsilon", type=float, default=.2)
    p.add_argument("--scope", choices=("group", "batch"), default="group")
    p.add_argument("--mask-application", choices=("full", "policy-only"), default="full")
    p.add_argument("--loo-min-keep-fraction", type=float, default=.25,
                   help="Production default .25; explicit 0 is threshold-only, NOT production cap.")
    p.add_argument("--random-repeats", type=int)
    p.add_argument("--finite-random-repeats", type=int, default=1)
    p.add_argument("--probe-step-sizes", type=float, nargs="+")
    p.add_argument("--probe-max-grad-norm", type=float, default=.1)
    p.add_argument("--bootstrap-repeats", type=int, default=200)
    p.add_argument("--zero-epsilon", type=float, default=1e-12)
    p.add_argument("--min-headroom-fraction", type=float, default=.05)
    p.add_argument("--numerical-rtol", type=float, default=.1)
    p.add_argument("--numerical-atol", type=float, default=1e-7)
    return p


def validate(args):
    if args.partition_file and args.replay_rollouts:
        raise ValueError("choose partition-file OR replay-rollouts, not both")
    fixed_partition = None
    if args.replay_rollouts:
        _, metadata = replay_metadata(args.replay_rollouts, args.output_dir)
        fixed_partition = metadata["data_partition"]
    elif args.partition_file:
        fixed_partition = read_partition(args.partition_file)
    if fixed_partition is not None:
        if args.candidate_prompts is None:
            args.candidate_prompts = len(fixed_partition["candidate_indices"])
        if args.heldout_prompts is None:
            args.heldout_prompts = len(fixed_partition["heldout_indices"])
    small = args.mode in {"roundtrip", "numerics", "smoke", "hbm"}
    args.candidate_prompts = args.candidate_prompts if args.candidate_prompts is not None else (4 if small else 32)
    args.heldout_prompts = args.heldout_prompts if args.heldout_prompts is not None else (8 if small else 64)
    args.random_repeats = args.random_repeats if args.random_repeats is not None else (1 if small else 3)
    args.probe_step_sizes = args.probe_step_sizes if args.probe_step_sizes is not None else (
        [1e-4] if small and args.mode != "numerics" else [1e-6, 1e-4])
    positive = (args.step, args.candidate_prompts, args.heldout_prompts, args.pool_prompts,
                args.lora_rank, args.lora_alpha, args.prompt_length, args.completion_length,
                args.temperature, args.top_k, args.zero_epsilon)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("sizes, step, LoRA and numerical epsilon must be finite and positive")
    if any(value % 4 for value in (args.candidate_prompts, args.heldout_prompts, args.pool_prompts)):
        raise ValueError("prompt counts must be multiples of the frozen four-prompt microbatch")
    if not 0 < args.train_fraction < 1 or not 0 < args.top_p <= 1:
        raise ValueError("invalid split/top_p")
    if not 0 <= args.candidate_noise <= 1 or not 0 <= args.loo_min_keep_fraction <= 1:
        raise ValueError("invalid noise/retention fraction")
    if not 0 <= args.min_headroom_fraction < 1:
        raise ValueError("invalid memory headroom")
    if any(value < 0 for value in (args.dataset_shuffle_seed, args.sample_seed, args.rollout_seed, args.noise_seed)):
        raise ValueError("seeds must be nonnegative")
    if args.random_repeats < 1 or not 0 <= args.finite_random_repeats <= args.random_repeats:
        raise ValueError("invalid random repeats")
    if args.bootstrap_repeats < 0 or args.probe_max_grad_norm < 0 or not np.isfinite(args.probe_max_grad_norm):
        raise ValueError("invalid bootstrap or clipping")
    if not 0 <= args.beta < float("inf") or not 0 < args.epsilon < 1:
        raise ValueError("invalid GRPO beta/epsilon")
    if any(not np.isfinite(eta) or eta <= 0 for eta in args.probe_step_sizes):
        raise ValueError("probe step sizes must be positive and finite")
    if len(set(args.probe_step_sizes)) != len(args.probe_step_sizes):
        raise ValueError("duplicate probe step sizes would duplicate observations")
    if not 0 <= args.numerical_rtol <= .25 or not 0 < args.numerical_atol <= 1e-6:
        raise ValueError("numerical tolerance outside declared diagnostic bounds")
    split = int((args.pool_prompts // 4) * args.train_fraction) * 4
    if args.candidate_prompts > split or args.heldout_prompts > args.pool_prompts - split:
        raise ValueError("requested audit prompts exceed their disjoint source split")
    if fixed_partition is not None:
        validate_partition(fixed_partition, args.pool_prompts, split, args.candidate_prompts, args.heldout_prompts)
    return args


def select_examples(args):
    # Reproduce the archived data construction with an EXPLICIT shuffle seed,
    # independent of the current seeding helper or inherited environment.
    import grain
    from my_example.data import _load_from_tfds, _as_text, _format_example
    dataset = grain.MapDataset.source(_load_from_tfds(args.train_data_dir, "train")).shuffle(
        seed=args.dataset_shuffle_seed).map(lambda item: _format_example(
            _as_text(item["question"]), _as_text(item["answer"])))
    if len(dataset) < args.pool_prompts:
        raise ValueError("TFDS source is smaller than the recorded prompt pool")
    pool = [dict(dataset[index], source_index=index) for index in range(args.pool_prompts)]
    split = int((args.pool_prompts // 4) * args.train_fraction) * 4
    fixed = None
    if args.partition_file:
        fixed = read_partition(args.partition_file)
        candidate_indices, heldout_indices = validate_partition(
            fixed, len(pool), split, args.candidate_prompts, args.heldout_prompts)
    else:
        candidate_indices, heldout_indices = partition_indices(
            len(pool), split, args.candidate_prompts, args.heldout_prompts, args.sample_seed)
    train_prompts = {str(item["prompts"]) for item in pool[:split]}
    if any(str(pool[index]["prompts"]) in train_prompts for index in heldout_indices):
        raise ValueError("heldout prompt duplicates a training prompt; do not silently replace it")
    candidates, heldout = [pool[i] for i in candidate_indices], [pool[i] for i in heldout_indices]
    if len({item["prompts"] for item in candidates + heldout}) != len(candidates) + len(heldout):
        raise ValueError("audit contains duplicate prompts")
    hashes = {name: [hashlib.sha256(item["prompts"].encode()).hexdigest() for item in examples]
              for name, examples in (("candidate_prompt_hashes", candidates), ("heldout_prompt_hashes", heldout))}
    if fixed is not None and any(name in fixed and fixed[name] != values for name, values in hashes.items()):
        raise ValueError("fixed source IDs now map to different prompt text; dataset/format/shuffle changed")
    return candidates, heldout, {"training_pool_prompts": split, "heldout_pool_prompts": len(pool) - split,
                                  "candidate_indices": candidate_indices.tolist(),
                                  "heldout_indices": heldout_indices.tolist(),
                                  "sampling_schema": "independent-permutation-prefix-v2" if not args.partition_file else "fixed-IDs",
                                  **hashes,
                                  "final_test_used": False, "split_unit": "four-prompt batch"}


def append_csv(path, rows):
    if not rows:
        return
    existed = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not existed:
            writer.writeheader()
        writer.writerows(rows)


def report_reference(stats, h_norm, args, batch_id, host=None, h_report=None):
    result = []
    width = 4 if args.scope == "group" else 16
    for start in range(0, 16, width):
        norm = max(0., float(stats["reference_norm_squared"][start // width])) ** .5
        dot = float(stats["utility_policy"][start:start + width].mean())
        informative = norm > args.zero_epsilon and h_norm > args.zero_epsilon
        cosine = float(np.clip(dot / (norm * h_norm), -1., 1.)) if informative else None
        row = {"batch_id": batch_id, "scope_index": start // width,
                       "prompt_id": (str(host["prompt_ids_hash"][start]) if host is not None and width == 4
                                     else f"batch_{batch_id}"),
                       "noise_selected": bool(host["noise_selected"][start // 4]) if host is not None and width == 4 else None,
                       "noise_effective": bool(host["noise_effective"][start // 4]) if host is not None and width == 4 else None,
                       "reference_norm": norm, "heldout_gradient_norm": h_norm,
                       "dot_with_heldout": dot, "cosine": cosine, "informative": informative}
        split = (h_report or {}).get("split_half", {})
        for index, half in enumerate(("a", "b")):
            half_norm = split.get("l2_norms", [0., 0.])[index]
            available = split.get("available", False) and half_norm > args.zero_epsilon
            value = float(stats[f"utility_half_{half}"][start:start + width].mean()) if available else None
            row[f"dot_half_{half}"] = value
            row[f"cosine_half_{half}"] = (float(np.clip(value / (norm * half_norm), -1., 1.))
                                          if available and norm > args.zero_epsilon else None)
        result.append(row)
    return result


def build_unit_rows(host, stats, masks, batch_id, split_available=False):
    rows = []
    group_counts = host["correct"].reshape(-1, 4).sum(axis=1)
    for i in range(16):
        group = i // 4
        rows.append({
            "batch_id": batch_id, "prompt_id": str(host["prompt_ids_hash"][i]),
            "source_index": int(host["source_indices"][i]), "generation_index": i % 4,
            "clean_reward": float(host["clean_rewards"][i]),
            "observed_reward": float(host["observed_rewards"][i]),
            "advantage": float(host["advantages"][i]), "correct": bool(host["correct"][i]),
            "group_correct_count": int(group_counts[group]),
            "unique_correct": bool(host["correct"][i] and group_counts[group] == 1),
            "noise_selected": bool(host["noise_selected"][group]),
            "noise_effective": bool(host["noise_effective"][group]),
            "valid_completion_tokens": int(host["completion_mask"][i].sum()),
            "score_dtv": float(stats["score_dtv"][i]), "score_loo": float(stats["score_loo"][i]),
            "self_term": float(stats["self_term"][i]), "cross_term": float(stats["cross_term"][i]),
            "policy_gradient_norm": max(0., float(stats["gram"][i, i])) ** .5,
            "utility_policy": float(stats["utility_policy"][i]),
            "utility_full": float(stats["utility_full"][i]),
            "utility_half_a": float(stats["utility_half_a"][i]) if split_available else None,
            "utility_half_b": float(stats["utility_half_b"][i]) if split_available else None,
            "keep_dtv": bool(masks["dtv"][i]), "keep_loo": bool(masks["loo"][i]),
            "keep_loo_threshold": bool(masks["loo_threshold"][i]),
            "loo_cap_rescue": bool(masks["loo_cap_rescue"][i]),
        })
    return rows


def verify_end(runtime, checkpoint, preflight, summary, output, replay_files):
    from my_example.audit_grpo_checkpoint_restore import fingerprint_directory, write_report
    summary["parameter_validation"] = runtime.assert_original_parameters("audit_end")
    current = runtime.parameter_summary(runtime.jax, runtime.nnx.state(runtime.actor, runtime.nnx.LoRAParam))
    if current["sha256"] != runtime.initial_summary["sha256"]:
        write_report(output / "parameter_restore_failure.json", {
            "stage": "audit_end_sha256", "original_sha256": runtime.initial_summary["sha256"],
            "current_sha256": current["sha256"]})
        raise RuntimeError("audit final parameter SHA256 mismatch")
    summary["theta0_restored"] = True
    summary["source_unchanged"] = fingerprint_directory(checkpoint) == preflight["input_fingerprint"]
    summary["replay_source_unchanged"] = all(sha256_file(path) == digest for path, digest in replay_files.items())
    if not summary["source_unchanged"] or not summary["replay_source_unchanged"]:
        raise RuntimeError("checkpoint or replay source files changed")
    memory = json.loads((output / "hbm_report.json").read_text(encoding="utf-8"))
    summary["hbm_headroom_verified"] = memory["headroom_verified"]


def run(args):
    from my_example.audit_grpo_checkpoint_restore import (
        fingerprint_directory, metadata_preflight, validate_paths, write_report,
    )
    source, output, checkpoint = validate_paths(args.input_dir, args.output_dir, args.step)
    output.mkdir(parents=True, exist_ok=False)
    (output / "fixed_rollouts").mkdir()
    started = time.monotonic()
    summary = {"execution_passed": False, "utility_audit_completed": False, "mode": args.mode,
               "diagnostic_version": 2, "finite_step_validated": False,
               "gradient_geometry_completed": False,
               "optimizer_replayed": False,
               "objective": "independent clean fixed-rollout clipped-GRPO policy surrogate, no KL",
               "utility_scope": "local surrogate utility, NOT greedy accuracy or long-term return",
               "probe_update": "SGD direction, optional global-norm clip; NOT historical AdamW",
               "uncertainty_scope": "candidate prompt/batch clusters, conditional on checkpoint and heldout estimate",
               "warnings": []}
    runtime = None
    replay_files = {}
    try:
        preflight = metadata_preflight(source, checkpoint, args.expected_commit)
        manifest = {"arguments": vars(args), "checkpoint": str(checkpoint), "preflight": preflight,
                    "score_gradient": "KL-free policy loss in restored trainable LoRA space",
                    "gradient_execution": "production-style vmap per completion, 16 responses per call",
                    "inner_products": "exact FP32 tree reductions; no hash projection",
                    "diagnostic_heldout_gradient_or_labels_used_in_selection": False,
                    "all_methods_share_candidate_rollouts": True,
                    "test_data_used": False, "backbone_historical_revision_verified": False}
        repo = Path(__file__).resolve().parents[2]
        manifest["audit_source_sha256"] = {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in (
                "my_example/rebuttal/audit_grpo_utility.py", "my_example/rebuttal/utility_runtime.py",
                "my_example/rebuttal/utility_math.py", "my_example/rebuttal/parameter_snapshot.py",
                "my_example/rebuttal/utility_data.py",
                "tunix/rl/grpo/grpo_learner.py",
                "my_example/rewards.py", "my_example/reward_rank_noise.py", "my_example/prompts.py")}
        recorded_data_versions = {}
        freeze = source / "pip-freeze.txt"
        if freeze.is_file():
            for line in freeze.read_text(encoding="utf-8").splitlines():
                for package in ("grain", "tensorflow-datasets"):
                    if line.lower().startswith(package + "=="):
                        recorded_data_versions[package] = line.split("==", 1)[1]
        installed_data_versions = {}
        for package in ("grain", "tensorflow-datasets"):
            try:
                installed_data_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                installed_data_versions[package] = None
        manifest["data_library_versions"] = {"recorded": recorded_data_versions,
                                            "installed": installed_data_versions,
                                            "historical_split_ids_independently_verified": False}
        if any(installed_data_versions[name] != version for name, version in recorded_data_versions.items()):
            summary["warnings"].append("Data library versions differ: seed alone does not certify the historical split. Verify source IDs before interpreting as historical heldout.")
        try:
            code = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            manifest["audit_code_commit"] = code.stdout.strip() if code.returncode == 0 else None
        except FileNotFoundError:
            # Synced worker1 may have neither a .git directory nor git binary.
            # Source hashes above still identify the executed implementation.
            manifest["audit_code_commit"] = None
        replay_root, replay_manifest = None, None
        if args.replay_rollouts:
            replay_root, replay_manifest = replay_metadata(args.replay_rollouts, output)
            validate_replay_contract(replay_manifest, args, repo)
            partition = replay_manifest["data_partition"]
            manifest["data_partition"] = partition
            replay_files[str(replay_root / "audit_manifest.json")] = sha256_file(replay_root / "audit_manifest.json")
            identities = []
            for role, key in (("candidate", "candidate_indices"), ("heldout", "heldout_indices")):
                for start in range(0, len(partition[key]), 4):
                    path = replay_root / "fixed_rollouts" / f"{role}_{start // 4:03d}.npz"
                    host = load_rollout(path, partition[key][start:start + 4])
                    identities.extend(host["prompt_ids_hash"].reshape(4, 4)[:, 0].tolist())
                    replay_files[str(path)] = sha256_file(path)
            if len(set(identities)) != len(identities):
                raise ValueError("duplicate/overlapping prompts in replay archive")
            manifest["replay"] = {"source": str(replay_root), "source_files_sha256": replay_files,
                                  "tokens_reused": True, "no_generation": True,
                                  "candidate_rewards_reapplied": "exact prompt strings when available; legacy same-condition only"}
        write_report(output / "audit_manifest.json", manifest)
        if not args.historical_source_config_confirmed:
            summary["warnings"].append("Source launch overrides/reward branch are not independently confirmed; audit protocol is explicit, not historical replay.")
        if args.mode == "preflight":
            summary["execution_passed"] = True
            return output

        # Isolate before runtime imports JAX. A physical worker1 is logical0.
        os.environ.update({"TPU_PROCESS_BOUNDS": "1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
                           "TPU_VISIBLE_CHIPS": "0,1,2,3", "CLOUD_TPU_TASK_ID": "0",
                           "WANDB_DISABLED": "true", "WANDB_MODE": "disabled"})
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("JAX_PLATFORM_NAME", None)
        from .utility_runtime import UtilityRuntime
        if args.mode == "roundtrip":
            runtime = UtilityRuntime(args, checkpoint, output)
            manifest.update(topology=runtime.topology, backbone_path=runtime.model_path,
                            restored_lora=runtime.initial_summary,
                            snapshot_storage="detached_copied_array_tuple")
            write_report(output / "audit_manifest.json", manifest)
            summary["parameter_roundtrip"] = runtime.roundtrip_check()
            summary["theta0_restored"] = True
            summary["source_unchanged"] = fingerprint_directory(checkpoint) == preflight["input_fingerprint"]
            if not summary["source_unchanged"]:
                raise RuntimeError("source checkpoint files changed during parameter roundtrip")
            summary["execution_passed"] = True
            return output
        if replay_manifest is None:
            candidates, heldout_examples, partition = select_examples(args)
        else:
            partition = replay_manifest["data_partition"]
        manifest["data_partition"] = partition
        write_report(output / "data_partition.json", partition)
        write_report(output / "audit_manifest.json", manifest)
        runtime = UtilityRuntime(args, checkpoint, output)
        manifest["topology"] = runtime.topology
        manifest["backbone_path"] = runtime.model_path
        manifest["restored_lora"] = runtime.initial_summary
        manifest["snapshot_storage"] = "detached_copied_array_tuple"
        manifest["snapshot_bytes"] = int(runtime._snapshot.bytes)
        if replay_manifest is not None:
            if runtime.initial_summary["sha256"] != replay_manifest["restored_lora"]["sha256"]:
                raise ValueError("replay requires exactly the same restored theta0")
            old_backbone = replay_manifest.get("backbone_path")
            if not old_backbone or Path(runtime.model_path).name != Path(old_backbone).name:
                raise ValueError("replay backbone snapshot identity differs; pin the original --backbone-path")
        write_report(output / "audit_manifest.json", manifest)
        # Finish ALL rollouts at theta0 BEFORE any temporary parameter change.
        if replay_manifest is None:
            heldout = [runtime.prepare(heldout_examples[i:i + 4], role="heldout", batch_id=i // 4)
                       for i in range(0, len(heldout_examples), 4)]
            candidate_batches = [runtime.prepare(candidates[i:i + 4], role="candidate", batch_id=i // 4)
                                 for i in range(0, len(candidates), 4)]
        else:
            def replay(role, key):
                indices = partition[key]
                return [runtime.replay_prepare(
                    replay_root / "fixed_rollouts" / f"{role}_{i // 4:03d}.npz", indices[i:i + 4],
                    role=role, batch_id=i // 4, original_args=replay_manifest["arguments"])
                        for i in range(0, len(indices), 4)]
            heldout, candidate_batches = replay("heldout", "heldout_indices"), replay("candidate", "candidate_indices")
        runtime.assert_original_parameters("after_shared_rollouts")
        h, h_report = runtime.heldout_gradient(heldout)
        runtime.assert_original_parameters("after_heldout_gradients")
        summary["independent_gradient"] = h_report
        baseline = runtime.clean_value(heldout)
        baseline_repeat = runtime.clean_value(heldout)
        summary["baseline_repeat_abs_difference"] = abs(baseline - baseline_repeat)
        numerical = None
        if args.mode in {"numerics", "smoke", "hbm", "probe"}:
            numerical = runtime.numerical_preflight(heldout, h, baseline, baseline_repeat)
            summary["numerical_check"] = {"passed": numerical["passed"],
                                          "validated_step_sizes": numerical["validated_step_sizes"],
                                          "report": str(output / "numerical_report.json")}
            if not numerical["passed"]:
                summary["warnings"].append("Finite-step numerical controls FAILED: only gradient geometry is interpretable; method probes skipped.")
        if args.mode == "numerics":
            verify_end(runtime, checkpoint, preflight, summary, output, replay_files)
            summary["execution_passed"] = True
            return output
        if args.mode == "hbm":
            runtime.stress(heldout, h, baseline)
            summary["full_length_stress_completed"] = True
        units, batches, references, first_order, random_rows = [], [], [], [], []
        mean_reference_sum = None
        for batch_id, host in enumerate(candidate_batches):
            runtime.assert_original_parameters(f"before_candidate_batch_{batch_id}")
            # Synchronize each kernel: no overlapping total/policy AD programs.
            full = runtime.gradients(host, full=True)
            runtime.snapshot_memory(f"candidate_full_backward_{batch_id}")
            pg = runtime.gradients(host)
            stats = runtime.batch_statistics(pg, full, h)
            runtime.assert_original_parameters(f"after_candidate_gradients_{batch_id}")
            with runtime.mesh:
                mean_reference = runtime.average(pg, runtime.jnp.ones(16))
                mean_reference_sum = (mean_reference if mean_reference_sum is None
                                      else runtime.add(mean_reference_sum, mean_reference))
                runtime.jax.block_until_ready(mean_reference_sum)
            runtime.snapshot_memory(f"candidate_policy_gram_{batch_id}")
            masks = selection_masks(stats["score_dtv"], stats["score_loo"], scope=args.scope,
                                    min_keep_fraction=args.loo_min_keep_fraction)
            oracle = scores_from_gram(stats["gram"], scope=args.scope)
            reconstruction = {name: float(np.max(np.abs(stats[f"score_{name}"] - oracle[name])))
                              for name in ("dtv", "loo")}
            unit_rows = build_unit_rows(host, stats, masks, batch_id,
                                        split_available=h_report["split_half"]["available"])
            units.extend(unit_rows)
            append_csv(output / "unit_utility.csv", unit_rows)
            reference_rows = report_reference(stats, h_report["l2_norm"], args, batch_id, host, h_report)
            references.extend(reference_rows)
            append_csv(output / "reference_alignment.csv", reference_rows)
            method_masks = {name: masks[name] for name in ("vanilla", "dtv", "loo")}
            for target_id, name in enumerate(("dtv", "loo")):
                for repeat in range(args.random_repeats):
                    key = f"random_match_{name}_{repeat}"
                    mask = matched_random(masks[name], scope=args.scope, group_size=4,
                                          seed=args.sample_seed + 1000 * target_id, batch_id=batch_id, repeat=repeat)
                    method_masks[key] = mask
                    random_rows.extend({"batch_id": batch_id, "method": key, "unit_index": i,
                                        "selected": bool(mask[i])} for i in range(16))
            for name, mask in method_masks.items():
                direction = runtime.direction(pg, full, mask)
                with runtime.mesh:
                    utility = float(runtime.jax.device_get(runtime.dot(h, direction)))
                    norm = float(runtime.jax.device_get(runtime.dot(direction, direction))) ** .5
                if not np.isfinite(utility) or not np.isfinite(norm):
                    raise FloatingPointError("nonfinite aggregated direction utility/norm")
                record = {"batch_id": batch_id, "method": name, "kept_units": int(mask.sum()),
                          "retained_fraction": float(mask.mean()), "direction_dot_clean_gradient": utility,
                          "direction_norm": norm,
                          "direction_cosine_clean": (utility / (norm * h_report["l2_norm"])
                                                     if norm > args.zero_epsilon and h_report["informative"] else None),
                          "mean_selected_policy_unit_utility": fraction(stats["utility_policy"][mask]),
                          "score_gram_reconstruction_max_error": max(reconstruction.values())}
                first_order.append(record)
                append_csv(output / "direction_utility.csv", [record])
                finite = args.mode in {"probe", "smoke", "hbm"} and numerical["passed"]
                if name.startswith("random_") and int(name.rsplit("_", 1)[1]) >= args.finite_random_repeats:
                    finite = False
                if finite:
                    for eta in args.probe_step_sizes:
                        if eta not in numerical["validated_step_sizes"]:
                            continue
                        probe = runtime.probe(direction, h, heldout, baseline, eta)
                        floor = max(abs(baseline - baseline_repeat),
                                    float(np.finfo(np.float32).eps) * max(1., abs(baseline)))
                        probe["evaluation_noise_floor"] = floor
                        probe["utility_above_noise_floor"] = abs(probe["utility"]) > floor
                        residual = abs(probe["utility"] - probe["predicted_first_order_utility"])
                        probe["taylor_abs_residual"] = residual
                        probe["measurement_resolved"] = abs(probe["predicted_first_order_utility"]) > 10 * floor
                        probe["measurement_consistent"] = bool(
                            (probe["changed_parameter_elements"] == 0 and abs(probe["utility"]) <= args.numerical_atol) or
                            (probe["measurement_resolved"] and residual <= args.numerical_atol +
                             args.numerical_rtol * abs(probe["predicted_first_order_utility"])))
                        probe.update(batch_id=batch_id, method=name, kept_units=int(mask.sum()))
                        batches.append(probe)
                        append_csv(output / "batch_utility.csv", [probe])
                del direction
            del full, pg
            print(f"Candidate audit: {(batch_id + 1) * 4}/{args.candidate_prompts} prompts", flush=True)
        append_csv(output / "random_controls.csv", random_rows)
        summary["units"] = unit_summary(units, args.zero_epsilon, args.bootstrap_repeats)
        summary["reference_alignment"] = reference_summary(references, epsilon=args.zero_epsilon,
                                                           repeats=args.bootstrap_repeats)
        for name in ("dtv", "loo"):
            append_csv(output / "filter_strata.csv", [dict(method=name, **row)
                       for row in summary["units"][name]["noise_advantage_joint"]])
        summary["assumption_evidence"] = assumption_evidence(summary["reference_alignment"], summary["units"], h_report)
        with runtime.mesh:
            mean_reference = runtime.scale(mean_reference_sum, 1. / len(candidate_batches))
            mean_reference_norm = float(runtime.jax.device_get(runtime.dot(mean_reference, mean_reference))) ** .5
            mean_reference_dot = float(runtime.jax.device_get(runtime.dot(mean_reference, h)))
        summary["reference_alignment"]["mean_reference_norm"] = mean_reference_norm
        summary["reference_alignment"]["mean_reference_dot_clean"] = mean_reference_dot
        summary["reference_alignment"]["cosine_mean_reference_with_clean"] = (
            mean_reference_dot / (mean_reference_norm * h_report["l2_norm"])
            if mean_reference_norm > args.zero_epsilon and h_report["informative"] else None)
        summary["first_order_directions"] = direction_summary(first_order, repeats=args.bootstrap_repeats)
        summary["finite_step"] = {}
        for eta in args.probe_step_sizes:
            subset = [row for row in batches if row["eta"] == eta]
            if not subset:
                continue
            by_method = {name: [row for row in subset if row["method"] == name]
                         for name in sorted({row["method"] for row in subset})}
            results = {}
            for name, rows in by_method.items():
                values = [row["utility"] for row in rows]
                results[name] = {"batches": len(rows), "mean_utility": fraction(values),
                                 "measurement_consistency_passed": all(row["measurement_consistent"] for row in rows),
                                 "interpretation": "raw observed mean; interpret only if consistency passes",
                                 "batch_cluster_ci95": cluster_interval(values, [row["batch_id"] for row in rows],
                                                                         repeats=args.bootstrap_repeats),
                                 "no_parameter_change_probes": sum(not row["parameter_delta_informative"] for row in rows)}
                results[name]["below_evaluation_noise_floor_probes"] = sum(
                    not row["utility_above_noise_floor"] for row in rows)
            for left, right in (("dtv", "loo"), ("dtv", "random_match_dtv_0"),
                                ("loo", "random_match_loo_0")):
                if left in by_method and right in by_method:
                    right_values = {row["batch_id"]: row["utility"] for row in by_method[right]}
                    differences = [row["utility"] - right_values[row["batch_id"]] for row in by_method[left]]
                    results[f"paired_{left}_minus_{right}"] = {
                        "mean": fraction(differences),
                        "batch_cluster_ci95": cluster_interval(differences, [row["batch_id"] for row in by_method[left]],
                                                                repeats=args.bootstrap_repeats)}
            summary["finite_step"][str(eta)] = results
        summary["finite_step_validated"] = bool(batches) and all(row["measurement_consistent"] for row in batches)
        summary["assumption_evidence"]["finite_step_measurement"] = {
            "performed": bool(batches), "validated": summary["finite_step_validated"],
            "numerical_control_passed": numerical["passed"] if numerical is not None else None}
        write_report(output / "assumption_report.json", summary["assumption_evidence"])
        if batches and not summary["finite_step_validated"]:
            summary["warnings"].append("Some method probes are unresolved/inconsistent: do not treat raw finite-step means as verified utility.")
        verify_end(runtime, checkpoint, preflight, summary, output, replay_files)
        if args.mode == "hbm" and not summary["hbm_headroom_verified"]:
            raise RuntimeError("execution completed but TPU peak counters are unavailable; HBM headroom NOT verified")
        summary["utility_audit_completed"] = True
        summary["gradient_geometry_completed"] = True
        summary["finite_step_completed"] = bool(batches)
        summary["execution_passed"] = True
        if not h_report["informative"]:
            summary["warnings"].append("Independent heldout gradient near zero: completed execution is NOT evidence supporting reference quality.")
        return output
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["reviewer_evidence_validated"] = False
        write_report(output / "audit_summary.json", summary)
        print(json.dumps({"output": str(output), "execution_passed": summary["execution_passed"],
                          "utility_audit_completed": summary["utility_audit_completed"],
                          "finite_step_completed": summary.get("finite_step_completed", False),
                          "finite_step_validated": summary["finite_step_validated"],
                          "numerical_check_passed": summary.get("numerical_check", {}).get("passed")}, indent=2))


def main(argv=None):
    args = validate(parser().parse_args(argv))
    output = run(args)
    if args.mode == "numerics":
        report = json.loads((output / "numerical_report.json").read_text(encoding="utf-8"))
        if not report["passed"]:
            print("NUMERICAL GATE FAILED (exit 2): no method benefit inferred; inspect numerical_report.json", flush=True)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
