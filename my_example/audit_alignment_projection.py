"""Read-only, same-rollout LoRA projection audit; never calls trainer.train.

Run --self-test on CPU. TPU mode requires an existing run directory/checkpoint.
Exact refers to the current selector objective in LoRA space, not full-model
gradients or historical selection equivalence. Raw gradients are disk-backed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .alignment_baselines.scoring import (
    cosine_scores, learnability, learnalign_scores, normalize_rows,
    stable_top_indices,
)


def mix_u32(values, seed):
    with np.errstate(over="ignore"):
        x = np.asarray(values, dtype=np.uint32) + np.uint32(seed & 0xFFFFFFFF)
        x = (x ^ (x >> np.uint32(16))) * np.uint32(0x7FEB352D)
        x = (x ^ (x >> np.uint32(15))) * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def hash_features(raw, leaf_sizes, dimension, seed, *, decorrelated=False):
    """CPU mirror of current hash; optional separate sign stream for diagnosis.

    The alternate stream is a deterministic engineering control, not a claim
    of mathematically independent hash families or a JL guarantee.
    """
    result = np.zeros((raw.shape[0], dimension), dtype=np.float32)
    offset = 0
    for index, size in enumerate(leaf_sizes):
        coordinates = np.arange(size, dtype=np.uint32) + np.uint32(offset)
        leaf_seed = seed + 0x9E3779B9 * (index + 1)
        mixed = mix_u32(coordinates, leaf_seed)
        buckets = (mixed % np.uint32(dimension)).astype(np.int64)
        sign_hash = mix_u32(coordinates, leaf_seed + 0xD1B54A35) if decorrelated else mixed
        signs = np.where((sign_hash & np.uint32(1)) == 0, 1.0, -1.0)
        for row in range(raw.shape[0]):
            summed = np.bincount(
                buckets, weights=raw[row, offset:offset + size] * signs,
                minlength=dimension,
            ).astype(np.float32)
            result[row] += summed
        offset += size
    if offset != raw.shape[1]:
        raise ValueError("gradient leaf sizes do not match raw dimension")
    return result


def gram_matrix(raw, block=8192):
    result = np.zeros((raw.shape[0], raw.shape[0]), dtype=np.float64)
    for start in range(0, raw.shape[1], block):
        part = np.asarray(raw[:, start:start + block], dtype=np.float64)
        result += part @ part.T
    return result


def gram_cosines(gram):
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    return gram / np.maximum(norms[:, None], 1e-12) / np.maximum(norms[None, :], 1e-12)


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def rank_correlation(a, b):
    x, y = average_ranks(a), average_ranks(b)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sign_comparison(a, b, epsilon):
    strong = np.abs(a) > epsilon
    near = ~strong
    return {
        "reference_non_near_zero_count": int(strong.sum()),
        "reference_near_zero_count": int(near.sum()),
        "non_near_zero_sign_agreement": (
            float(np.mean(np.sign(a[strong]) == np.where(np.abs(b[strong]) > epsilon, np.sign(b[strong]), 0)))
            if strong.any() else None
        ),
        "opposite_sign_count": int(np.sum(strong & (np.abs(b) > epsilon) & (a * b < 0))),
        "non_near_zero_reference_to_near_zero_count": int(np.sum(strong & (np.abs(b) <= epsilon))),
        "near_zero_reference_to_non_near_zero_count": int(np.sum(near & (np.abs(b) > epsilon))),
    }


def compare_scores(exact, projected, count, epsilon):
    a, b = stable_top_indices(exact, count), stable_top_indices(projected, count)
    overlap = len(set(a) & set(b))
    ordered = np.sort(exact)[::-1]
    return {
        "spearman": rank_correlation(exact, projected),
        "max_abs_score_error": float(np.max(np.abs(exact - projected))),
        "mean_abs_score_error": float(np.mean(np.abs(exact - projected))),
        "signs": sign_comparison(exact, projected, epsilon),
        "top_k": count,
        "top_k_overlap_fraction": overlap / count,
        "top_k_jaccard": overlap / (2 * count - overlap),
        "top_k_order_identical": bool(np.array_equal(a, b)),
        "exact_cutoff_gap": float(ordered[count - 1] - ordered[count]),
        "exact_selected_local_indices": a.tolist(),
        "projected_selected_local_indices": b.tolist(),
    }


def geometry_comparison(exact_cos, projected_cos, epsilon):
    i, j = np.triu_indices(len(exact_cos), 1)
    a, b = exact_cos[i, j], projected_cos[i, j]
    errors = np.abs(a - b)
    return {
        "pair_count": len(a),
        "mean_abs_cosine_error": float(errors.mean()),
        "p95_abs_cosine_error": float(np.quantile(errors, 0.95)),
        "max_abs_cosine_error": float(errors.max()),
        "signs": sign_comparison(a, b, epsilon),
    }


def exact_scores(gram, values, candidate_count, method):
    if method == "learnalign":
        cos = gram_cosines(gram)
        return values * (cos @ values) / len(values)
    ref_count = len(gram) - candidate_count
    cross = gram[:candidate_count, candidate_count:].mean(axis=1)
    ref_norm = np.sqrt(max(gram[candidate_count:, candidate_count:].sum() / ref_count**2, 0.0))
    if ref_norm <= 1e-12:
        return np.zeros(candidate_count)
    return cross / np.maximum(np.sqrt(np.maximum(np.diag(gram)[:candidate_count], 0)), 1e-12) / ref_norm


def self_test():
    g = np.array([[2., -1.], [-1., 2.]])
    assert np.isclose(gram_cosines(gram_matrix(g))[0, 1], -0.8)
    merged = g.sum(axis=1, keepdims=True)
    assert np.isclose((normalize_rows(merged) @ normalize_rows(merged).T)[0, 1], 1)
    r, a, b = np.array([1., 0., 1.]), np.array([1., -1., 1.]), np.array([0., 1., 1.])
    raw = np.array([a, b, r])
    phi = np.column_stack((raw[:, 0] + raw[:, 1], -raw[:, 2]))
    exact = exact_scores(gram_matrix(raw), None, 2, "gradalign")
    projected = cosine_scores(phi[:2], phi[2])
    assert stable_top_indices(exact, 1).tolist() == [0]
    assert stable_top_indices(projected, 1).tolist() == [1]
    unit = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., 1.]])
    values = np.full(4, .25)
    assert np.allclose(exact_scores(gram_matrix(unit), values, 4, "learnalign"), learnalign_scores(unit, values))
    assert compare_scores(np.array([2., 1., 0.]), np.array([2., 1., 0.]), 1, 1e-8)["top_k_jaccard"] == 1
    assert rank_correlation(np.zeros(4), np.zeros(4)) is None
    assert sign_comparison(np.array([1.]), np.array([1e-10]), 1e-8)["non_near_zero_sign_agreement"] == 0
    assert np.allclose(gram_matrix(raw, block=1), raw @ raw.T)
    mixed = mix_u32(np.arange(1000, dtype=np.uint32), 0)
    assert np.all((mixed % 4096) % 2 == mixed % 2)
    print("PASS: toy negative-sign flip, ranking flip, LearnAlign formula, ties and hash parity")


def parameter_digest(model):
    """Checks weights, without printing parameters or credentials."""
    import hashlib
    from flax import nnx
    import jax
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(nnx.state(model, nnx.LoRAParam)):
        values = np.ascontiguousarray(jax.device_get(leaf))
        digest.update(str(values.shape).encode())
        digest.update(values.dtype.str.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def raw_gradient_function(estimator, rollout_count):
    """Audit-only copy of existing completion-gradient AD; no monkey patching."""
    from flax import nnx
    import jax
    from tunix.rl import function_registry
    from .alignment_baselines.gradient_features import project_gradient_tree

    score_cfg = dataclasses.replace(estimator.training_algo_config, beta=0., num_generations=rollout_count)
    loss_fn = function_registry.get_policy_loss_fn(score_cfg.policy_loss_fn)

    def function(model, example):
        def loss(model, one):
            value, _ = loss_fn(model, one, algo_config=score_cfg,
                               pad_id=estimator.rl_cluster.rollout.pad_id(),
                               eos_id=estimator.rl_cluster.rollout.eos_id())
            return value
        derivative = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.LoRAParam))
        axes = jax.tree.map(lambda x: None if x is None else 0, example,
                            is_leaf=lambda x: x is None)
        _, gradients = jax.vmap(derivative, in_axes=(None, axes))(model, example)
        means = jax.tree.map(lambda x: x.reshape((-1, rollout_count) + x.shape[1:]).mean(axis=1), gradients)
        current = project_gradient_tree(means, projection_dim=estimator.projection_dim,
                                        seed=estimator.projection_seed)
        return means, current
    return nnx.jit(function)


def collect_gradients(estimator, examples, rollouts, destination, max_disk_gb):
    import jax
    import jax.numpy as jnp
    from tunix.rl import rl_cluster as cluster_lib, utils as rl_utils
    from .alignment_baselines.equivalence import rollout_subbatch_completion_indices
    from .alignment_baselines.artifacts import prompt_id

    feature_rollouts = min(4, rollouts)
    if rollouts % feature_rollouts:
        raise ValueError("rollouts must be divisible by the memory-bounded rollout batch")
    raw_fn = raw_gradient_function(estimator, feature_rollouts)
    legacy_fn = estimator._feature_function(feature_rollouts)
    model = estimator.rl_cluster.actor_trainer.model
    features, rewards, records = [], [], []
    matrix, leaf_sizes = None, None
    for start in range(0, len(examples), 4):
        chunk = examples[start:start + 4]
        example, clean, reward, adv, noisy, effective = estimator._generate_binary_train_example(
            [x[0] for x in chunk], num_rollouts=rollouts,
            apply_mismatch=chunk[0][2] == "candidate")
        raw_sum, audit_sum, legacy_sum = None, None, None
        parts = rollout_subbatch_completion_indices(prompt_count=len(chunk), num_rollouts=rollouts,
                                                     rollout_subbatch_size=feature_rollouts)
        actor_mesh = estimator.rl_cluster.cluster_config.role_to_mesh[cluster_lib.Role.ACTOR]
        with actor_mesh, estimator.rl_cluster._get_logical_axis_rules_cm(cluster_lib.Role.ACTOR):
            for indices in parts:
                sub = rl_utils.get_batch_slice(example, jnp.asarray(indices))
                tree, current = jax.block_until_ready(raw_fn(model, sub))
                leaves = [np.asarray(jax.device_get(x), dtype=np.float32) for x in jax.tree.leaves(tree)]
                sizes = [int(np.prod(x.shape[1:])) for x in leaves]
                flat = np.concatenate([x.reshape(len(chunk), -1) for x in leaves], axis=1)
                current = np.asarray(jax.device_get(current), dtype=np.float32)
                del tree, leaves
                legacy = np.asarray(jax.device_get(jax.block_until_ready(legacy_fn(model, sub))), dtype=np.float32)
                if leaf_sizes is None:
                    leaf_sizes = sizes
                    byte_count = len(examples) * sum(sizes) * 4
                    if byte_count > max_disk_gb * 1024**3:
                        raise ValueError(f"raw gradient disk estimate {byte_count / 1024**3:.2f} GiB exceeds --max-raw-disk-gb")
                    print(f"Raw LoRA dimension={sum(sizes)}; disk estimate={byte_count / 1024**3:.2f} GiB", flush=True)
                    matrix = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                                       shape=(len(examples), sum(sizes)))
                if sizes != leaf_sizes:
                    raise ValueError("LoRA gradient tree layout changed")
                if raw_sum is None:
                    raw_sum, audit_sum, legacy_sum = flat, current.copy(), legacy.copy()
                else:
                    raw_sum += flat
                    audit_sum += current
                    legacy_sum += legacy
                del flat
        calls = rollouts // feature_rollouts
        matrix[start:start + len(chunk)] = raw_sum / calls
        features.append(np.stack((audit_sum / calls, legacy_sum / calls)))
        rewards.append(reward)
        for k, (item, source, role) in enumerate(chunk):
            records.append({"prompt_id": prompt_id(str(item["prompts"])), "source_index": source,
                            "role": role, "clean_binary_outcomes": clean[k].tolist(),
                            "selector_rewards": reward[k].tolist(), "selector_advantages": adv[k].tolist(),
                            "mismatch_selected": bool(noisy[k]), "mismatch_effective": bool(effective[k])})
        print(f"Computed {start + len(chunk)}/{len(examples)} {chunk[0][2]} gradients", flush=True)
    matrix.flush()
    return matrix, leaf_sizes, np.concatenate(features, axis=1), np.concatenate(rewards), records


def run(args, forwarded):
    # Set seeds before importing data/model code. Read source run, write elsewhere.
    run_dir = Path(args.run_dir).expanduser().resolve()
    metadata = json.loads((run_dir / "selection/run_metadata.json").read_text())
    alignment = metadata["alignment"]
    method = args.method or alignment["method"]
    if method != alignment["method"]:
        raise ValueError("audit method must match the source run metadata")
    seed = metadata.get("experiment_seed") or 0
    noise = metadata.get("reward_rank_mismatch", {})
    os.environ["TUNIX_EXPERIMENT_SEED"] = str(seed)
    os.environ["TUNIX_REWARD_RANK_NOISE_SEED"] = str(noise.get("seed", seed))
    os.environ["TUNIX_REWARD_RANK_NOISE_FRACTION"] = str(noise.get("fraction", 0))
    os.environ["WANDB_DISABLED"], os.environ["WANDB_MODE"] = "true", "disabled"
    output = Path(args.audit_output_dir).expanduser().resolve()
    if output == run_dir or run_dir in output.parents:
        raise ValueError("audit output must be outside the source run")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.lora_checkpoint).expanduser().resolve() if args.lora_checkpoint else (
        run_dir / "checkpoints/actor" / str(metadata["effective_max_steps"]) / "model_params")
    if not checkpoint.is_dir():
        raise ValueError(f"LoRA model_params checkpoint missing: {checkpoint}; supply --lora-checkpoint explicitly")
    from flax import nnx
    import jax
    from orbax import checkpoint as ocp
    from tunix.generate import tokenizer_adapter as tokenizer_lib
    from tunix.rl import rl_cluster as cluster_lib
    from .config import config_from_args
    from .data import batch_dataset, get_dataset
    from .alignment_baselines.data_utils import unbatch
    from .alignment_baselines.gradient_features import PromptGradientEstimator
    from .main import _suppress_jax_monitoring
    from .model import apply_lora, download_model, load_eos_tokens, load_model, resolve_model_config
    from .reward_rank_noise import config_from_env
    from .sharding import make_mesh
    from .train import build_cluster_config, build_grpo_config, build_optimizer

    defaults = ["--source", "tfds", "--max-train-examples", "3072", "--train-fraction", "0.9",
                "--train-micro-batch-size", "4", "--num-generations", "4", "--learning-rate", "1e-6",
                "--mesh-counts", "4,1", "--no-wandb", "--skip-eval-before", "--skip-eval-after"]
    frozen = metadata.get("frozen_update_configuration", {})
    for key, flag in [("max_prompt_length", "--max-prompt-length"),
                      ("total_generation_steps", "--total-generation-steps"),
                      ("beta", "--beta"), ("epsilon", "--epsilon")]:
        if key in frozen:
            defaults += [flag, str(frozen[key])]
    cfg = config_from_args(defaults + forwarded + ["--metrics-log-dir", str(output / "tensorboard"),
                                                  "--checkpoint-root", str(output / "unused_checkpoints")])
    if cfg.training.use_dynamic_batch_curation or cfg.data.train_micro_batch_size != 4:
        raise ValueError("audit requires existing alignment batch=4 without dynamic curation")
    if cfg.data.source != "tfds":
        raise ValueError("this independent audit currently supports the frozen TFDS GSM8K path")
    actual_frozen = {"train_micro_batch_size": cfg.data.train_micro_batch_size,
                     "num_generations": cfg.grpo.num_generations,
                     "learning_rate": cfg.training.learning_rate, "beta": cfg.grpo.beta,
                     "epsilon": cfg.grpo.epsilon, "max_prompt_length": cfg.grpo.max_prompt_length,
                     "total_generation_steps": cfg.grpo.total_generation_steps}
    mismatches = {key: {"source": value, "audit": actual_frozen[key]} for key, value in frozen.items()
                  if key in actual_frozen and actual_frozen[key] != value}
    if mismatches:
        raise ValueError(f"audit must preserve recorded frozen configuration: {mismatches}")
    all_examples = unbatch(batch_dataset(get_dataset(cfg.data.train_data_dir, "train", cfg.data.source),
                                         4, cfg.data.max_train_examples))
    split = int((len(all_examples) // 4) * cfg.data.train_fraction) * 4
    train, heldout = all_examples[:split], all_examples[split:]
    count = args.candidate_count or (64 if method == "learnalign" else 160)
    refs = args.reference_count or int(alignment.get("gradalign_validation_prompts", 30))
    q = int(alignment.get("selection_ratio", 4))
    selected_count = (count // q) // 4 * 4
    if count % 4 or count < 8 or count > len(train) or not 0 < selected_count < count:
        raise ValueError("candidate count must fit training pool, be divisible by 4, and yield nonempty top-k")
    if method == "gradalign" and not 0 < refs <= len(heldout):
        raise ValueError("reference count does not fit heldout pool")
    if method == "learnalign":
        indices = np.sort(np.random.default_rng(args.sample_seed).choice(len(train), count, replace=False)).tolist()
    else:
        indices = [(args.candidate_offset + k) % len(train) for k in range(count)]
    candidates = [(train[k], k, "candidate") for k in indices]
    reference_examples = [(item, split + k, "reference") for k, item in enumerate(heldout[:refs])]
    model_path = download_model(cfg.model)
    mesh, _ = make_mesh(cfg.runtime.mesh_counts)
    base = load_model(model_path, resolve_model_config(cfg.model.model_id), mesh)
    actor = apply_lora(base, cfg.lora, mesh)
    with mesh, _suppress_jax_monitoring():
        target = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), nnx.state(actor, nnx.LoRAParam))
        restored = ocp.StandardCheckpointer().restore(str(checkpoint), target=target)
        nnx.update(actor, restored)
    before_digest = parameter_digest(actor)
    eos = load_eos_tokens(model_path)
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=cfg.model.tokenizer_path)
    if tokenizer.eos_id() not in eos:
        eos.append(tokenizer.eos_id())
    max_steps = int(metadata["effective_max_steps"])
    cluster_cfg = build_cluster_config(mesh, cfg.grpo, cfg.eval, cfg.training,
                                        build_optimizer(cfg.training, max_steps), max_steps, 4, eos, False)
    cluster = cluster_lib.RLCluster(actor=actor, reference=base, tokenizer=tokenizer, cluster_config=cluster_cfg)
    dimension = int(alignment.get("projection_dim", 4096))
    estimator = PromptGradientEstimator(rl_cluster=cluster, training_algo_config=build_grpo_config(cfg.grpo),
                                       projection_dim=dimension, projection_seed=seed, selection_micro_batch_size=4,
                                       noise_config=config_from_env(), grouped_feature_estimation=method == "learnalign")
    started = time.monotonic()
    rollouts = int(alignment.get("learnalign_estimation_rollouts", 8) if method == "learnalign"
                   else alignment.get("gradalign_candidate_rollouts", 4))
    raw, sizes, current_parts, rewards, records = collect_gradients(
        estimator, candidates, rollouts, output / "candidate_raw.npy",
        args.max_raw_disk_gb * count / (2 * (count + (refs if method == "gradalign" else 0))))
    raw_arrays, current_arrays = [raw], [current_parts]
    if method == "gradalign":
        ref_raw, ref_sizes, ref_parts, _, ref_records = collect_gradients(
            estimator, reference_examples, int(alignment.get("gradalign_validation_rollouts", 4)),
            output / "reference_raw.npy", args.max_raw_disk_gb * refs / (2 * (count + refs)))
        if sizes != ref_sizes:
            raise ValueError("reference/candidate LoRA layouts differ")
        raw_arrays.append(ref_raw)
        current_arrays.append(ref_parts)
        records += ref_records
    # Disk-backed combination avoids an all-pool TPU allocation or RAM copy.
    total = sum(x.shape[0] for x in raw_arrays)
    if shutil.disk_usage(output).free < total * raw.shape[1] * 4 + 512 * 1024**2:
        raise ValueError("insufficient disk space for combined audit gradients plus 512 MiB reserve")
    combined = np.lib.format.open_memmap(output / "combined_raw.npy", mode="w+", dtype=np.float32,
                                         shape=(total, raw.shape[1]))
    offset = 0
    for array in raw_arrays:
        combined[offset:offset + len(array)] = array
        offset += len(array)
    combined.flush()
    after_digest = parameter_digest(cluster.actor_trainer.model)
    audit_current = np.concatenate([x[0] for x in current_arrays])
    legacy_current = np.concatenate([x[1] for x in current_arrays])
    cpu_current = hash_features(combined, sizes, dimension, seed)
    gram = gram_matrix(combined)
    exact_cos = gram_cosines(gram)
    values = learnability(rewards) if method == "learnalign" else None
    exact = exact_scores(gram, values, count, method)
    scales = np.maximum(np.linalg.norm(legacy_current.astype(np.float64), axis=1), 1e-12)
    extraction_errors = np.linalg.norm(audit_current.astype(np.float64) - legacy_current, axis=1) / scales
    mirror_errors = np.linalg.norm(cpu_current.astype(np.float64) - audit_current, axis=1) / np.maximum(
        np.linalg.norm(audit_current.astype(np.float64), axis=1), 1e-12)
    valid = bool(before_digest == after_digest and np.max(extraction_errors) <= args.extraction_rtol
                 and np.max(mirror_errors) <= args.extraction_rtol)
    report = {"method": method, "checkpoint": str(checkpoint), "seed": seed,
              "candidate_prompts": count, "reference_prompts": total - count, "projection_dim": dimension,
              "raw_lora_dimension": raw.shape[1], "gradient_leaf_sizes": sizes,
              "scope": "posthoc same-checkpoint, same-rollout LoRA-selector audit; NOT historical full-pool equivalence",
              "implementation_checks": {"passed": valid, "maximum_relative_row_error_vs_existing_estimator": float(extraction_errors.max()),
                                        "maximum_relative_row_error_cpu_hash_mirror": float(mirror_errors.max()),
                                        "lora_weights_unchanged": before_digest == after_digest,
                                        "tolerance": args.extraction_rtol},
              "frozen_configuration": actual_frozen,
              "thresholds": {"cosine_near_zero": args.cosine_epsilon, "score_near_zero": args.score_epsilon,
                             "min_sign_agreement": args.min_sign_agreement, "min_top_k_overlap": args.min_overlap,
                             "min_score_spearman": args.min_spearman,
                             "max_p95_abs_cosine_error": args.max_cosine_error},
              "candidates_with_V_zero": int(np.sum(values == 0)) if values is not None else None,
              "exact_zero_gradient_candidates": int(np.sum(np.diag(gram)[:count] == 0)),
              "current_zero_feature_candidates": int(np.sum(np.linalg.norm(legacy_current[:count], axis=1) == 0)),
              "exact_selected_mismatch_fraction": float(np.mean([records[k]["mismatch_selected"]
                                                                  for k in stable_top_indices(exact, selected_count)])),
              "comparisons": {}}
    variants = {"current": legacy_current}
    for k in range(args.signed_controls):
        variants[f"decorrelated_sign_control_{k}"] = hash_features(combined, sizes, dimension, seed + k * 104729, decorrelated=True)
    score_arrays = {"exact": exact}
    for name, feature in variants.items():
        projected = learnalign_scores(feature, values) if method == "learnalign" else cosine_scores(feature[:count], feature[count:].mean(axis=0))
        comparison = compare_scores(exact, projected, selected_count, args.score_epsilon)
        comparison["score_counts"] = {"positive": int(np.sum(projected > 0)),
                                       "exact_zero": int(np.sum(projected == 0)),
                                       "negative": int(np.sum(projected < 0)),
                                       "near_zero": int(np.sum(np.abs(projected) <= args.score_epsilon))}
        comparison["geometry"] = geometry_comparison(exact_cos[:count, :count],
                                                      (normalize_rows(feature) @ normalize_rows(feature).T)[:count, :count], args.cosine_epsilon)
        selected = stable_top_indices(projected, selected_count)
        comparison["selected_mismatch_fraction"] = float(np.mean([records[k]["mismatch_selected"] for k in selected]))
        report["comparisons"][name] = comparison
        score_arrays[name] = projected
    current = report["comparisons"]["current"]
    if values is not None:
        zero_score = (values > 0) & (score_arrays["current"] == 0)
        zero_raw = np.diag(gram)[:count] == 0
        zero_feature = np.linalg.norm(legacy_current[:count], axis=1) == 0
        report["V_positive_zero_score_breakdown"] = {
            "count": int(zero_score.sum()),
            "with_zero_raw_gradient": int(np.sum(zero_score & zero_raw)),
            "with_nonzero_raw_but_zero_projected_feature": int(np.sum(zero_score & ~zero_raw & zero_feature)),
            "with_nonzero_raw_and_projected_feature": int(np.sum(zero_score & ~zero_raw & ~zero_feature)),
        }
    agreement = current["signs"]["non_near_zero_sign_agreement"]
    rho = current["spearman"]
    informative = int(np.sum(np.abs(exact) > args.score_epsilon)) >= max(4, count // 10)
    geometry_agreement = current["geometry"]["signs"]["non_near_zero_sign_agreement"]
    passed = valid and informative and agreement is not None and agreement >= args.min_sign_agreement and (
        rho is not None and rho >= args.min_spearman) and current["top_k_overlap_fraction"] >= args.min_overlap and (
        geometry_agreement is not None and geometry_agreement >= args.min_sign_agreement) and (
        current["geometry"]["p95_abs_cosine_error"] <= args.max_cosine_error)
    report.update({"informative_scores": informative, "sample_checks_passed": bool(passed),
                   "wall_seconds": time.monotonic() - started,
                   "note": "Thresholds are diagnostic choices, not theoretical guarantees; passing does not validate every seed/update."})
    (output / "audit_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "prompt_audit.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    np.savez(output / "scores_and_cosines.npz", exact_gram=gram, exact_cosines=exact_cos,
             current_features=legacy_current, **score_arrays)
    if not args.keep_raw:
        # Only these three files newly created by this audit; never a run checkpoint.
        del combined, raw_arrays, raw
        if method == "gradalign":
            del ref_raw
        for filename in ("candidate_raw.npy", "reference_raw.npy", "combined_raw.npy"):
            path = output / filename
            if path.exists():
                path.unlink()
    print(json.dumps({"report": str(output / "audit_report.json"), "sample_checks_passed": bool(passed),
                      "implementation_checks_passed": valid, "top_k_overlap": current["top_k_overlap_fraction"],
                      "score_spearman": rho, "score_sign_agreement": agreement}, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--lora-checkpoint", help="Explicit Orbax model_params directory; default is final actor checkpoint")
    parser.add_argument("--audit-output-dir")
    parser.add_argument("--method", choices=("learnalign", "gradalign"))
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--reference-count", type=int)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260917)
    parser.add_argument("--signed-controls", type=int, default=2)
    parser.add_argument("--max-raw-disk-gb", type=float, default=8)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--cosine-epsilon", type=float, default=1e-3)
    parser.add_argument("--score-epsilon", type=float, default=1e-8)
    parser.add_argument("--extraction-rtol", type=float, default=1e-3)
    parser.add_argument("--min-sign-agreement", type=float, default=.95)
    parser.add_argument("--min-overlap", type=float, default=.95)
    parser.add_argument("--min-spearman", type=float, default=.95)
    parser.add_argument("--max-cosine-error", type=float, default=.1,
                        help="Diagnostic ceiling on p95 absolute pairwise cosine error")
    args, forwarded = parser.parse_known_args()
    if args.self_test:
        self_test()
        return
    if not args.run_dir or not args.audit_output_dir:
        parser.error("--run-dir and --audit-output-dir are required except in --self-test")
    if args.signed_controls < 1 or args.max_raw_disk_gb <= 0:
        parser.error("signed control count and disk budget must be positive")
    if args.reference_count is not None and args.reference_count <= 0:
        parser.error("reference count must be positive")
    if args.candidate_count is not None and args.candidate_count <= 0:
        parser.error("candidate count must be positive")
    if args.cosine_epsilon <= 0 or args.score_epsilon <= 0 or args.extraction_rtol <= 0:
        parser.error("near-zero and extraction tolerances must be positive")
    if args.max_cosine_error < 0:
        parser.error("cosine error ceiling must be nonnegative")
    if not all(0 <= x <= 1 for x in (args.min_sign_agreement, args.min_overlap, args.min_spearman)):
        parser.error("agreement, overlap and rank thresholds must be between 0 and 1")
    raise SystemExit(run(args, forwarded))


if __name__ == "__main__":
    main()
