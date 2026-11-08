"""One bounded FP32 measurement control, NOT a production utility/method audit.

Replay exactly eight held-out prompts, do no generation/candidate scan, and
never overwrite the archived BF16 logps. FP32 theta0 old-policy logps define a
NEW diagnostic objective. A pass does not certify the BF16 production path.
JAX/model imports are lazy so the argument/report tests can run on server CPU.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np

from .audit_grpo_utility import parser as audit_parser, validate
from .utility_data import load_rollout, replay_metadata, sha256_file, validate_replay_contract
from .utility_math import numerical_pair_check


def parse_args(argv=None):
    p = audit_parser()
    p.description = __doc__
    p.set_defaults(mode="numerics")
    args = validate(p.parse_args(argv))
    if not args.replay_rollouts or not args.backbone_path:
        raise ValueError("FP32 control requires existing --replay-rollouts and --backbone-path")
    if args.mode != "numerics" or args.heldout_prompts != 8 or args.candidate_prompts != 4:
        raise ValueError("bounded control ONLY: numerics, eight heldout and four archived candidate prompts")
    if args.probe_step_sizes != [1e-6, 1e-4]:
        raise ValueError("use the two predeclared control steps [1e-6, 1e-4]; no step search")
    if args.numerical_rtol != .1 or args.numerical_atol != 1e-7:
        raise ValueError("control tolerances are fixed; do not relax the previous numerical gate")
    if args.probe_max_grad_norm != .1 or args.min_headroom_fraction < .05:
        raise ValueError("keep clip=.1 and at least 5% observed peak HBM headroom")
    if args.heldout_reward_mode != "same":
        raise ValueError("keep the archived clean dense reward objective, not a new binary objective")
    return args


def clean_host(host):
    """Validate the OLD clean rewards/A before defining the new FP32 objective."""
    from tunix.rl.grpo.grpo_learner import compute_advantages

    if host["noise_selected"].any() or host["noise_effective"].any() or not np.array_equal(
            host["clean_rewards"], host["observed_rewards"]):
        raise ValueError("heldout archive contains corrupted rewards")
    expected = compute_advantages(host["observed_rewards"], 4).astype(np.float32)
    if not np.array_equal(expected, host["advantages"]):
        raise ValueError("heldout advantages differ from the archived clean reward calculation")
    if not np.any(host["completion_mask"]):
        raise ValueError("heldout batch has no valid tokens")
    return dict(host)  # independent mapping; never replace fields in the source


def logp_difference(current, archived, mask, epsilon):
    current, archived = np.asarray(current), np.asarray(archived)
    if current.dtype != np.float32 or current.shape != archived.shape:
        raise ValueError("FP32 forward must actually return float32 logps with the original shape")
    delta = (current - archived.astype(np.float32))[np.asarray(mask, dtype=bool)]
    if not delta.size or not np.isfinite(delta).all():
        raise ValueError("nonfinite or empty valid logps in FP32 control")
    return {"valid_tokens": int(delta.size), "mean_abs": float(np.abs(delta).mean()),
            "max_abs": float(np.abs(delta).max()),
            "importance_ratio_clipped_fraction": float(np.mean(
                (delta < np.log(1. - epsilon)) | (delta > np.log(1. + epsilon))))}


def traced_float_dtypes(value):
    """Inspect nested JAXPR avals, not full logits or activation array values."""
    found, seen = set(), set()

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if hasattr(item, "jaxpr"):
            visit(item.jaxpr)
        if hasattr(item, "eqns"):
            variables = list(item.invars) + list(item.outvars) + list(item.constvars)
            for equation in item.eqns:
                variables += list(equation.invars) + list(equation.outvars)
                visit(equation.params)
            for variable in variables:
                dtype = str(getattr(getattr(variable, "aval", None), "dtype", ""))
                if dtype.startswith(("float", "bfloat", "complex")):
                    found.add(dtype)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def check_forward_precision(runtime, host):
    """Trace the actual mean-loss graph once; fail rather than assume FP32."""
    with runtime.mesh, runtime.monitor(runtime.jax):
        graph, state = runtime.nnx.split(runtime.actor)

        def forward(parameters, example):
            model = runtime.nnx.merge(graph, parameters)
            return runtime.clean_loss(model, example)

        traced = runtime.jax.make_jaxpr(forward)(state, runtime.as_example(host))
    dtypes = traced_float_dtypes(traced)
    if dtypes != ["float32"]:
        raise TypeError(f"control forward contains non-FP32 floating intermediates: {dtypes}")
    return {"floating_intermediate_dtypes": dtypes,
            "scope": "nested JAXPR for the actual mean loss, not compiler assembly/accuracy guarantee"}


def run_pairs(runtime, heldout, h, baseline, repeat):
    """Same zero/+/- gate as v2, but explicitly under the NEW FP32 objective."""
    from my_example.audit_grpo_checkpoint_restore import write_report

    args = runtime.args
    floor = max(abs(baseline - repeat), np.finfo(np.float32).eps * max(1., abs(baseline)))
    report = {"passed": False, "validated_step_sizes": [], "pairs": [],
              "baseline": baseline, "same_parameter_repeat_difference": abs(baseline - repeat),
              "minimum_resolution_floor": float(floor), "rtol": .1, "atol": 1e-7,
              "production_path_validated": False,
              "objective": "FP32 mean(per-unit clipped GRPO loss); FP32 unit-forward theta0 old logps",
              "not_method_evidence": True}
    zero = runtime.probe(h, h, heldout, baseline, 0.)
    zero_restore = abs(runtime.clean_value(heldout) - baseline)
    report.update(zero_step=zero, zero_step_restored_loss_difference=zero_restore,
                  zero_step_passed=bool(zero["changed_parameter_elements"] == 0 and
                                       abs(zero["utility"]) <= args.numerical_atol and
                                       zero_restore <= args.numerical_atol))
    write_report(runtime.output / "numerical_report.json", report)
    for eta in args.probe_step_sizes:
        print(f"FP32 measurement control +/- eta={eta}; NO candidate selection", flush=True)
        plus = runtime.probe(h, h, heldout, baseline, eta)
        restore_a = abs(runtime.clean_value(heldout) - baseline)
        minus = runtime.probe(h, h, heldout, baseline, -eta)
        restore_b = abs(runtime.clean_value(heldout) - baseline)
        check = numerical_pair_check(plus, minus, floor=floor, rtol=.1, atol=1e-7)
        check["restored_loss_max_difference"] = max(restore_a, restore_b)
        check["passed"] = bool(check["passed"] and report["zero_step_passed"] and
                               max(restore_a, restore_b) <= args.numerical_atol)
        report["pairs"].append({"eta": eta, "positive": plus, "negative": minus, "check": check})
        if check["passed"]:
            report["validated_step_sizes"].append(eta)
        report["passed"] = bool(report["validated_step_sizes"])
        write_report(runtime.output / "numerical_report.json", report)
    return report


def configure_fp32(runtime):
    """Promote diagnostic actor state, not files or production model defaults.

    Casting the backbone is essential: casting only the final loss cannot undo
    BF16 activation/log-softmax quantization. Gemma's activation casts follow
    their input dtype. The output logps dtype is checked separately at runtime.
    """
    from .parameter_snapshot import ParameterSnapshot
    from .utility_runtime import production_mean_loss
    from tunix.rl.grpo.grpo_learner import grpo_loss_fn

    jax, jp, nnx = runtime.jax, runtime.jnp, runtime.nnx
    original_lora = runtime.initial_summary
    state = nnx.state(runtime.actor)
    leaves = jax.tree.leaves(state)
    before = sorted({str(x.dtype) for x in leaves if jp.issubdtype(x.dtype, jp.inexact)})
    if not before or not set(before).issubset({"bfloat16", "float16", "float32"}):
        raise ValueError(f"cannot promise value-preserving FP32 widening of this actor: {before}")
    # The state belongs only to this separate process; the reference/sampler
    # are never called. No global dtype changes, model rewrite or file save.
    with runtime.mesh, runtime.monitor(jax):
        promoted = jax.tree.map(
            lambda x: x.astype(jp.float32) if jp.issubdtype(x.dtype, jp.inexact) else x, state)
        jax.block_until_ready(promoted)
        nnx.update(runtime.actor, promoted)
    del state, leaves, promoted
    after = nnx.state(runtime.actor)
    floating = [x for x in jax.tree.leaves(after) if jp.issubdtype(x.dtype, jp.inexact)]
    if not floating or any(x.dtype != jp.float32 for x in floating):
        raise RuntimeError("not all floating actor state was promoted to FP32")
    # No new random initialization. Widening preserves the restored values;
    # LoRA is the only mutable parameter subset used by the +/- probes.
    runtime.actor.config = replace(runtime.actor.config, param_dtype=jp.float32)
    runtime._snapshot = ParameterSnapshot(nnx.state(runtime.actor, nnx.LoRAParam))
    runtime.initial_summary = runtime.parameter_summary(jax, runtime.snapshot)
    policy_config = replace(runtime.algo, beta=0.)

    def loss(model, example):
        value = grpo_loss_fn(model, example, policy_config, runtime.pad_id, runtime.eos_id)[0]
        if value.dtype != jp.float32:
            raise TypeError("diagnostic mean-loss forward is not FP32")
        return value

    runtime.clean_loss = production_mean_loss(loss)
    # Numerics only: grad of the SAME scalar evaluated by clean_value/probe.
    # No 16 separate full parameter-gradient trees: no candidate/ranking test.
    runtime.control_mean_grad = nnx.jit(nnx.value_and_grad(
        runtime.clean_loss, argnums=nnx.DiffState(0, nnx.LoRAParam)))
    runtime.snapshot_memory("fp32_actor_promoted")
    return {"source_restored_lora": original_lora,
            "fp32_control_lora": runtime.initial_summary,
            "floating_actor_state_dtypes_before": before,
            "floating_actor_state_dtypes_after": ["float32"],
            "floating_actor_state_bytes": sum(x.size * x.dtype.itemsize for x in floating),
            "matmul_precision": "highest",
            "activation_contract": "FP32 floating actor state and input-following Gemma activations; logps/loss dtype checked",
            "gradient_contract": "gradient of SAME mean(vmap(per-unit loss)) scalar; not production per-unit feature audit"}


def control_gradient(runtime, heldout):
    """Two batch-mean backwards, same frozen 16-answer shapes, no candidate AD."""
    means = []
    for index, host in enumerate(heldout):
        print(f"FP32 heldout backward {index + 1}/{len(heldout)}", flush=True)
        with runtime.mesh, runtime.monitor(runtime.jax):
            value, gradient = runtime.control_mean_grad(runtime.actor, runtime.as_example(host))
            runtime.jax.block_until_ready((value, gradient))
        runtime.assert_original_parameters("fp32_after_backward")
        means.append(gradient)
        runtime.snapshot_memory(f"fp32_heldout_mean_backward_{index}")
    if len(means) != 2:
        raise ValueError("this bounded control requires exactly two heldout batches")
    with runtime.mesh:
        h = runtime.scale(runtime.add(*means), .5)
        norms = [float(runtime.jax.device_get(runtime.dot(x, x))) ** .5 for x in means]
        norm = float(runtime.jax.device_get(runtime.dot(h, h))) ** .5
        cosine = (float(runtime.jax.device_get(runtime.dot(*means))) / np.prod(norms)
                  if min(norms) > runtime.args.zero_epsilon else None)
    if not np.isfinite(norm) or norm <= runtime.args.zero_epsilon:
        raise ValueError("FP32 heldout gradient is zero/nonfinite: cannot validate a directional control")
    return h, {"l2_norm": norm, "units": 32,
               "split_half": {"prompts": [4, 4], "l2_norms": norms,
                              "cosine": float(np.clip(cosine, -1., 1.)) if cosine is not None else None},
               "stability_certified": False}


def run(args):
    from my_example.audit_grpo_checkpoint_restore import (
        fingerprint_directory, metadata_preflight, validate_paths, write_report,
    )

    source, output, checkpoint = validate_paths(args.input_dir, args.output_dir, args.step)
    replay, manifest = replay_metadata(args.replay_rollouts, output)
    repo = Path(__file__).resolve().parents[2]
    validate_replay_contract(manifest, args, repo)
    preflight = metadata_preflight(source, checkpoint, args.expected_commit)
    # clean_host imports the repository GRPO module (and therefore JAX).
    # Isolate BEFORE that import even when invoked without the shell launcher.
    os.environ.update(TPU_PROCESS_BOUNDS="1,1,1", TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1",
                      TPU_VISIBLE_CHIPS="0,1,2,3", CLOUD_TPU_TASK_ID="0",
                      WANDB_DISABLED="true", WANDB_MODE="disabled")
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ.pop("JAX_PLATFORM_NAME", None)
    inputs = {str(replay / "audit_manifest.json"): sha256_file(replay / "audit_manifest.json")}
    hosts = []
    indices = manifest["data_partition"]["heldout_indices"]
    for start in range(0, 8, 4):
        path = replay / "fixed_rollouts" / f"heldout_{start // 4:03d}.npz"
        hosts.append(clean_host(load_rollout(path, indices[start:start + 4])))
        inputs[str(path)] = sha256_file(path)
    identities = [str(x) for host in hosts for x in host["prompt_ids_hash"].reshape(4, 4)[:, 0]]
    if len(set(identities)) != 8:
        raise ValueError("duplicate heldout prompt identities")
    output.mkdir(parents=True, exist_ok=False)
    (output / "fp32_control_rollouts").mkdir()
    summary = {"execution_passed": False, "fp32_numerical_control_passed": False,
               "utility_audit_completed": False, "production_path_validated": False,
               "candidate_scan_completed": False, "method_benefit_inferred": False,
               "assumption_proven": False,
               "scope": "measurement control ONLY; FP32 precision AND old-policy contract change together",
               "warnings": ["Not a causal attribution to BF16 alone; no accuracy/long-term benefit measured.",
                            "Eight heldout prompts do not establish reliable task direction.",
                            "A pass does not repair/validate prior BF16 finite-step utility."]}
    started, runtime = time.monotonic(), None
    try:
        import jax
        from .utility_runtime import UtilityRuntime

        with jax.default_matmul_precision("highest"):
            runtime = UtilityRuntime(args, checkpoint, output)
            if runtime.initial_summary["sha256"] != manifest["restored_lora"]["sha256"]:
                raise ValueError("not the same restored LoRA theta0 as the archived BF16 run")
            if Path(runtime.model_path).name != Path(manifest["backbone_path"]).name:
                raise ValueError("not the same pinned backbone snapshot")
            protocol = configure_fp32(runtime)
            write_report(output / "control_manifest.json", {
                "arguments": vars(args), "source_checkpoint": str(checkpoint),
                "source_rollouts": str(replay), "source_files_sha256": inputs,
                "implementation_sha256": {str(p.relative_to(repo)): sha256_file(p) for p in (
                    Path(__file__).resolve(), repo / "my_example/rebuttal/utility_runtime.py",
                    repo / "my_example/rebuttal/parameter_snapshot.py", repo / "my_example/rebuttal/utility_math.py",
                    repo / "tunix/models/gemma3/model.py", repo / "tunix/rl/common.py")},
                "topology": runtime.topology, "precision": protocol,
                "no_generation": True, "no_candidate_gradients": True, "no_training": True,
                "old_policy": "NEW FP32 theta0 unit-forward logps in separate files, archived BF16 logps unchanged",
                "objective": "fixed clean rewards/A and tokens; no KL; diagnostic on-policy scalar"})
            heldout, comparisons = [], []
            for index, archived in enumerate(hosts):
                host = dict(archived)
                with runtime.mesh, runtime.monitor(jax):
                    current = np.asarray(jax.device_get(runtime.unit_logps(runtime.actor, runtime.as_example(host))))
                comparisons.append({"batch_id": index, "archived_dtype": str(
                    archived.get("old_per_token_logps_original_dtype", archived["old_per_token_logps"].dtype)),
                    "archived_BF16_vs_NEW_FP32": logp_difference(
                        current, archived["old_per_token_logps"], host["completion_mask"], args.epsilon)})
                host["old_per_token_logps"] = current.copy()
                host.pop("ref_per_token_logps", None)  # beta=0; no new reference model forward
                np.savez_compressed(output / "fp32_control_rollouts" / f"heldout_{index:03d}.npz",
                                    old_per_token_logps=current, source_indices=host["source_indices"],
                                    prompt_ids_hash=host["prompt_ids_hash"])
                heldout.append(host)
                runtime.snapshot_memory(f"fp32_matched_old_policy_{index}")
            write_report(output / "logp_comparison.json", comparisons)
            summary["forward_precision_check"] = check_forward_precision(runtime, heldout[0])
            runtime.assert_original_parameters("fp32_before_controls")
            h, summary["independent_gradient"] = control_gradient(runtime, heldout)
            baseline, repeat = runtime.clean_value(heldout), runtime.clean_value(heldout)
            report = run_pairs(runtime, heldout, h, baseline, repeat)
            summary["fp32_numerical_control_passed"] = report["passed"]
            summary["validated_step_sizes"] = report["validated_step_sizes"]
            summary["parameter_validation"] = runtime.assert_original_parameters("fp32_control_end")
            runtime.snapshot_memory("fp32_control_end")
            summary["hbm_headroom_verified"] = json.loads((output / "hbm_report.json").read_text())["headroom_verified"]
            summary["execution_passed"] = True
    finally:
        summary["source_checkpoint_unchanged"] = fingerprint_directory(checkpoint) == preflight["input_fingerprint"]
        summary["source_rollouts_unchanged"] = all(sha256_file(p) == digest for p, digest in inputs.items())
        summary["elapsed_seconds"] = time.monotonic() - started
        if not summary["source_checkpoint_unchanged"] or not summary["source_rollouts_unchanged"]:
            summary["execution_passed"] = False
        write_report(output / "control_summary.json", summary)
        print(json.dumps({"output": str(output), **summary}, indent=2), flush=True)
    if not summary["execution_passed"]:
        raise RuntimeError("control source files changed")
    return 0 if summary["fp32_numerical_control_passed"] and summary["hbm_headroom_verified"] else 2


def main(argv=None):
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
