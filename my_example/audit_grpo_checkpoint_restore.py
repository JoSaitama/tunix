"""Read-only historical checkpoint preflight and optional single-worker restore.

This is NOT a utility audit, training resume, gradient test or HBM stress test.
Metadata mode imports only the standard library and never initializes JAX.
Restore mode loads one backbone and one LoRA actor, without a trainer/optimizer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time


PACKAGES = ("jax", "jaxlib", "flax", "orbax-checkpoint", "qwix")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_report(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_paths(input_dir, report_dir, step):
    source = Path(input_dir).expanduser().resolve(strict=True)
    output = Path(report_dir).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("report directory must be separate from the source input tree")
    if step <= 0:
        raise ValueError("step must be positive")
    checkpoint = source / "checkpoints" / "actor" / str(step)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint step directory not found: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"use a fresh report directory: {output}")
    return source, output, checkpoint


def fingerprint_directory(path):
    """Hash every source file, including OCDBT shards; never change source files."""
    root = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    count = 0
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"source symlinks are not accepted: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        with item.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
        digest.update(b"\0")
        count += 1
    if not count:
        raise ValueError("checkpoint contains no files")
    return {"sha256": digest.hexdigest(), "files": count, "bytes": byte_count}


def recorded_versions(source):
    path = Path(source) / "pip-freeze.txt"
    versions = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "==" in line:
                package, version = line.split("==", 1)
                if package.lower() in PACKAGES:
                    versions[package.lower()] = version.strip()
    return versions


def metadata_preflight(source, checkpoint, expected_commit=None):
    step_metadata_path = checkpoint / "_CHECKPOINT_METADATA"
    params_path = checkpoint / "model_params"
    params_metadata_path = params_path / "_METADATA"
    if not step_metadata_path.is_file() or not params_metadata_path.is_file():
        raise ValueError("both step _CHECKPOINT_METADATA and model_params/_METADATA are required")
    step_metadata = read_json(step_metadata_path)
    params_metadata = read_json(params_metadata_path)
    if not isinstance(step_metadata, dict) or not isinstance(params_metadata, dict):
        raise ValueError("Orbax metadata must be JSON objects")
    commit_path = source / "git-commit.txt"
    commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.is_file() else None
    if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("git-commit.txt must contain one full commit hash")
    if expected_commit and commit != expected_commit:
        raise ValueError(f"archived commit mismatch: expected {expected_commit}, found {commit}")
    tree_metadata = params_metadata.get("tree_metadata")
    if tree_metadata is not None and not isinstance(tree_metadata, dict):
        raise ValueError("unexpected Orbax tree_metadata schema")
    # A metadata-only download must never be reported as a restored checkpoint.
    payload = [item for item in params_path.rglob("*")
               if item.is_file() and item.name not in {"_METADATA", "_sharding", "manifest.ocdbt"}]
    if not payload or not any(item.stat().st_size for item in payload):
        raise ValueError("no nonempty parameter payload found; download the complete step directory")
    versions = recorded_versions(source)
    current = {}
    for name in PACKAGES:
        try:
            current[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            current[name] = None
    mismatches = {name: {"recorded": version, "installed": current[name]}
                  for name, version in versions.items() if current[name] != version}
    return {
        "archived_commit": commit,
        "archived_branch": ((source / "git-branch.txt").read_text(encoding="utf-8").strip()
                            if (source / "git-branch.txt").is_file() else None),
        "recorded_python": ((source / "python-version.txt").read_text(encoding="utf-8").strip()
                            if (source / "python-version.txt").is_file() else None),
        "current_python": platform.python_version(),
        "recorded_package_versions": versions,
        "installed_package_versions": current,
        "package_version_differences": mismatches,
        "step_metadata_keys": sorted(step_metadata),
        "parameter_metadata_keys": sorted(params_metadata),
        "parameter_metadata_entry_count": None if tree_metadata is None else len(tree_metadata),
        "parameter_metadata_key_sample": [] if tree_metadata is None else sorted(tree_metadata)[:6],
        "parameter_payload_files": len(payload),
        "ocdbt_manifest_present": (params_path / "manifest.ocdbt").is_file(),
        "input_fingerprint": fingerprint_directory(checkpoint),
        "scope": "metadata and file presence only; payload validity requires actual restore",
    }


@contextmanager
def suppress_monitoring_scalars(jax):
    """Avoid historical Orbax/metric-listener WandB calls in this isolated process."""
    import jax._src.dispatch as dispatch
    import jax._src.monitoring as monitoring

    replaced = []
    for module in (jax.monitoring, monitoring, dispatch):
        if hasattr(module, "record_scalar"):
            replaced.append((module, module.record_scalar))
            module.record_scalar = lambda *args, **kwargs: None
    try:
        yield
    finally:
        for module, original in reversed(replaced):
            module.record_scalar = original


def parameter_summary(jax, state):
    import numpy as np

    digest = hashlib.sha256()
    leaves = jax.tree_util.tree_leaves(state)
    elements = 0
    zeros = 0
    squared_norm = 0.0
    shapes = []
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        # Device leaves are transferred one at a time; no flattened model copy.
        values = array.astype(np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("restored LoRA parameters contain NaN or Inf")
        shapes.append({"shape": list(array.shape), "dtype": str(array.dtype)})
        digest.update(str((array.shape, str(array.dtype))).encode("utf-8") + b"\0")
        digest.update(array.tobytes())
        elements += int(array.size)
        zeros += int(np.count_nonzero(values == 0))
        squared_norm += float(np.sum(values * values))
    if not leaves or not elements:
        raise ValueError("no LoRA parameter leaves found")
    return {"sha256": digest.hexdigest(), "leaf_count": len(leaves),
            "elements": elements, "zero_elements": zeros,
            "l2_norm": squared_norm ** .5, "leaves": shapes}


def restore_checkpoint(args, checkpoint, output, report):
    # Direct module invocations must be isolated too, before importing JAX.
    os.environ.update({"TPU_PROCESS_BOUNDS": "1,1,1",
                       "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
                       "TPU_VISIBLE_CHIPS": "0,1,2,3", "CLOUD_TPU_TASK_ID": "0",
                       "WANDB_DISABLED": "true", "WANDB_MODE": "disabled"})
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ.pop("JAX_PLATFORM_NAME", None)
    import jax
    from flax import nnx
    from orbax import checkpoint as ocp
    from .config import LoraConfig, ModelConfig
    from .model import apply_lora, download_model, load_model, resolve_model_config
    from .sharding import make_mesh

    devices = jax.local_devices()
    if jax.process_count() != 1 or jax.device_count() != 4 or len(devices) != 4 or any(
        device.platform != "tpu" for device in devices
    ):
        raise ValueError("restore requires one process and four local TPU devices; no multiworker run")
    report["topology"] = {"process_count": jax.process_count(),
                          "process_index": jax.process_index(),
                          "device_count": jax.device_count(),
                          "local_device_count": len(devices),
                          "devices": [str(device) for device in devices]}
    write_report(output / "restore_report.json", report)
    model_path = (str(Path(args.backbone_path).expanduser().resolve(strict=True))
                  if args.backbone_path else download_model(ModelConfig(model_id=args.model_id)))
    report["model"] = {"model_id": args.model_id, "backbone_path": model_path,
                       "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                       "backbone_revision_verified_against_historical_run": False}
    mesh, _ = make_mesh((4, 1))
    with mesh, suppress_monitoring_scalars(jax):
        base = load_model(model_path, resolve_model_config(args.model_id), mesh)
        actor = apply_lora(base, LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha), mesh)
        initial_state = nnx.state(actor, nnx.LoRAParam)
        initial_summary = parameter_summary(jax, initial_state)
        target = jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding),
            initial_state,
        )
        checkpointer = ocp.StandardCheckpointer()
        try:
            restored = checkpointer.restore(str(checkpoint / "model_params"), target=target)
        finally:
            checkpointer.close()
        # Validate that the abstract tree actually matches before applying it.
        if jax.tree_util.tree_structure(initial_state) != jax.tree_util.tree_structure(restored):
            raise ValueError("restored LoRA tree does not match the current model")
        for expected, actual in zip(jax.tree.leaves(initial_state), jax.tree.leaves(restored)):
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise ValueError("restored LoRA shape/dtype mismatch")
        nnx.update(actor, restored)
        actual_summary = parameter_summary(jax, nnx.state(actor, nnx.LoRAParam))
        if parameter_summary(jax, restored)["sha256"] != actual_summary["sha256"]:
            raise ValueError("nnx.update did not reproduce restored checkpoint parameters")
        report["parameters"] = actual_summary
        report["differs_from_newly_initialized_lora"] = (
            initial_summary["sha256"] != actual_summary["sha256"])
        if not report["differs_from_newly_initialized_lora"]:
            report["warnings"].append("checkpoint equals initial LoRA; investigate before utility analysis")
        jax.effects_barrier()
        stats_report = []
        headrooms = []
        for device in devices:
            try:
                stats = device.memory_stats()
            except (NotImplementedError, RuntimeError):
                stats = None
            stats = None if stats is None else {key: int(value) for key, value in stats.items()}
            stats_report.append({"device": str(device), "stats": stats})
            if stats and stats.get("bytes_limit") and "peak_bytes_in_use" in stats:
                headrooms.append(1 - stats["peak_bytes_in_use"] / stats["bytes_limit"])
        report["restore_memory"] = {"devices": stats_report, "peak_headroom_fractions": headrooms,
                                     "headroom_verified": len(headrooms) == 4,
                                     "scope": "model loading/restoration only, NOT gradient/rollout peak HBM"}
        write_report(output / "restore_report.json", report)
        if headrooms and min(headrooms) < args.min_headroom_fraction:
            raise RuntimeError("restoration peak memory leaves less than requested headroom")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--mode", choices=("metadata", "restore"), default="metadata")
    parser.add_argument("--expected-commit")
    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument("--backbone-path", help="Optional existing local HF snapshot; avoids download lookup")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=64.)
    parser.add_argument("--min-headroom-fraction", type=float, default=.05)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.lora_rank <= 0 or not 0 < args.lora_alpha < float("inf"):
        raise ValueError("LoRA rank/alpha must be finite and positive")
    if not 0 <= args.min_headroom_fraction < 1:
        raise ValueError("min headroom must be in [0, 1)")
    source, output, checkpoint = validate_paths(args.input_dir, args.report_dir, args.step)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"passed": False, "mode": args.mode, "source": str(source),
              "checkpoint": str(checkpoint), "step": args.step, "warnings": [],
              "scope": "checkpoint preflight/restore only; no gradients, utility or optimizer updates",
              "optimizer_restored": False, "utility_audit_completed": False}
    report_path = output / "restore_report.json"
    try:
        report["preflight"] = metadata_preflight(source, checkpoint, args.expected_commit)
        write_report(report_path, report)
        if args.mode == "restore":
            restore_checkpoint(args, checkpoint, output, report)
        after = fingerprint_directory(checkpoint)
        report["source_unchanged"] = after == report["preflight"]["input_fingerprint"]
        if not report["source_unchanged"]:
            raise RuntimeError("checkpoint source files changed during this audit")
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(report_path, report)
        print(json.dumps({"report": str(report_path), "passed": report["passed"],
                          "mode": args.mode, "utility_audit_completed": False}, indent=2))


if __name__ == "__main__":
    main()
