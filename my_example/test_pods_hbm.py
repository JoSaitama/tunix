"""Disposable TPU test: real generation, then full-length nonzero PODS updates.

Only this test replaces generation/rewards with synthetic stress inputs. The
production PODS path has no synthetic-input flag. No existing run is restored.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


def make_hbm_batch():
    """Use array leaves, like Grain batches consumed by RLLearner.

    JAX treats Python lists as pytree containers. A list of four strings would
    become four array leaves of length four under the learner's repeat map,
    rather than one flat array of sixteen prompts.
    """
    return {
        "prompts": np.asarray([f"What is {i}+1?" for i in range(4)]),
        "answer": np.asarray([str(i+1) for i in range(4)]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--keep-count", type=int, default=2)
    parser.add_argument("--min-headroom-fraction", type=float, default=.05)
    args = parser.parse_args()
    # Also isolate direct `python -m` invocations, before importing JAX/Flax.
    # Physical worker1 is logical process0 when it runs independently.
    os.environ.update({
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
        "TPU_VISIBLE_CHIPS": "0,1,2,3",
        "CLOUD_TPU_TASK_ID": "0",
    })
    if not 0 <= args.min_headroom_fraction < 1:
        parser.error("headroom fraction must lie in [0,1)")
    from .pods import validate_keep_count
    validate_keep_count(4, args.keep_count)
    output = Path(args.report_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "phases": [], "headroom_verified": False,
              "scope": "one real rollout/update plus two full-length synthetic updates",
              "caveat": "Measured execution does not guarantee all future allocator peaks."}

    def save():
        (output / "hbm_report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        import jax
        import jax.numpy as jnp
        from tunix.generate import tokenizer_adapter
        from tunix.rl import rl_cluster as cluster_lib
        from .config import config_from_args
        from .deps import assert_dependencies
        from .model import apply_lora, download_model, load_eos_tokens, load_model, resolve_model_config
        from .sharding import make_mesh
        from .train import build_cluster_config, build_optimizer, build_trainer

        assert_dependencies()
        if jax.process_count() != 1 or len(jax.local_devices()) != 4 or any(
            x.platform != "tpu" for x in jax.local_devices()
        ):
            raise ValueError("HBM stress requires one process / four local TPU devices")
        report["topology"] = {
            "process_count": jax.process_count(), "process_index": jax.process_index(),
            "device_count": jax.device_count(), "local_device_count": jax.local_device_count(),
            "devices": [str(x) for x in jax.local_devices()],
        }
        cfg = config_from_args([
            "--num-generations", "4", "--train-micro-batch-size", "4",
            "--learning-rate", "1e-6", "--max-prompt-length", "256",
            "--total-generation-steps", "768", "--mesh-counts", "4,1", "--no-wandb",
            "--metrics-log-dir", str(output / "tensorboard"),
            "--checkpoint-root", str(output / "checkpoints"), "--save-interval-steps", "10000",
        ])
        model_path = download_model(cfg.model)
        mesh, _ = make_mesh(cfg.runtime.mesh_counts)
        base = load_model(model_path, resolve_model_config(cfg.model.model_id), mesh)
        actor = apply_lora(base, cfg.lora, mesh=mesh)
        tokenizer = tokenizer_adapter.Tokenizer(tokenizer_path=cfg.model.tokenizer_path)
        eos = load_eos_tokens(model_path)
        if tokenizer.eos_id() not in eos:
            eos.append(tokenizer.eos_id())
        cluster = cluster_lib.RLCluster(
            actor=actor, reference=base, tokenizer=tokenizer,
            cluster_config=build_cluster_config(
                mesh, cfg.grpo, cfg.eval, cfg.training, build_optimizer(cfg.training, 691),
                691, 4, eos, False),
        )
        with mock.patch.dict(os.environ, {
            "TUNIX_PODS_KEEP_COUNT": str(args.keep_count),
            "TUNIX_PODS_ARTIFACT_DIR": str(output / "selection"),
        }):
            trainer = build_trainer(cluster, cfg.grpo)
        batch = make_hbm_batch()
        valid = np.array([i for i in range(10, 100) if i not in eos and i != cluster.rollout.pad_id()])

        def synthetic(*, prompts, **unused):
            size = len(prompts)
            tokens = valid[(np.arange(768)[None,:] + np.arange(size)[:,None]) % len(valid)].astype(np.int32)
            tokens[:, -1] = tokenizer.eos_id()
            return SimpleNamespace(
                tokens=jnp.asarray(tokens), text=["synthetic"]*size,
                left_padded_prompt_tokens=jnp.full((size, 256), int(valid[0]), dtype=jnp.int32),
            )

        def synthetic_rewards(prompts, completions, **unused):
            return np.tile([0., .5, 3., 4.], len(prompts)//4).tolist()

        def snapshot(name):
            jax.effects_barrier()
            devices, headrooms = [], []
            for device in jax.local_devices():
                try:
                    stats = device.memory_stats()
                except (RuntimeError, NotImplementedError):
                    stats = None
                stats = None if stats is None else {k: int(v) for k,v in stats.items()}
                devices.append({"device": str(device), "stats": stats})
                if stats and stats.get("bytes_limit") and "peak_bytes_in_use" in stats:
                    headrooms.append(1-stats["peak_bytes_in_use"]/stats["bytes_limit"])
            report["phases"].append({"name": name, "devices": devices, "headroom": headrooms})
            save()
            if headrooms and min(headrooms) < args.min_headroom_fraction:
                raise RuntimeError("PODS stress leaves less than requested peak HBM headroom")

        def batches():
            yield batch
            snapshot("real_generation_and_actor_update")
            with mock.patch.object(cluster, "generate", side_effect=synthetic), \
                 mock.patch.object(trainer, "reward_fns", [synthetic_rewards]):
                for i in range(2):
                    yield batch
                    snapshot(f"full_length_actor_update_{i}")

        report["configuration"] = {"n": 4, "m": args.keep_count, "prompts": 4,
                                   "prompt_length": 256, "completion_length": 768, "beta": cfg.grpo.beta}
        with mesh:
            trainer.train(batches())
        if cluster.actor_trainer.train_steps != 3:
            raise RuntimeError("expected three optimizer updates")
        records = [json.loads(x) for x in (output / "selection/pods_selection.jsonl").read_text().splitlines()]
        if len(records) != 12 or len(report["phases"]) != 3:
            raise RuntimeError("expected 4 real and 8 synthetic prompt records across three phases")
        if not all(any(abs(v) > 0 for v in r["selected_advantages"]) for r in records[4:]):
            raise RuntimeError("synthetic stress failed to exercise nonzero policy advantages")
        report["headroom_verified"] = all(len(x["headroom"]) == 4 for x in report["phases"])
        report["passed"] = True
        save()
        print(f"PASS: {output / 'hbm_report.json'}")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
