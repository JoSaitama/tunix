"""Disposable current-worker norm-top50 HBM test, not a scientific result.

Sequentially prepare/update: one real generation and two synthetic all-valid
256+768 updates, so asynchronous learner prefetch cannot mislabel stress phases.
All phases use controlled rewards to ensure policy AD is exercised independently
of sampled correctness. End-to-end production reward tests are separate smokes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


def initialize_direct_preparation(learner):
    """Mirror RLLearner.train's default batch-size initialization, test only.

    Direct sequential preparation intentionally bypasses its asynchronous
    training loop, NOT its required four-prompt service-size defaults.
    Refuse incompatible explicit settings rather than silently resizing them.
    """
    config = learner._training_config
    mini = config.mini_batch_size or 4
    train = config.train_micro_batch_size or mini
    rollout = learner._rollout_micro_batch_size or train
    logps = learner._compute_logps_micro_batch_size or train
    if learner.algo_config.num_generations != 4 or (mini, train, rollout, logps) != (4, 4, 4, 4):
        raise ValueError("sequential HBM fixture requires four prompts/four generations in every service batch")
    learner._rollout_micro_batch_size = rollout
    learner._compute_logps_micro_batch_size = logps
    return {"train_prompt_micro_batch_size": train, "rollout_prompt_micro_batch_size": rollout,
            "logps_prompt_micro_batch_size": logps, "service_completions": rollout * 4,
            "initialization": "same fallback defaults as RLLearner.train, no training/parameter change"}


def validate_records(records):
    if len(records) != 3 or any(len(row["groups"]) != 4 for row in records):
        raise RuntimeError("expected three recorded updates, each with four prompt groups")
    for phase, row in enumerate(records):
        for group in row["groups"]:
            values = np.asarray(group["policy_gradient_squared_l2_norms"], dtype=np.float64)
            expected = np.sort(np.argsort(-values, kind="stable")[:2]).tolist()
            if len(values) != 4 or not np.isfinite(values).all() or group["selected_local_indices"] != expected:
                raise RuntimeError("HBM update did not select exactly the two largest norms per prompt")
        if phase and not any(v > 0 for group in row["groups"] for v in group["policy_gradient_l2_norms"]):
            raise RuntimeError("synthetic stress did not exercise nonzero policy gradients")
        if row["advantages_recomputed"] or row["score_objective"] != "policy" or row["mask_application"] != "full":
            raise RuntimeError("HBM method contract changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--backbone-path")
    parser.add_argument("--min-headroom-fraction", type=float, default=.05)
    args = parser.parse_args()
    if not .05 <= args.min_headroom_fraction < 1:
        parser.error("require at least 5% observed HBM headroom")
    os.environ.update(TPU_PROCESS_BOUNDS="1,1,1", TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1",
                      TPU_VISIBLE_CHIPS="0,1,2,3", CLOUD_TPU_TASK_ID="0",
                      TUNIX_DBC_VARIANT="gradient_norm_top50", TUNIX_GRPO_NUM_GENERATIONS="4",
                      TUNIX_DYNAMIC_SAMPLING="0", TUNIX_PODS_KEEP_COUNT="",
                      WANDB_DISABLED="true", WANDB_MODE="disabled")
    os.environ.pop("JAX_PLATFORMS", None)
    os.environ.pop("JAX_PLATFORM_NAME", None)
    output = Path(args.report_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["TUNIX_NORM_ARTIFACT_DIR"] = str(output / "selection")
    report = {"passed": False, "headroom_verified": False, "phases": [],
              "scope": "one real generation + two all-valid full-length updates; controlled rewards in ALL phases",
              "caveat": "observed execution peaks only, not a guarantee for arbitrary future shapes"}

    def save():
        (output / "hbm_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        import jax
        import jax.numpy as jnp
        from tunix.generate import tokenizer_adapter
        from tunix.rl import rl_cluster as cluster_lib
        from my_example.config import config_from_args
        from my_example.deps import assert_dependencies
        from my_example.model import apply_lora, download_model, load_eos_tokens, load_model, resolve_model_config
        from my_example.prompts import SYSTEM_PROMPT, TEMPLATE
        from my_example.sharding import make_mesh
        from my_example.train import build_cluster_config, build_optimizer, build_trainer
        from .gradient_norm_trainer import GradientNormTop50Trainer

        assert_dependencies()
        if jax.process_count() != 1 or jax.device_count() != 4 or jax.local_device_count() != 4 or any(
                d.platform != "tpu" for d in jax.local_devices()):
            raise ValueError("norm HBM requires one process and four local TPU devices")
        report["topology"] = {"process_count": jax.process_count(), "process_index": jax.process_index(),
                              "device_count": jax.device_count(), "local_device_count": jax.local_device_count(),
                              "devices": [str(d) for d in jax.local_devices()]}
        cfg = config_from_args(["--num-generations", "4", "--train-micro-batch-size", "4",
            "--learning-rate", "1e-6", "--max-prompt-length", "256", "--total-generation-steps", "768",
            "--mesh-counts", "4,1", "--no-wandb", "--use-dynamic-batch-curation",
            "--metrics-log-dir", str(output / "tensorboard"), "--checkpoint-root", str(output / "checkpoints"),
            "--save-interval-steps", "10000"])
        path = str(Path(args.backbone_path).resolve(strict=True)) if args.backbone_path else download_model(cfg.model)
        mesh, _ = make_mesh(cfg.runtime.mesh_counts)
        base = load_model(path, resolve_model_config(cfg.model.model_id), mesh)
        actor = apply_lora(base, cfg.lora, mesh)
        tokenizer = tokenizer_adapter.Tokenizer(tokenizer_path=cfg.model.tokenizer_path)
        eos = load_eos_tokens(path)
        if tokenizer.eos_id() not in eos:
            eos.append(tokenizer.eos_id())
        cluster = cluster_lib.RLCluster(actor=actor, reference=base, tokenizer=tokenizer,
            cluster_config=build_cluster_config(mesh, cfg.grpo, cfg.eval, cfg.training,
                build_optimizer(cfg.training, 691), 691, 4, eos, False))
        learner = build_trainer(cluster, cfg.grpo)
        if not isinstance(cluster.actor_trainer, GradientNormTop50Trainer):
            raise RuntimeError("norm actor-trainer routing did not activate")
        preparation = initialize_direct_preparation(learner)
        prompts = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=f"What is {i}+1?") for i in range(4)]
        training_input = {"prompts": np.repeat(prompts, 4), "answer": np.repeat([str(i+1) for i in range(4)], 4)}
        valid = np.array([i for i in range(10, 100) if i not in eos and i != tokenizer.pad_id()])

        def synthetic(*, prompts, **unused):
            n = len(prompts)
            if n != 16:
                raise ValueError("HBM generation must stay at four prompts x four completions")
            tokens = valid[(np.arange(768)[None, :] + np.arange(n)[:, None]) % len(valid)].astype(np.int32)
            tokens[:, -1] = tokenizer.eos_id()
            return SimpleNamespace(tokens=jnp.asarray(tokens), text=["synthetic"] * n,
                left_padded_prompt_tokens=jnp.full((n, 256), int(valid[0]), dtype=jnp.int32))

        def rewards(prompts, completions, **unused):
            if len(prompts) != 16 or len(completions) != 16:
                raise ValueError("HBM reward fixture expects exactly16 completions")
            return np.tile([0., .5, 3., 4.], 4)

        report["configuration"] = {"prompts": 4, "num_generations": 4, "selected_per_prompt": 2,
            "backward_completions": 16, "prompt_length": 256, "stress_completion_length": 768,
            "beta": cfg.grpo.beta, "lr_schedule_budget": 691, "controlled_rewards_in_all_phases": True,
            "direct_preparation": preparation}
        for index in range(3):
            name = "real_generation_and_actor_update" if index == 0 else f"full_length_actor_update_{index}"
            print(f"Norm HBM phase: {name}", flush=True)
            generation = mock.patch.object(cluster, "generate", side_effect=synthetic) if index else mock.patch.dict(os.environ, {})
            with generation, mock.patch.object(learner, "reward_fns", [rewards]), mesh:
                example = learner._generate_and_compute_advantage(dict(training_input), mode=cluster_lib.Mode.TRAIN)
                if len(example.advantages) != 16:
                    raise RuntimeError("norm test unexpectedly shrank the actor input")
                if index and (example.completion_ids.shape != (16, 768) or
                              not np.all(np.asarray(example.completion_mask).sum(1) == 768)):
                    raise RuntimeError("synthetic HBM input is not full-length/all-valid")
                # Sequential real actor.train path, not a toy optimizer or a
                # prefetch-dependent learner callback. Same 691-step schedule.
                cluster.update_actor(iter([example]), eval_ds=None)
                jax.effects_barrier()
            devices, headroom = [], []
            for device in jax.local_devices():
                stats = device.memory_stats()
                stats = None if stats is None else {k: int(v) for k, v in stats.items()}
                devices.append({"device": str(device), "stats": stats})
                if stats and stats.get("bytes_limit") and "peak_bytes_in_use" in stats:
                    headroom.append(1. - stats["peak_bytes_in_use"] / stats["bytes_limit"])
            report["phases"].append({"name": name, "devices": devices, "headroom": headroom,
                                     "actor_train_steps": int(cluster.actor_trainer.train_steps)})
            save()
            if len(headroom) != 4 or min(headroom) < args.min_headroom_fraction:
                raise RuntimeError("norm HBM peak headroom unavailable or below safety margin")
        records = [json.loads(line) for line in (output / "selection/norm_selection.jsonl").read_text().splitlines()]
        validate_records(records)
        if cluster.actor_trainer.train_steps != 3:
            raise RuntimeError("expected three real actor updates")
        report.update(passed=True, headroom_verified=True)
        save()
        print(f"PASS: {output / 'hbm_report.json'}", flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
