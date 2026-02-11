from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
from typing import Optional, Tuple

from flax import nnx
import jax
from orbax import checkpoint as ocp
from tunix.generate import sampler as sampler_lib
from tunix.rl import rl_cluster as rl_cluster_lib

from my_example.sharding import make_mesh

from .auth import maybe_init_wandb
from .config import config_from_args
from .data import batch_dataset, get_dataset
from .deps import assert_dependencies
from .eval import evaluate
from .generate import SamplerWrapper
from .model import (
    apply_lora,
    download_model,
    load_eos_tokens,
    load_model,
    load_tokenizer,
    resolve_model_config,
    save_merged_lora,
)
from .train import build_cluster_config, build_optimizer, build_trainer


def build_sampler(
    lora_policy,
    tokenizer,
    model_config,
    max_prompt_length,
    total_steps,
    eos_tokens,
):
    cache_config = sampler_lib.CacheConfig(
        cache_size=max_prompt_length + total_steps + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    )
    return SamplerWrapper(
        transformer=lora_policy,
        tokenizer=tokenizer,
        cache_config=cache_config,
        eos_tokens=eos_tokens,
    )


@contextmanager
def _suppress_jax_monitoring():
    try:
        import jax._src.dispatch as jax_dispatch
        import jax._src.monitoring as jax_monitoring
    except Exception:
        yield
        return

    def _noop_record_scalar(*_args, **_kwargs):
        return None

    originals = {}
    for module, attr in (
        (jax_monitoring, "record_scalar"),
        (jax_dispatch, "record_scalar"),
        (jax.monitoring, "record_scalar"),
    ):
        try:
            originals[(module, attr)] = getattr(module, attr)
            setattr(module, attr, _noop_record_scalar)
        except Exception:
            pass

    try:
        yield
    finally:
        for (module, attr), original in originals.items():
            try:
                setattr(module, attr, original)
            except Exception:
                pass


def _safe_len(dataset):
    try:
        return len(dataset)
    except Exception:
        return None


def _choose_mesh_counts(
    requested_counts: Optional[Tuple[int, int]],
    num_devices: int,
    num_heads: int,
    num_kv_heads: int,
) -> Tuple[int, int]:
    if requested_counts is not None:
        fsdp, tp = requested_counts
        if fsdp * tp != num_devices:
            raise ValueError(
                "Invalid --mesh-counts. "
                f"Got fsdp*tp={fsdp * tp}, expected {num_devices} devices."
            )
        if num_kv_heads % tp != 0:
            # Keep a working path instead of failing later in device_put.
            divisors = [d for d in range(1, num_devices + 1) if num_devices % d == 0]
            valid_tps = [d for d in divisors if num_kv_heads % d == 0 and num_heads % d == 0]
            if not valid_tps:
                raise ValueError(
                    f"No valid tp found for num_devices={num_devices}, "
                    f"num_heads={num_heads}, num_kv_heads={num_kv_heads}."
                )
            fixed_tp = max(valid_tps)
            fixed_fsdp = num_devices // fixed_tp
            print(
                "Adjusted mesh-counts to satisfy sharding divisibility: "
                f"requested=({fsdp}, {tp}) -> effective=({fixed_fsdp}, {fixed_tp})."
            )
            return (fixed_fsdp, fixed_tp)
        return requested_counts

    divisors = [d for d in range(1, num_devices + 1) if num_devices % d == 0]
    valid_tps = [d for d in divisors if num_kv_heads % d == 0 and num_heads % d == 0]
    if not valid_tps:
        return (1, 1)
    tp = max(valid_tps)
    fsdp = num_devices // tp
    return (fsdp, tp)


def main() -> None:
    try:
        jax.monitoring.clear_event_listeners()
    except Exception:
        pass

    assert_dependencies()
    cfg = config_from_args()
    if not cfg.runtime.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    model_path = download_model(cfg.model)
    model_config = resolve_model_config(cfg.model.model_id)
    tokenizer = load_tokenizer(cfg.model)

    train_micro_batch_size = int(cfg.data.train_micro_batch_size)
    max_train_examples = int(cfg.data.max_train_examples)
    max_eval_examples = int(cfg.data.max_eval_examples)
    max_train_batches = max_train_examples // train_micro_batch_size
    requested_max_steps = int(
        max_train_batches
        * cfg.grpo.num_iterations
        * cfg.data.train_fraction
        * cfg.data.num_epochs
    )

    full_train_dataset = batch_dataset(
        get_dataset(cfg.data.train_data_path, tokenizer),
        train_micro_batch_size,
        max_train_examples,
    )
    test_dataset = batch_dataset(
        get_dataset(cfg.data.test_data_path, tokenizer),
        cfg.data.test_micro_batch_size,
        max_eval_examples,
    )

    train_len = _safe_len(full_train_dataset)
    test_len = _safe_len(test_dataset)

    if cfg.data.train_fraction == 1.0:
        train_dataset = full_train_dataset.repeat(cfg.data.num_epochs)
        val_dataset = None
    else:
        split_idx = int(len(full_train_dataset) * cfg.data.train_fraction)
        train_dataset = full_train_dataset[:split_idx].repeat(cfg.data.num_epochs)
        val_dataset = full_train_dataset[split_idx:].repeat(cfg.data.num_epochs)

    train_split_len = _safe_len(train_dataset)
    val_split_len = _safe_len(val_dataset) if val_dataset is not None else None

    max_steps = requested_max_steps
    if train_split_len is not None:
        max_steps = int(train_split_len * cfg.grpo.num_iterations)

    print(
        "Config summary:",
        f"model_id={cfg.model.model_id}",
        f"train_data_path={cfg.data.train_data_path}",
        f"test_data_path={cfg.data.test_data_path}",
        f"max_train_examples={cfg.data.max_train_examples}",
        f"train_fraction={cfg.data.train_fraction}",
        f"num_epochs={cfg.data.num_epochs}",
        f"num_iterations={cfg.grpo.num_iterations}",
        f"max_steps={max_steps}",
        f"train_micro_batch_size={cfg.data.train_micro_batch_size}",
        f"test_micro_batch_size={cfg.data.test_micro_batch_size}",
        f"max_eval_examples={cfg.data.max_eval_examples}",
        f"eval_before_train={cfg.runtime.eval_before_train}",
        f"eval_after_train={cfg.runtime.eval_after_train}",
        sep=" | ",
    )
    if max_steps != requested_max_steps:
        print(
            "Adjusted max_steps to match available training batches: "
            f"requested={requested_max_steps}, effective={max_steps}."
        )
    if train_len is not None or test_len is not None:
        print(f"Dataset sizes: train_batches={train_len}, test_batches={test_len}")
    if train_split_len is not None or val_split_len is not None:
        print(f"Split sizes: train={train_split_len}, val={val_split_len}")

    mesh_counts = _choose_mesh_counts(
        cfg.runtime.mesh_counts,
        len(jax.devices()),
        model_config.num_heads,
        model_config.num_kv_heads,
    )
    mesh, mesh_counts = make_mesh(mesh_counts)
    print(f"Using mesh counts: {mesh_counts}")

    qwen2 = load_model(model_path, model_config, mesh)
    lora_policy = apply_lora(qwen2, cfg.lora, mesh=mesh)

    eos_tokens = load_eos_tokens(model_path)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
        print(f"Using EOS token IDs: {eos_tokens}")

    sampler = build_sampler(
        lora_policy,
        tokenizer,
        model_config,
        cfg.grpo.max_prompt_length,
        cfg.grpo.total_generation_steps,
        eos_tokens,
    )

    if cfg.runtime.eval_before_train:
        num_correct, total, accuracy = evaluate(
            test_dataset,
            sampler,
            temperature=cfg.eval.temperature,
            top_k=cfg.eval.top_k,
            top_p=cfg.eval.top_p,
            num_passes=cfg.runtime.eval_num_passes,
            verbose=cfg.runtime.verbose_eval,
        )
        print(
            f"pre-train: num_correct={num_correct}, total={total}, "
            f"accuracy={accuracy}%"
        )

    optimizer = build_optimizer(cfg.training, max_steps)
    cluster_config = build_cluster_config(
        mesh,
        cfg.grpo,
        cfg.eval,
        cfg.training,
        optimizer,
        max_steps,
        train_micro_batch_size,
        eos_tokens,
        cfg.runtime.use_wandb,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=lora_policy,
        reference=qwen2,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    trainer = build_trainer(rl_cluster, cfg.grpo)
    maybe_init_wandb(cfg.runtime.use_wandb)

    try:
        print("Starting training...")
        with mesh:
            trainer.train(train_dataset, val_dataset)
        print("Training complete.")

        try:
            jax.monitoring.clear_event_listeners()
        except Exception:
            pass

        trained_ckpt_path = os.path.join(
            cfg.training.checkpoint_root_directory,
            "actor",
            str(max_steps),
            "model_params",
        )

        if os.path.exists(trained_ckpt_path):
            abs_params = jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                nnx.state(lora_policy, nnx.LoRAParam),
            )
            checkpointer = ocp.StandardCheckpointer()
            with _suppress_jax_monitoring():
                try:
                    trained_lora_params = checkpointer.restore(
                        trained_ckpt_path, target=abs_params
                    )
                except Exception:
                    trained_lora_params = checkpointer.restore(
                        trained_ckpt_path, target=abs_params
                    )

            nnx.update(
                lora_policy,
                jax.tree.map(
                    lambda _a, b: b,
                    nnx.state(lora_policy, nnx.LoRAParam),
                    trained_lora_params,
                ),
            )
        else:
            print(f"Checkpoint not found at {trained_ckpt_path}. Skipping restore.")

        if cfg.runtime.eval_after_train:
            with _suppress_jax_monitoring():
                sampler = build_sampler(
                    lora_policy,
                    tokenizer,
                    model_config,
                    cfg.eval.max_prompt_length,
                    (
                        cfg.eval.total_generation_steps
                        if cfg.eval.total_generation_steps is not None
                        else cfg.grpo.total_generation_steps
                    ),
                    eos_tokens,
                )
                num_correct, total, accuracy = (
                    evaluate(
                        test_dataset,
                        sampler,
                        temperature=cfg.eval.temperature,
                        top_k=cfg.eval.top_k,
                        top_p=cfg.eval.top_p,
                        num_passes=cfg.runtime.eval_num_passes,
                        verbose=cfg.runtime.verbose_eval,
                    )
                )
                print(
                    f"post-train: num_correct={num_correct}, total={total}, "
                    f"accuracy={accuracy}%"
                )

        output_dir = cfg.runtime.output_dir or f"./{cfg.model.model_id}-lora"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print(f"Saving merged LoRA model to {output_dir}")
        with _suppress_jax_monitoring():
            save_merged_lora(model_path, output_dir, lora_policy, cfg.lora)

        print("\n" + "=" * 60)
        print("Model saved successfully!")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        print("\nSaved files:")
        for file_name in os.listdir(output_dir):
            size_mb = os.path.getsize(os.path.join(output_dir, file_name)) / (
                1024 * 1024
            )
            print(f"  {file_name:<30} {size_mb:>10.2f} MB")
    finally:
        try:
            rl_cluster.close()
        except Exception:
            pass
        try:
            jax.monitoring.clear_event_listeners()
        except Exception:
            pass


if __name__ == "__main__":
    main()
