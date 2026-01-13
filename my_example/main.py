from __future__ import annotations

from contextlib import contextmanager
import os
import shutil

from flax import nnx
import jax
from orbax import checkpoint as ocp
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.rl import rl_cluster as rl_cluster_lib

from .auth import ensure_kaggle_login, maybe_init_wandb
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
    resolve_model_config,
    save_merged_lora,
)
from .sharding import make_mesh
from .train import build_cluster_config, build_optimizer, build_trainer


def build_sampler(lora_policy, tokenizer, model_config, max_prompt_length, total_steps, eos_tokens):
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


def main() -> None:
    # === Multi-host init (required for v4-32 and other multi-host TPU slices) ===
    import socket
    if "JAX_COORDINATOR_ADDRESS" in os.environ:
        hostname = socket.gethostname()
        pid = int(hostname.split("-w-")[-1])
        coord_addr = os.environ["JAX_COORDINATOR_ADDRESS"]
        num_procs = int(os.environ.get("JAX_PROCESS_COUNT", 1))
        print(f"[DEBUG] hostname={hostname}, pid={pid}, coord={coord_addr}, num_procs={num_procs}")
        jax.distributed.initialize(
            coordinator_address=coord_addr,
            num_processes=num_procs,
            process_id=pid,
        )
        print(f"[DEBUG] After init: process_index={jax.process_index()}, process_count={jax.process_count()}, device_count={jax.device_count()}, local_device_count={jax.local_device_count()}")
    else:
        print("[DEBUG] JAX_COORDINATOR_ADDRESS not set, skipping distributed init")
    # === End multi-host init ===

    try:
        jax.monitoring.clear_event_listeners()
    except Exception:
        pass

    assert_dependencies()
    cfg = config_from_args()
    if not cfg.runtime.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    if cfg.data.source == "kaggle":
        ensure_kaggle_login()

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
        get_dataset(cfg.data.train_data_dir, "train", cfg.data.source),
        train_micro_batch_size,
        max_train_examples,
    )
    test_dataset = batch_dataset(
        get_dataset(cfg.data.test_data_dir, "test", cfg.data.source),
        cfg.data.test_micro_batch_size,
        max_eval_examples,
    )

    def _safe_len(dataset):
        try:
            return len(dataset)
        except Exception:
            return None

    train_len = _safe_len(full_train_dataset)
    test_len = _safe_len(test_dataset)

    if cfg.data.train_fraction == 1.0:
        train_dataset = full_train_dataset.repeat(cfg.data.num_epochs)
        val_dataset = None
    else:
        train_dataset = full_train_dataset[: int(len(full_train_dataset) * cfg.data.train_fraction)]
        train_dataset = train_dataset.repeat(cfg.data.num_epochs)
        val_dataset = full_train_dataset[int(len(full_train_dataset) * cfg.data.train_fraction) :].repeat(
            cfg.data.num_epochs
        )

    train_split_len = _safe_len(train_dataset)
    val_split_len = _safe_len(val_dataset) if val_dataset is not None else None

    max_steps = requested_max_steps
    if train_split_len is not None:
        max_steps = int(train_split_len * cfg.grpo.num_iterations)

    print(
        "Config summary:",
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

    model_path = download_model(cfg.model)
    model_config = resolve_model_config(cfg.model.model_id)

    mesh, mesh_counts = make_mesh(cfg.runtime.mesh_counts)
    print(f"Using mesh counts: {mesh_counts}")

    gemma3 = load_model(model_path, model_config, mesh)
    lora_policy = apply_lora(gemma3, cfg.lora, mesh=mesh)

    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=cfg.model.tokenizer_path)
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
        num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
            test_dataset,
            sampler,
            temperature=None,
            top_k=1,
            top_p=None,
            num_passes=cfg.runtime.eval_num_passes,
            verbose=cfg.runtime.verbose_eval,
        )
        print(
            f"pre-train: num_correct={num_correct}, total={total}, "
            f"accuracy={accuracy}%, partial_accuracy={partial_accuracy}%, "
            f"format_accuracy={format_accuracy}%"
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
        reference=gemma3,
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

        # Clear monitoring listeners to prevent WandB errors during restore/eval
        try:
            jax.monitoring.clear_event_listeners()
        except Exception:
            pass

        if cfg.training.checkpoint_root_directory:
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
                        # Retry once if needed
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
                print(
                    f"Checkpoint not found at {trained_ckpt_path}. Skipping restore."
                )
        else:
            print("Checkpointing disabled. Skipping restore.")

        if cfg.runtime.eval_after_train:
            with _suppress_jax_monitoring():
                sampler = build_sampler(
                    lora_policy,
                    tokenizer,
                    model_config,
                    cfg.grpo.max_prompt_length,
                    cfg.grpo.total_generation_steps,
                    eos_tokens,
                )
                num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
                    test_dataset,
                    sampler,
                    temperature=None,
                    top_k=1,
                    top_p=None,
                    num_passes=cfg.runtime.eval_num_passes,
                    verbose=cfg.runtime.verbose_eval,
                )
                print(
                    f"post-train: num_correct={num_correct}, total={total}, "
                    f"accuracy={accuracy}%, partial_accuracy={partial_accuracy}%, "
                    f"format_accuracy={format_accuracy}%"
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
            size_mb = os.path.getsize(os.path.join(output_dir, file_name)) / (1024 * 1024)
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
