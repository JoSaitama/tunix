"""Independent LearnAlign/GradAlign entrypoint for frozen GSM8K GRPO runs."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from flax import nnx
import jax
from orbax import checkpoint as ocp
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.rl import rl_cluster as rl_cluster_lib

from .alignment_baselines.artifacts import SelectionArtifactWriter
from .alignment_baselines.config import parse_alignment_args
from .alignment_baselines.curriculum import (
    GradAlignCurriculum,
    LearnAlignCurriculum,
)
from .alignment_baselines.data_utils import batch_examples, unbatch
from .alignment_baselines.gradient_features import PromptGradientEstimator
from .auth import ensure_kaggle_login, maybe_init_wandb
from .config import config_from_args
from .data import batch_dataset, get_dataset
from .deps import assert_dependencies
from .eval import evaluate
from .main import _suppress_jax_monitoring, build_sampler
from .model import (
    apply_lora,
    download_model,
    load_eos_tokens,
    load_model,
    resolve_model_config,
    save_merged_lora,
)
from .reward_rank_noise import config_from_env as reward_rank_noise_config_from_env
from .seeding import experiment_seed, seed_summary
from .sharding import make_mesh
from .train import (
    build_cluster_config,
    build_grpo_config,
    build_optimizer,
    build_trainer,
)


def _batched(examples, batch_size: int):
    if not examples:
        return None
    return [
        batch_examples(examples[start : start + batch_size])
        for start in range(0, len(examples) - batch_size + 1, batch_size)
    ]


def _write_run_metadata(path: Path, *, alignment, cfg, max_steps: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    selector_noise = reward_rank_noise_config_from_env()
    adaptations = [
        "selection gradients are computed only in the actor LoRA space",
        "full gradients use a deterministic sparse-JL feature hash",
        (
            "selector base rewards use binary exact correctness; selected "
            "training prompts receive deterministic within-group rank reversal"
        ),
        "GradAlign held-out validation selector rewards remain clean",
        "actual updates retain the frozen dense reward and optional rank mismatch",
    ]
    if alignment.method == "learnalign":
        adaptations.append(
            "LearnAlign keeps four-prompt rollout generation but evaluates "
            "configured-rollout gradient features one prompt at a time to bound HBM"
        )
    metadata = {
        "alignment": alignment.to_dict(),
        "experiment_seed": experiment_seed(),
        "seed_summary": seed_summary(),
        "effective_max_steps": max_steps,
        "reward_rank_mismatch": {
            "fraction": selector_noise.fraction,
            "seed": selector_noise.seed,
            "training_update_scope": "dense_reward_training_groups",
            "selector_scope": (
                "all_learnalign_candidates_or_gradalign_candidates_only"
            ),
            "gradalign_validation": "clean_heldout_reference",
        },
        "frozen_update_configuration": {
            "train_micro_batch_size": cfg.data.train_micro_batch_size,
            "num_generations": cfg.grpo.num_generations,
            "learning_rate": cfg.training.learning_rate,
            "beta": cfg.grpo.beta,
            "epsilon": cfg.grpo.epsilon,
            "max_prompt_length": cfg.grpo.max_prompt_length,
            "total_generation_steps": cfg.grpo.total_generation_steps,
        },
        "adaptations": adaptations,
    }
    (path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    try:
        jax.monitoring.clear_event_listeners()
    except Exception:
        pass

    assert_dependencies()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    alignment, frozen_argv = parse_alignment_args(raw_argv)
    cfg = config_from_args(frozen_argv)
    alignment.validate(train_batch_size=cfg.data.train_micro_batch_size)

    if cfg.training.use_dynamic_batch_curation:
        raise ValueError(
            "LearnAlign/GradAlign are standalone baselines and cannot be combined "
            "with DTV/DTV-Loo/dynamic-batch-curation flags."
        )
    if not cfg.runtime.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    if cfg.data.source == "kaggle":
        ensure_kaggle_login()

    train_batch_size = int(cfg.data.train_micro_batch_size)
    full_train_dataset = batch_dataset(
        get_dataset(cfg.data.train_data_dir, "train", cfg.data.source),
        train_batch_size,
        cfg.data.max_train_examples,
    )
    all_examples = unbatch(full_train_dataset)
    full_batches = len(all_examples) // train_batch_size
    train_batches_per_epoch = int(full_batches * cfg.data.train_fraction)
    train_prompt_count = train_batches_per_epoch * train_batch_size
    training_examples = all_examples[:train_prompt_count]
    heldout_examples = all_examples[train_prompt_count:]
    frozen_max_steps = (
        train_batches_per_epoch * cfg.grpo.num_iterations * cfg.data.num_epochs
    )
    if not training_examples or frozen_max_steps <= 0:
        raise ValueError("the requested dataset split has no training updates")

    if alignment.method == "gradalign":
        if len(heldout_examples) < alignment.gradalign_validation_prompts:
            raise ValueError(
                "held-out training split is too small for GradAlign validation: "
                f"need {alignment.gradalign_validation_prompts}, "
                f"have {len(heldout_examples)}"
            )
        effective_max_steps = frozen_max_steps
    else:
        effective_warmup_prompts = min(
            alignment.learnalign_warmup_prompts,
            len(training_examples),
        )
        warmup_steps = (
            effective_warmup_prompts + train_batch_size - 1
        ) // train_batch_size
        effective_max_steps = frozen_max_steps
        if not alignment.learnalign_warmup_counts_toward_budget:
            effective_max_steps += warmup_steps

    test_dataset = batch_dataset(
        get_dataset(cfg.data.test_data_dir, "test", cfg.data.source),
        cfg.data.test_micro_batch_size,
        cfg.data.max_eval_examples,
    )
    val_dataset = _batched(heldout_examples, train_batch_size)

    print(
        "Alignment baseline:",
        f"method={alignment.method}",
        f"train_prompts={len(training_examples)}",
        f"heldout_prompts={len(heldout_examples)}",
        f"frozen_max_steps={frozen_max_steps}",
        f"effective_max_steps={effective_max_steps}",
        f"seed={seed_summary()}",
        sep=" | ",
    )

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

    sampler = build_sampler(
        lora_policy,
        tokenizer,
        model_config,
        cfg.grpo.max_prompt_length,
        cfg.grpo.total_generation_steps,
        eos_tokens,
    )
    if cfg.runtime.eval_before_train:
        result = evaluate(
            test_dataset,
            sampler,
            temperature=None,
            top_k=1,
            top_p=None,
            num_passes=cfg.runtime.eval_num_passes,
            verbose=cfg.runtime.verbose_eval,
        )
        print(
            f"pre-train: num_correct={result[0]}, total={result[1]}, "
            f"accuracy={result[2]}%, partial_accuracy={result[3]}%, "
            f"format_accuracy={result[4]}%"
        )

    optimizer = build_optimizer(cfg.training, effective_max_steps)
    cluster_config = build_cluster_config(
        mesh,
        cfg.grpo,
        cfg.eval,
        cfg.training,
        optimizer,
        effective_max_steps,
        train_batch_size,
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

    run_artifact_dir = Path(alignment.artifact_dir)
    writer = SelectionArtifactWriter(str(run_artifact_dir), alignment.method)
    _write_run_metadata(
        run_artifact_dir,
        alignment=alignment,
        cfg=cfg,
        max_steps=effective_max_steps,
    )
    selector_noise_config = reward_rank_noise_config_from_env()
    estimator = PromptGradientEstimator(
        rl_cluster=rl_cluster,
        training_algo_config=build_grpo_config(cfg.grpo),
        projection_dim=alignment.projection_dim,
        projection_seed=experiment_seed() or 0,
        selection_micro_batch_size=alignment.selection_micro_batch_size,
        noise_config=selector_noise_config,
        promptwise_feature_estimation=(alignment.method == "learnalign"),
    )
    common_curriculum = dict(
        training_examples=training_examples,
        train_batch_size=train_batch_size,
        total_update_steps=frozen_max_steps,
        estimator=estimator,
        writer=writer,
        rl_cluster=rl_cluster,
    )
    if alignment.method == "learnalign":
        curriculum = LearnAlignCurriculum(
            **common_curriculum,
            warmup_prompts=alignment.learnalign_warmup_prompts,
            estimation_rollouts=alignment.learnalign_estimation_rollouts,
            selection_ratio=alignment.selection_ratio,
            warmup_counts_toward_budget=(
                alignment.learnalign_warmup_counts_toward_budget
            ),
        )
    else:
        curriculum = GradAlignCurriculum(
            **common_curriculum,
            validation_examples=heldout_examples[
                : alignment.gradalign_validation_prompts
            ],
            candidate_rollouts=alignment.gradalign_candidate_rollouts,
            validation_rollouts=alignment.gradalign_validation_rollouts,
            selection_ratio=alignment.selection_ratio,
            selection_interval=alignment.gradalign_selection_interval,
        )

    maybe_init_wandb(cfg.runtime.use_wandb)
    try:
        print("Starting alignment-baseline training...")
        with mesh:
            trainer.train(curriculum, val_dataset)
        print("Training complete.")
        try:
            jax.monitoring.clear_event_listeners()
        except Exception:
            pass

        trained_ckpt_path = os.path.join(
            cfg.training.checkpoint_root_directory,
            "actor",
            str(effective_max_steps),
            "model_params",
        )
        if os.path.exists(trained_ckpt_path):
            abs_params = jax.tree.map(
                lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
                nnx.state(lora_policy, nnx.LoRAParam),
            )
            with _suppress_jax_monitoring():
                trained_lora_params = ocp.StandardCheckpointer().restore(
                    trained_ckpt_path,
                    target=abs_params,
                )
            nnx.update(
                lora_policy,
                jax.tree.map(
                    lambda _old, new: new,
                    nnx.state(lora_policy, nnx.LoRAParam),
                    trained_lora_params,
                ),
            )
        else:
            print(f"Checkpoint not found at {trained_ckpt_path}; using live weights.")

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
                result = evaluate(
                    test_dataset,
                    sampler,
                    temperature=None,
                    top_k=1,
                    top_p=None,
                    num_passes=cfg.runtime.eval_num_passes,
                    verbose=cfg.runtime.verbose_eval,
                )
                print(
                    f"post-train: num_correct={result[0]}, total={result[1]}, "
                    f"accuracy={result[2]}%, partial_accuracy={result[3]}%, "
                    f"format_accuracy={result[4]}%"
                )

        output_dir = cfg.runtime.output_dir or f"./{cfg.model.model_id}-lora"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        with _suppress_jax_monitoring():
            save_merged_lora(model_path, output_dir, lora_policy, cfg.lora)
        print(f"Model saved successfully: {output_dir}")
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
