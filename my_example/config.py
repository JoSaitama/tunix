from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "google/gemma-3-1b-it"
    tokenizer_path: str = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
    ignore_patterns: Tuple[str, ...] = ("*.pth",)


@dataclass(frozen=True)
class DataConfig:
    train_data_dir: str = "./data/train"
    test_data_dir: str = "./data/test"
    train_fraction: float = 0.9
    source: str = "tfds"  # tfds | kaggle
    train_micro_batch_size: int = 2
    test_micro_batch_size: int = 1
    max_train_examples: int = 7472
    max_eval_examples: int = 1319
    num_epochs: int = 1


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 64
    alpha: float = 64.0


@dataclass(frozen=True)
class GrpoGenerationConfig:
    max_prompt_length: int = 256
    total_generation_steps: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    num_generations: int = 4
    num_iterations: int = 1
    beta: float = 0.08
    epsilon: float = 0.2


@dataclass(frozen=True)
class EvalGenerationConfig:
    max_prompt_length: int = 256
    total_generation_steps: int | None = None  # fallback to grpo total if None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-6
    b1: float = 0.9
    b2: float = 0.99
    weight_decay: float = 0.1
    warmup_fraction: float = 0.1
    max_grad_norm: float = 0.1
    eval_every_n_steps: int = 256
    checkpoint_root_directory: str | None = "/tmp/content/ckpts"
    save_interval_steps: int = 500
    max_to_keep: int = 4
    metrics_log_dir: str = "/tmp/content/tmp/tensorboard/grpo"
    metrics_flush_every_n_steps: int = 20
    use_dynamic_batch_curation: bool = False
    curation_threshold: float = 3.0


@dataclass(frozen=True)
class RuntimeConfig:
    use_wandb: bool = True
    eval_before_train: bool = True
    eval_after_train: bool = True
    eval_num_passes: int = 1
    output_dir: Optional[str] = None
    mesh_counts: Optional[Tuple[int, int]] = None
    verbose_eval: bool = False


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    grpo: GrpoGenerationConfig
    eval: EvalGenerationConfig
    training: TrainingConfig
    runtime: RuntimeConfig


def _parse_mesh_counts(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mesh-counts must be like '1,4'")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mesh-counts must be integers") from exc


def _normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("", "none", "null"):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO Gemma training entrypoint")

    parser.add_argument("--source", default="tfds", choices=["tfds", "kaggle"])
    parser.add_argument("--train-data-dir", default="./data/train")
    parser.add_argument("--test-data-dir", default="./data/test")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--train-micro-batch-size", type=int, default=2)
    parser.add_argument("--test-micro-batch-size", type=int, default=1)
    parser.add_argument("--max-train-examples", type=int, default=7472)
    parser.add_argument("--max-eval-examples", type=int, default=1319)
    parser.add_argument("--num-epochs", type=int, default=1)

    parser.add_argument("--model-id", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--tokenizer-path",
        default="gs://gemma-data/tokenizers/tokenizer_gemma3.model",
    )

    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=64.0)

    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--total-generation-steps", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--epsilon", type=float, default=0.2)

    parser.add_argument("--eval-max-prompt-length", type=int, default=None)
    parser.add_argument("--eval-total-generation-steps", type=int, default=None)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument("--eval-top-p", type=float, default=None)
    parser.add_argument("--eval-top-k", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--eval-every-n-steps", type=int, default=64)
    parser.add_argument("--checkpoint-root", default="/tmp/content/ckpts")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--save-interval-steps", type=int, default=500)
    parser.add_argument("--max-to-keep", type=int, default=4)
    parser.add_argument("--metrics-log-dir", default="/tmp/content/tmp/tensorboard/grpo")
    parser.add_argument("--metrics-flush-every", type=int, default=20)
    
    parser.add_argument("--use-dynamic-batch-curation", action="store_true")
    parser.add_argument("--curation-threshold", type=float, default=3.0)

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mesh-counts", type=_parse_mesh_counts, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--skip-eval-before", action="store_true")
    parser.add_argument("--skip-eval-after", action="store_true")
    parser.add_argument("--eval-num-passes", type=int, default=1)
    parser.add_argument("--verbose-eval", action="store_true")

    return parser


def config_from_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = build_parser()
    args = parser.parse_args(argv)

    model = ModelConfig(
        model_id=args.model_id,
        tokenizer_path=args.tokenizer_path,
    )
    data = DataConfig(
        train_data_dir=args.train_data_dir,
        test_data_dir=args.test_data_dir,
        train_fraction=args.train_fraction,
        source=args.source,
        train_micro_batch_size=args.train_micro_batch_size,
        test_micro_batch_size=args.test_micro_batch_size,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        num_epochs=args.num_epochs,
    )
    lora = LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    grpo = GrpoGenerationConfig(
        max_prompt_length=args.max_prompt_length,
        total_generation_steps=args.total_generation_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
    )
    eval_cfg = EvalGenerationConfig(
        max_prompt_length=args.eval_max_prompt_length
        if args.eval_max_prompt_length is not None
        else args.max_prompt_length,
        total_generation_steps=args.eval_total_generation_steps,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        top_k=args.eval_top_k,
    )
    checkpoint_root = _normalize_optional_path(args.checkpoint_root)
    if args.no_checkpoint:
        checkpoint_root = None
    training = TrainingConfig(
        learning_rate=args.learning_rate,
        b1=args.b1,
        b2=args.b2,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        max_grad_norm=args.max_grad_norm,
        eval_every_n_steps=args.eval_every_n_steps,
        checkpoint_root_directory=checkpoint_root,
        save_interval_steps=args.save_interval_steps,
        max_to_keep=args.max_to_keep,
        metrics_log_dir=args.metrics_log_dir,
        metrics_flush_every_n_steps=args.metrics_flush_every,
        use_dynamic_batch_curation=args.use_dynamic_batch_curation,
        curation_threshold=args.curation_threshold,
    )
    runtime = RuntimeConfig(
        use_wandb=not args.no_wandb,
        eval_before_train=not args.skip_eval_before,
        eval_after_train=not args.skip_eval_after,
        eval_num_passes=args.eval_num_passes,
        output_dir=args.output_dir,
        mesh_counts=args.mesh_counts,
        verbose_eval=args.verbose_eval,
    )

    return Config(
        model=model,
        data=data,
        lora=lora,
        grpo=grpo,
        eval=eval_cfg,
        training=training,
        runtime=runtime,
    )
