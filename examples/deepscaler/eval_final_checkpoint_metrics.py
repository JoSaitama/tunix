#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a DeepScaler final actor checkpoint with @16 math metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import jax
from flax import nnx
import pandas as pd
import transformers

from tunix.generate import mappings
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen2 import model as qwen2_model_lib
from tunix.models.qwen2 import params as qwen2_params_lib
from tunix.sft import checkpoint_manager as checkpoint_manager_lib
from tunix.utils import math_eval_metrics


MODEL_CONFIGS = {
    "deepseek_r1_distill_qwen_1p5b": (
        qwen2_model_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b
    ),
    "qwen2p5_1p5b": qwen2_model_lib.ModelConfig.qwen2p5_1p5b,
    "qwen2p5_math_1p5b": qwen2_model_lib.ModelConfig.qwen2p5_math_1p5b,
}


CHECKPOINT_SOURCE_CHECKPOINT = "checkpoint"
CHECKPOINT_SOURCE_BASE_MODEL = "base_model"


def _parse_csv_tuple(value: str, *, item_type=str) -> tuple[Any, ...]:
  return tuple(item_type(x.strip()) for x in value.split(",") if x.strip())


def _resolve_actor_checkpoint_root(args: argparse.Namespace) -> Path:
  if args.checkpoint_root:
    return Path(args.checkpoint_root).expanduser()
  if not args.run_root:
    raise ValueError("Either --run_root or --checkpoint_root must be set.")
  return (
      Path(args.run_root).expanduser() / "checkpoints" / "actor"
  )


def _latest_checkpoint_step(actor_checkpoint_root: Path) -> int:
  steps = [
      int(path.name)
      for path in actor_checkpoint_root.iterdir()
      if path.is_dir() and path.name.isdigit()
  ]
  if not steps:
    raise ValueError(
        f"No numeric checkpoint steps found under {actor_checkpoint_root}."
    )
  return max(steps)


def _resolve_checkpoint_step(
    actor_checkpoint_root: Path, checkpoint_step: str
) -> int:
  if checkpoint_step == "latest":
    return _latest_checkpoint_step(actor_checkpoint_root)
  return int(checkpoint_step)


def _uses_actor_checkpoint(args: argparse.Namespace) -> bool:
  return args.checkpoint_source == CHECKPOINT_SOURCE_CHECKPOINT


def _make_mesh(mesh_shape: Sequence[int], mesh_axes: Sequence[str]):
  required_devices = math.prod(mesh_shape)
  devices = jax.local_devices()
  if len(devices) < required_devices:
    raise ValueError(
        f"Mesh shape {tuple(mesh_shape)} requires {required_devices} devices, "
        f"but only {len(devices)} local devices are visible."
    )
  device_mesh = jax._src.mesh_utils.create_device_mesh(  # pylint: disable=protected-access
      mesh_shape, devices[:required_devices]
  )
  return jax.sharding.Mesh(
      device_mesh,
      axis_names=mesh_axes,
      axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_shape),
  )


def _maybe_initialize_jax_distributed() -> None:
  if os.environ.get("TUNIX_INIT_JAX_DISTRIBUTED") != "1":
    return
  if jax.distributed.is_initialized():
    return
  print("Initializing JAX distributed runtime.", flush=True)
  jax.distributed.initialize()


def _resolve_output_dir(
    args: argparse.Namespace,
    actor_checkpoint_root: Path | None,
    checkpoint_step: int | None,
) -> Path:
  if args.output_dir:
    return Path(args.output_dir).expanduser()
  if not _uses_actor_checkpoint(args):
    if args.run_root:
      return (
          Path(args.run_root).expanduser()
          / "eval"
          / f"base_model_k{args.num_samples}"
      )
    model_name = Path(args.model_version).expanduser().resolve().name
    return Path.cwd() / "eval" / f"{model_name}_base_k{args.num_samples}"
  if actor_checkpoint_root is None or checkpoint_step is None:
    raise ValueError("Checkpoint eval output directory requires a checkpoint.")
  return (
      actor_checkpoint_root.parent.parent
      / "eval"
      / f"final_actor_{checkpoint_step}_k{args.num_samples}"
  )


def _done_sentinel_path(output_dir: Path) -> Path:
  return output_dir / ".primary_done"


def _wait_for_primary_done(args: argparse.Namespace) -> None:
  if args.secondary_sleep_seconds is not None:
    print(
        "Secondary process sleeping for "
        f"{args.secondary_sleep_seconds}s before exit.",
        flush=True,
    )
    time.sleep(args.secondary_sleep_seconds)
    return
  actor_checkpoint_root = None
  checkpoint_step = None
  if _uses_actor_checkpoint(args):
    actor_checkpoint_root = _resolve_actor_checkpoint_root(args)
    checkpoint_step = _resolve_checkpoint_step(
        actor_checkpoint_root, args.checkpoint_step
    )
  output_dir = _resolve_output_dir(args, actor_checkpoint_root, checkpoint_step)
  sentinel = _done_sentinel_path(output_dir)
  start_time = time.time()
  deadline = start_time + args.secondary_wait_timeout_seconds
  print(
      f"Waiting for primary eval completion sentinel at {sentinel}.",
      flush=True,
  )
  while time.time() < deadline:
    try:
      if sentinel.exists() and sentinel.stat().st_mtime >= start_time - 1:
        print("Primary eval completion sentinel observed.", flush=True)
        return
    except OSError:
      pass
    time.sleep(args.secondary_poll_seconds)
  raise TimeoutError(
      "Timed out waiting for primary eval completion sentinel at "
      f"{sentinel}."
  )


def _load_dataset(args: argparse.Namespace) -> pd.DataFrame:
  dataset_path = Path(args.dataset).expanduser()
  if dataset_path.suffix == ".parquet":
    df = pd.read_parquet(dataset_path)
  elif dataset_path.suffix == ".jsonl":
    df = pd.read_json(dataset_path, lines=True)
  elif dataset_path.suffix == ".json":
    df = pd.read_json(dataset_path)
  else:
    raise ValueError(f"Unsupported dataset format: {dataset_path}")

  if args.limit is not None:
    df = df.head(args.limit)
  return df.reset_index(drop=True)


def _format_prompt(
    tokenizer: transformers.PreTrainedTokenizerBase,
    question: str,
    dataset_type: str,
) -> str:
  if dataset_type == "aime":
    instruction = (
        "Let's think step by step, and put your final answer within \\boxed{}."
    )
    prompt = f"{question} {instruction}"
  else:
    instruction = (
        "Please reason step by step. Your final answer must appear inside "
        "\\boxed{...} and nothing else."
    )
    prompt = f"{instruction} {question}"
  return tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      tokenize=False,
      add_generation_prompt=True,
  )


def _load_final_actor_model(args: argparse.Namespace, mesh: jax.sharding.Mesh):
  config = MODEL_CONFIGS[args.model_config]()
  actor_checkpoint_root = _resolve_actor_checkpoint_root(args)
  checkpoint_step = _resolve_checkpoint_step(
      actor_checkpoint_root, args.checkpoint_step
  )
  print(
      "Loading final actor checkpoint "
      f"{actor_checkpoint_root / str(checkpoint_step)}.",
      flush=True,
  )

  with mesh:
    model = qwen2_model_lib.Qwen2(config, rngs=nnx.Rngs(params=0))
  manager = checkpoint_manager_lib.CheckpointManager(str(actor_checkpoint_root))
  restored_step, _ = manager.maybe_restore(model, step=checkpoint_step)
  manager.close()
  if restored_step != checkpoint_step:
    raise RuntimeError(
        f"Requested checkpoint step {checkpoint_step}, restored {restored_step}."
    )
  print(f"Restored actor checkpoint step {restored_step}.", flush=True)
  return model, config, actor_checkpoint_root, checkpoint_step


def _load_base_model(args: argparse.Namespace, mesh: jax.sharding.Mesh):
  config = MODEL_CONFIGS[args.model_config]()
  print(
      f"Loading base model safetensors from {args.model_version}.",
      flush=True,
  )
  with mesh:
    model = qwen2_params_lib.create_model_from_safe_tensors(
        file_dir=args.model_version,
        config=config,
        mesh=mesh,
    )
  print("Base model safetensors loaded into NNX model.", flush=True)
  return model, config


def _create_sampler(
    args: argparse.Namespace,
    model: nnx.Module | None,
    model_config: qwen2_model_lib.ModelConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
    mesh: jax.sharding.Mesh,
):
  if args.sampler_type == "vanilla":
    print("Creating vanilla sampler.", flush=True)
    cache_config = sampler_lib.CacheConfig(
        cache_size=args.max_prompt_length + args.max_generation_steps + 100,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    )
    return sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=cache_config,
    )

  from tunix.generate import vllm_sampler as vllm_sampler_lib  # pylint: disable=g-import-not-at-top

  print("Creating vLLM sampler.", flush=True)
  if model is None:
    mapping_model = qwen2_model_lib.Qwen2
  else:
    mapping_model = model
  mapping_config = mappings.MappingConfig.build(
      mapping_obj=None,
      model=mapping_model,
      backend="vllm_jax",
  )
  engine_kwargs = {
      "model": args.model_version,
      "max_model_len": (
          args.max_prompt_length + args.max_generation_steps + 100
      ),
      "max_num_seqs": args.max_num_seqs,
      "max_num_batched_tokens": args.max_num_batched_tokens,
      "disable_log_stats": True,
  }
  sampler = vllm_sampler_lib.VllmSampler(
      tokenizer=tokenizer,
      config=vllm_sampler_lib.VllmConfig(
          mesh=mesh,
          hbm_utilization=args.vllm_hbm_utilization,
          init_with_random_weights=model is not None,
          tpu_backend_type=args.tpu_backend_type,
          server_mode=args.vllm_server_mode,
          tensor_parallel_size=args.tensor_parallel_size,
          data_parallel_size=args.data_parallel_size,
          mapping_config=mapping_config,
          engine_kwargs=engine_kwargs,
      ),
  )
  if model is not None:
    if _uses_actor_checkpoint(args):
      print("Syncing restored actor weights into vLLM sampler.", flush=True)
    else:
      print("Syncing base model weights into vLLM sampler.", flush=True)
    sampler.update_params(nnx.state(model))
  else:
    print("Using vLLM-loaded base model weights.", flush=True)
  print("vLLM sampler is ready.", flush=True)
  return sampler


def _generate_once(
    args: argparse.Namespace,
    sampler,
    prompts: list[str],
    sample_index: int,
):
  top_p = None if args.top_p is not None and args.top_p <= 0 else args.top_p
  seed = None if args.sampler_type == "vllm" else sample_index
  return sampler(
      input_strings=prompts,
      max_generation_steps=args.max_generation_steps,
      max_prompt_length=args.max_prompt_length,
      temperature=args.temperature,
      top_p=top_p,
      top_k=args.top_k,
      seed=seed,
      echo=False,
      pad_output=False,
  )


def _write_json(path: Path, value: Any) -> None:
  with path.open("w", encoding="utf-8") as f:
    json.dump(value, f, indent=2, sort_keys=True)
    f.write("\n")


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
  _maybe_initialize_jax_distributed()
  if jax.process_index() != 0:
    print(
        f"Process {jax.process_index()} initialized; primary process performs "
        "eval.",
        flush=True,
    )
    _wait_for_primary_done(args)
    return {
        "process_index": jax.process_index(),
        "status": "secondary_eval_done",
    }

  mesh_shape = _parse_csv_tuple(args.mesh_shape, item_type=int)
  mesh_axes = _parse_csv_tuple(args.mesh_axes, item_type=str)
  mesh = _make_mesh(mesh_shape, mesh_axes)

  tokenizer_path = args.tokenizer_path or args.model_version
  print(f"Loading tokenizer from {tokenizer_path}.", flush=True)
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )

  actor_checkpoint_root = None
  checkpoint_step = None
  if _uses_actor_checkpoint(args):
    model, model_config, actor_checkpoint_root, checkpoint_step = (
        _load_final_actor_model(args, mesh)
    )
  else:
    model, model_config = _load_base_model(args, mesh)
  sampler = _create_sampler(args, model, model_config, tokenizer, mesh)
  df = _load_dataset(args)

  output_dir = _resolve_output_dir(args, actor_checkpoint_root, checkpoint_step)
  output_dir.mkdir(parents=True, exist_ok=True)
  done_sentinel = _done_sentinel_path(output_dir)
  done_sentinel.unlink(missing_ok=True)
  samples_jsonl = output_dir / "samples.jsonl"
  summary_json = output_dir / "summary.json"

  records = []
  with samples_jsonl.open("w", encoding="utf-8") as f:
    for start in range(0, len(df), args.batch_size):
      batch_df = df.iloc[start : start + args.batch_size]
      print(
          f"Generating rows {start}-{start + len(batch_df) - 1} "
          f"with {args.num_samples} samples each.",
          flush=True,
      )
      expanded_prompts = []
      expanded_metadata = []
      for item_index, (_, row) in enumerate(batch_df.iterrows()):
        prompt = _format_prompt(
            tokenizer, str(row[args.question_col]), args.dataset_type
        )
        for sample_index in range(args.num_samples):
          expanded_prompts.append(prompt)
          expanded_metadata.append((item_index, sample_index, row))

      output = _generate_once(args, sampler, expanded_prompts, start)
      for output_index, (item_index, sample_index, row) in enumerate(
          expanded_metadata
      ):
        tokens = output.tokens[output_index]
        token_count = len(tokens)
        response = output.text[output_index]
        ground_truth = str(row[args.answer_col])
        record = {
            "problem_id": int(start + item_index),
            "sample_index": sample_index,
            "question": str(row[args.question_col]),
            "ground_truth": ground_truth,
            "response": response,
            "extracted_answer": math_eval_metrics.extract_response_answer(
                response
            ),
            "token_count": token_count,
            "truncated": token_count >= args.max_generation_steps,
        }
        records.append(record)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
      f.flush()

  metrics = math_eval_metrics.compute_at_k_metrics(
      records,
      k=args.num_samples,
      max_generation_steps=args.max_generation_steps,
  )
  summary = {
      "metrics": metrics,
      "checkpoint_source": args.checkpoint_source,
      "checkpoint_root": (
          str(actor_checkpoint_root) if actor_checkpoint_root else None
      ),
      "checkpoint_step": checkpoint_step,
      "checkpoint_path": (
          str(actor_checkpoint_root / str(checkpoint_step))
          if actor_checkpoint_root and checkpoint_step is not None
          else None
      ),
      "model_version": str(Path(args.model_version).expanduser()),
      "dataset": str(Path(args.dataset).expanduser()),
      "num_dataset_rows": len(df),
      "samples_jsonl": str(samples_jsonl),
      "args": vars(args),
  }
  _write_json(summary_json, summary)
  print(json.dumps(summary, indent=2, sort_keys=True))
  _write_json(
      done_sentinel,
      {"checkpoint_step": checkpoint_step, "status": "primary_eval_done"},
  )
  return summary


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Evaluate a final DeepScaler actor checkpoint."
  )
  parser.add_argument("--run_root", default=None)
  parser.add_argument("--checkpoint_root", default=None)
  parser.add_argument(
      "--checkpoint_source",
      choices=(CHECKPOINT_SOURCE_CHECKPOINT, CHECKPOINT_SOURCE_BASE_MODEL),
      default=CHECKPOINT_SOURCE_CHECKPOINT,
  )
  parser.add_argument("--checkpoint_step", default="latest")
  parser.add_argument(
      "--dataset", default="/home/eve/tunix-hf-data/aime_eval.parquet"
  )
  parser.add_argument("--dataset_type", choices=("aime", "math"), default="aime")
  parser.add_argument("--question_col", default="problem")
  parser.add_argument("--answer_col", default="answer")
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--output_dir", default=None)

  parser.add_argument("--model_version", default="/tmp/models")
  parser.add_argument("--tokenizer_path", default=None)
  parser.add_argument(
      "--model_config",
      choices=sorted(MODEL_CONFIGS),
      default="deepseek_r1_distill_qwen_1p5b",
  )
  parser.add_argument("--mesh_shape", default="4,1")
  parser.add_argument("--mesh_axes", default="fsdp,tp")

  parser.add_argument("--sampler_type", choices=("vllm", "vanilla"), default="vllm")
  parser.add_argument("--num_samples", type=int, default=16)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=2048)
  parser.add_argument("--max_generation_steps", type=int, default=8192)
  parser.add_argument("--temperature", type=float, default=0.6)
  parser.add_argument("--top_p", type=float, default=0.95)
  parser.add_argument("--top_k", type=int, default=None)

  parser.add_argument("--vllm_hbm_utilization", type=float, default=0.8)
  parser.add_argument("--tpu_backend_type", default="jax")
  parser.add_argument("--vllm_server_mode", action="store_true")
  parser.add_argument("--tensor_parallel_size", type=int, default=-1)
  parser.add_argument("--data_parallel_size", type=int, default=-1)
  parser.add_argument("--max_num_seqs", type=int, default=16)
  parser.add_argument("--max_num_batched_tokens", type=int, default=38400)
  # Kept for old tmux commands created before secondary processes used a real
  # distributed barrier.
  parser.add_argument(
      "--secondary_sleep_seconds",
      type=int,
      default=None,
      help=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--secondary_wait_timeout_seconds",
      type=int,
      default=86400,
      help=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--secondary_poll_seconds",
      type=float,
      default=5.0,
      help=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--disable_hard_exit",
      action="store_true",
      help=argparse.SUPPRESS,
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  run_eval(args)
  if not args.disable_hard_exit:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
  main()
