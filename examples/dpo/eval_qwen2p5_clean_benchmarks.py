#!/usr/bin/env python3

"""Prepare and optionally generate clean external benchmark artifacts."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
import csv
import gc
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
import urllib.request

from omegaconf import OmegaConf


REPO_ROOT = Path("/home/lhf_hongfu_gmail_com/tunix")
CONFIG_PATH = (
    REPO_ROOT / "examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
)
RUN_GLOB = os.environ.get(
    "DPO_EVAL_RUN_GLOB",
    "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_*_clean_lora_full_*",
)
RUN_RE = re.compile(
    os.environ.get(
        "DPO_EVAL_RUN_RE",
        r"dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
        r"(?P<variant>.+)_clean_lora_full_(?P<run_ts>\\d{8}_\\d{6})",
    )
)
METHOD_ORDER = {
    "vanilla_dpo": 0,
    "random_pair_filtering": 1,
    "reward_based_filtering": 2,
    "self_inf": 3,
    "self_inf_norm": 3.05,
    "self_inf_re_loo": 1.15,
}
METHOD_DISPLAY = {
    "vanilla_dpo": "Vanilla DPO",
    "random_pair_filtering": "Random Pair Filtering",
    "reward_based_filtering": "Reward-based Filtering",
    "self_inf": "Self-inf (ours)",
    "self_inf_re_loo": "Self-DTV-Re-LOO",
    "self_inf_norm": "Self-DTV-Norm",
}
METHOD_ALIASES = {
    "vanilla_dpo": "vanilla_dpo",
    "vanilla": "vanilla_dpo",
    "vanilla dpo": "vanilla_dpo",
    "vanilla dpo (all pairs)": "vanilla_dpo",
    "random_pair_filtering": "random_pair_filtering",
    "random pair filtering": "random_pair_filtering",
    "reward_based_filtering": "reward_based_filtering",
    "reward-based filtering": "reward_based_filtering",
    "reward based filtering": "reward_based_filtering",
    "self_inf_norm": "self_inf_norm",
    "dtv norm": "self_inf_norm",
    "self dtv norm": "self_inf_norm",
    "self-dtv norm": "self_inf_norm",
    "self-dtv-norm": "self_inf_norm",
    "self influence norm": "self_inf_norm",
    "self-influence norm": "self_inf_norm",
    "self-influence-norm": "self_inf_norm",
    "self inf norm": "self_inf_norm",
    "self-inf-norm": "self_inf_norm",
    "self-inf norm": "self_inf_norm",
    "self_dtv_norm": "self_inf_norm",
    "dtv_norm": "self_inf_norm",
    "self_inf_norm_batch": "self_inf_norm",
    "self_inf": "self_inf",
    "self_inf_re_loo": "self_inf_re_loo",
    "self_inf_re_loo_batch": "self_inf_re_loo",
    "self inf re loo": "self_inf_re_loo",
    "self-inf-re-loo": "self_inf_re_loo",
    "self-inf re loo": "self_inf_re_loo",
    "self_inf_reliable_loo": "self_inf_re_loo",
    "self inf reliable loo": "self_inf_re_loo",
    "self-inf reliable loo": "self_inf_re_loo",
    "self-inf-reliable-loo": "self_inf_re_loo",
    "self inf reliable loo batch": "self_inf_re_loo",
    "self-inf-reliable-loo-batch": "self_inf_re_loo",
    "self_inf_reliable_loo_batch": "self_inf_re_loo",
    "self influence reliable loo": "self_inf_re_loo",
    "self-influence-reliable-loo": "self_inf_re_loo",
    "self_dtv_re_loo": "self_inf_re_loo",
    "dtv_re_loo": "self_inf_re_loo",
    "dtv re loo": "self_inf_re_loo",
    "reliable dtv loo": "self_inf_re_loo",
    "reliable-dtv-loo": "self_inf_re_loo",
    "self-inf": "self_inf",
    "self inf": "self_inf",
    "self-inf (ours)": "self_inf",
}
BENCHMARKS = ("mt_bench", "alpacaeval2", "arena_hard", "ifeval", "ifbench")
MT_BENCH_TEMPERATURE = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}
MT_BENCH_QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
MT_BENCH_JUDGE_PROMPTS_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/judge_prompts.jsonl"
)
MT_BENCH_REFERENCE_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"
)
ALPACA_EVAL_DATASET = "tatsu-lab/alpaca_eval"
ALPACA_EVAL_REFERENCE_FILE = "alpaca_eval_gpt4_baseline.json"
ARENA_HARD_DATASET = "lmarena-ai/arena-hard-auto"
ARENA_HARD_QUESTION_FILE = "data/arena-hard-v2.0/question.jsonl"
ARENA_HARD_BASELINE_FILES = (
    "data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl",
    "data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl",
)
ARENA_HARD_CLONE = Path("/home/lhf_hongfu_gmail_com/.cache/arena-hard-auto")
DEFAULT_MT_BENCH_JUDGE_VENV = Path("/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-MTB")
DEFAULT_OPENAI_JUDGE_VENV = Path("/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OAI")
DEFAULT_OFFLINE_EVAL_VENV = Path("/home/lhf_hongfu_gmail_com/.venvs/DPO-EVAL-OFFLINE")
DEFAULT_MT_BENCH_JUDGE_MODEL = "gpt-4-turbo"
DEFAULT_ALPACAEVAL_ANNOTATOR = "weighted_alpaca_eval_gpt-4o-mini-2024-07-18"
DEFAULT_ARENA_HARD_JUDGE = "gpt-4.1"
MIN_SAMPLER_CACHE_SIZE = 12288
ARENA_HARD_PROMPT_LENGTH_BUDGET = 10000
IFEVAL_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/google-research-ifeval")
IFBENCH_ROOT = Path("/home/lhf_hongfu_gmail_com/.cache/IFBench")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Prepare clean external benchmark workspaces and optionally generate "
          "local TPU answers for the four clean Qwen2.5 DPO runs."
      )
  )
  parser.add_argument("--repo-root", default=str(REPO_ROOT))
  parser.add_argument("--config-path", default=str(CONFIG_PATH))
  parser.add_argument("--run-ts", required=True)
  parser.add_argument(
      "--methods",
      nargs="+",
      default=list(METHOD_ORDER),
      help="Subset of clean methods to include.",
  )
  parser.add_argument(
      "--benchmarks",
      nargs="+",
      default=list(BENCHMARKS),
      help="Subset of benchmarks to prepare or generate.",
  )
  parser.add_argument(
      "--judge-model-family",
      default="gpt4",
      help="Label for downstream judge planning metadata.",
  )
  parser.add_argument(
      "--output-root",
      default=None,
      help="Defaults to runs/results/clean_benchmarks_<run-ts> under repo root.",
  )
  parser.add_argument(
      "--skip-generation",
      action="store_true",
      help="Only prepare benchmark assets and stage-B command files.",
  )
  parser.add_argument(
      "--question-limit",
      type=int,
      default=None,
      help="Optional debug cap applied independently to each benchmark.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Base generation seed for deterministic local sampling.",
  )
  parser.add_argument(
      "--max-prompt-length",
      type=int,
      default=4096,
      help="Prompt cache budget for local generation.",
  )
  parser.add_argument(
      "--max-generation-steps",
      type=int,
      default=1024,
      help="Generation budget for each answer.",
  )
  parser.add_argument("--top-k", type=int, default=50)
  parser.add_argument("--top-p", type=float, default=0.95)
  parser.add_argument(
      "--force",
      action="store_true",
      help="Overwrite existing answer files instead of skipping them.",
  )
  parser.add_argument(
      "--generation-batch-size",
      type=int,
      default=8,
      help="Batch size for single-turn local generation benchmarks.",
  )
  parser.add_argument(
      "--offline-eval-venv",
      default=str(DEFAULT_OFFLINE_EVAL_VENV),
      help="Python venv used for offline benchmark evaluators like IFEval.",
  )
  return parser.parse_args()



# Self-Inf-lambda compatibility.
# This method is implemented as a unified DTV-family score:
#   S_j(lambda) = g_j^T G_{-j} + lambda * ||g_j||^2
# The default experiment uses lambda = 0.5.
METHOD_ORDER["self_inf_lambda"] = 1.25
METHOD_ALIASES.update({
    "self_inf_lambda": "self_inf_lambda",
    "self inf lambda": "self_inf_lambda",
    "self-inf-lambda": "self_inf_lambda",
    "self-inf lambda": "self_inf_lambda",
    "self_inf_lambda_batch": "self_inf_lambda",
    "self inf lambda batch": "self_inf_lambda",
    "self-inf lambda batch": "self_inf_lambda",
    "self-inf-lambda-batch": "self_inf_lambda",
    "self influence lambda batch": "self_inf_lambda",
    "self-influence-lambda-batch": "self_inf_lambda",
    "dtv lambda batch": "self_inf_lambda",
    "dtv-lambda-batch": "self_inf_lambda",
    "self influence lambda": "self_inf_lambda",
    "self-influence-lambda": "self_inf_lambda",
    "self_dtv_lambda": "self_inf_lambda",
    "dtv_lambda": "self_inf_lambda",
    "dtv lambda": "self_inf_lambda",
})
if "METHOD_DISPLAY" in globals():
    METHOD_DISPLAY["self_inf_lambda"] = "Self-Inf-Lambda"
if "METHOD_DISPLAY_NAMES" in globals():
    METHOD_DISPLAY_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"
if "DISPLAY_NAMES" in globals():
    DISPLAY_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"
if "METHOD_NAMES" in globals():
    METHOD_NAMES["self_inf_lambda"] = "Self-Inf-Lambda"

def _normalize_method(method: str) -> str:
  normalized = method.strip().lower().replace("_", " ").replace("-", " ")
  normalized = re.sub(r"\s+", " ", normalized)
  if normalized not in METHOD_ALIASES:
    raise ValueError(
        f"Unsupported method {method!r}. Expected one of: "
        f"{', '.join(sorted(METHOD_ALIASES))}"
    )
  return METHOD_ALIASES[normalized]


def _normalize_benchmark(name: str) -> str:
  normalized = name.strip().lower().replace("-", "_")
  aliases = {
      "mt_bench": "mt_bench",
      "mtbench": "mt_bench",
      "alpacaeval2": "alpacaeval2",
      "alpaca_eval_2": "alpacaeval2",
      "alpaca_eval2": "alpacaeval2",
      "arena_hard": "arena_hard",
      "arenahard": "arena_hard",
      "arena_hard_auto": "arena_hard",
      "ifeval": "ifeval",
      "if_eval": "ifeval",
      "ifbench": "ifbench",
      "if_bench": "ifbench",
  }
  if normalized not in aliases:
    raise ValueError(
        f"Unsupported benchmark {name!r}. Expected one of: "
        f"{', '.join(sorted(aliases))}"
    )
  return aliases[normalized]


def _make_mesh(mesh_config: dict[str, Any]):
  import jax

  axis_shapes = ast.literal_eval(mesh_config["shape"])
  axis_names = ast.literal_eval(mesh_config["axis_names"])
  return jax.make_mesh(
      tuple(axis_shapes),
      tuple(axis_names),
      axis_types=(jax.sharding.AxisType.Auto,) * len(tuple(axis_names)),
  )


def _discover_clean_runs(
    repo_root: Path,
    run_ts: str,
    methods: Sequence[str],
) -> list[dict[str, Any]]:
  wanted = {_normalize_method(method) for method in methods}
  candidates = []
  for run_dir in sorted((repo_root / "runs").glob(RUN_GLOB)):
    if not run_dir.is_dir():
      continue
    match = RUN_RE.fullmatch(run_dir.name)
    if match is None or match.group("run_ts") != run_ts:
      continue
    variant = match.group("variant")
    if variant not in wanted:
      continue
    exported_model = run_dir / "exported_model"
    if not exported_model.is_dir():
      continue
    candidates.append(
        {
            "variant": variant,
            "display_name": METHOD_DISPLAY[variant],
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "exported_model_path": str(exported_model),
            "tensorboard_dir": str(run_dir / "tensorboard"),
        }
    )
  candidates.sort(key=lambda item: METHOD_ORDER[item["variant"]])
  missing = [method for method in wanted if method not in {c["variant"] for c in candidates}]
  if missing:
    raise SystemExit(
        "Missing clean exported_model directories for: "
        + ", ".join(missing)
        + f" (run_ts={run_ts})"
    )
  return candidates


def _ensure_parent(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)


def _download_text(url: str, dst: Path) -> None:
  _ensure_parent(dst)
  with urllib.request.urlopen(url) as response:
    dst.write_bytes(response.read())


def _copy_if_missing(src: Path, dst: Path) -> None:
  if dst.exists():
    return
  _ensure_parent(dst)
  shutil.copy2(src, dst)


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
  rows = []
  with path.open() as f:
    for line in f:
      if not line.strip():
        continue
      rows.append(json.loads(line))
      if limit is not None and len(rows) >= limit:
        break
  return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
  _ensure_parent(path)
  count = 0
  with path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
      count += 1
  return count


def _prepare_mt_bench_assets(root: Path) -> dict[str, str]:
  data_root = root / "mt_bench"
  question_path = data_root / "data/mt_bench/question.jsonl"
  judge_prompts_path = data_root / "data/judge_prompts.jsonl"
  reference_path = data_root / "data/mt_bench/reference_answer/gpt-4.jsonl"
  turbo_reference_path = data_root / "data/mt_bench/reference_answer/gpt-4-turbo.jsonl"
  if not question_path.exists():
    _download_text(MT_BENCH_QUESTION_URL, question_path)
  if not judge_prompts_path.exists():
    _download_text(MT_BENCH_JUDGE_PROMPTS_URL, judge_prompts_path)
  if not reference_path.exists():
    _download_text(MT_BENCH_REFERENCE_URL, reference_path)
  _copy_if_missing(reference_path, turbo_reference_path)
  return {
      "workspace": str(data_root),
      "question_path": str(question_path),
      "judge_prompts_path": str(judge_prompts_path),
      "reference_answer_path": str(reference_path),
      "answer_dir": str(data_root / "data/mt_bench/model_answer"),
  }


def _prepare_alpacaeval_assets(root: Path) -> dict[str, str]:
  from huggingface_hub import hf_hub_download

  data_root = root / "alpacaeval2"
  reference_dst = data_root / "reference_outputs/alpaca_eval_gpt4_baseline.json"
  src = Path(
      hf_hub_download(
          ALPACA_EVAL_DATASET,
          filename=ALPACA_EVAL_REFERENCE_FILE,
          repo_type="dataset",
      )
  )
  _copy_if_missing(src, reference_dst)
  return {
      "workspace": str(data_root),
      "reference_output_path": str(reference_dst),
      "model_output_dir": str(data_root / "model_outputs"),
  }


def _prepare_arena_hard_assets(root: Path) -> dict[str, str]:
  from huggingface_hub import hf_hub_download

  data_root = root / "arena_hard"
  question_dst = data_root / ARENA_HARD_QUESTION_FILE
  src = Path(
      hf_hub_download(
          ARENA_HARD_DATASET,
          filename=ARENA_HARD_QUESTION_FILE,
          repo_type="dataset",
      )
  )
  _copy_if_missing(src, question_dst)
  baseline_paths = []
  for filename in ARENA_HARD_BASELINE_FILES:
    baseline_src = Path(
        hf_hub_download(
            ARENA_HARD_DATASET,
            filename=filename,
            repo_type="dataset",
        )
    )
    baseline_dst = data_root / filename
    _copy_if_missing(baseline_src, baseline_dst)
    baseline_paths.append(str(baseline_dst))
  return {
      "workspace": str(data_root),
      "question_path": str(question_dst),
      "answer_dir": str(data_root / "data/arena-hard-v2.0/model_answer"),
      "baseline_answer_paths": baseline_paths,
  }


def _prepare_ifeval_assets(root: Path) -> dict[str, str]:
  ifeval_root = IFEVAL_ROOT / "instruction_following_eval"
  input_src = ifeval_root / "data/input_data.jsonl"
  if not input_src.exists():
    raise SystemExit(
        "IFEval input_data.jsonl is missing. Expected clone at "
        f"{ifeval_root}."
  )
  data_root = root / "ifeval"
  input_dst = data_root / "data/input_data.jsonl"
  _ensure_parent(input_dst)
  shutil.copy2(input_src, input_dst)
  return {
      "workspace": str(data_root),
      "input_path": str(input_dst),
      "official_root": str(ifeval_root),
      "response_dir": str(data_root / "responses"),
      "result_dir": str(data_root / "results"),
  }


def _prepare_ifbench_assets(root: Path) -> dict[str, str]:
  from datasets import load_dataset

  data_root = root / "ifbench"
  input_dst = data_root / "data/IFBench_test.jsonl"
  dataset = load_dataset("allenai/IFBench_test", split="train")
  _write_jsonl(input_dst, list(dataset))
  return {
      "workspace": str(data_root),
      "input_path": str(input_dst),
      "official_root": str(IFBENCH_ROOT),
      "response_dir": str(data_root / "responses"),
      "result_dir": str(data_root / "results"),
  }


class TunixChatGenerator:
  """Local Qwen2.5 sampler backed by the current Tunix/JAX TPU stack."""

  def __init__(
      self,
      *,
      config_path: str,
      exported_model_path: str,
      max_prompt_length: int,
      max_generation_steps: int,
      top_k: int,
      top_p: float,
  ):
    import jax
    from tunix.cli.utils import model as model_lib
    from tunix.generate import sampler as sampler_lib

    base_cfg = OmegaConf.load(config_path)
    actor_config = OmegaConf.to_container(base_cfg.actor_model_config, resolve=True)
    tokenizer_config = OmegaConf.to_container(
        base_cfg.tokenizer_config, resolve=True
    )

    actor_config["model_path"] = exported_model_path
    actor_config.pop("lora_config", None)
    self._mesh = _make_mesh(actor_config["mesh"])
    self._jax = jax

    self._model, tokenizer_path, _ = model_lib.create_model(
        actor_config,
        tokenizer_config,
        self._mesh,
        return_model_path=True,
    )
    self._tokenizer = model_lib.create_tokenizer(tokenizer_config, tokenizer_path)
    cache_size = max(
        MIN_SAMPLER_CACHE_SIZE,
        max_prompt_length + max_generation_steps + 256,
    )
    cache_config = sampler_lib.CacheConfig(
        cache_size=cache_size,
        num_layers=self._model.config.num_layers,
        num_kv_heads=self._model.config.num_kv_heads,
        head_dim=self._model.config.head_dim,
    )
    self._sampler = sampler_lib.Sampler(
        transformer=self._model,
        tokenizer=self._tokenizer,
        cache_config=cache_config,
    )
    self._max_prompt_length = max_prompt_length
    self._max_generation_steps = max_generation_steps
    self._top_k = top_k
    self._top_p = top_p
    self._stop_token_ids = self._resolve_stop_token_ids()

  def _resolve_sampling_controls(
      self,
      *,
      temperature: float,
  ) -> tuple[float, int | None, float | None]:
    """Avoid invalid top-p sampling when benchmark requests greedy decoding."""
    if temperature <= 0.0:
      return 0.0, None, None
    return temperature, self._top_k, self._top_p

  def _resolve_stop_token_ids(self) -> list[int]:
    candidate_ids = []
    eos_id = self._tokenizer.eos_id()
    if eos_id is not None:
      candidate_ids.append(int(eos_id))
    for token in ("<|im_end|>", "<｜end▁of▁sentence｜>"):
      token_ids = self._tokenizer.encode(token, add_special_tokens=False)
      if len(token_ids) == 1:
        candidate_ids.append(int(token_ids[0]))
    return list(dict.fromkeys(candidate_ids))

  def generate(self, messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
    return self.generate_many([messages], temperature=temperature, seed=seed)[0]

  def generate_many(
      self,
      messages_batch: Sequence[list[dict[str, str]]],
      *,
      temperature: float,
      seed: int,
      max_prompt_length: int | None = None,
  ) -> list[str]:
    prompts = [
        self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]
    temperature, top_k, top_p = self._resolve_sampling_controls(
        temperature=temperature
    )
    output = self._sampler(
        input_strings=prompts,
        max_generation_steps=self._max_generation_steps,
        max_prompt_length=(
            self._max_prompt_length
            if max_prompt_length is None
            else max_prompt_length
        ),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        eos_tokens=self._stop_token_ids,
        seed=self._jax.random.PRNGKey(seed),
    )
    return [text.strip() for text in output.text]

  def count_tokens(self, text: str) -> int:
    return len(self._tokenizer.encode(text, add_special_tokens=False))

  def close(self) -> None:
    try:
      del self._sampler
      del self._tokenizer
      del self._model
      gc.collect()
      self._jax.clear_caches()
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def _batched_range(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
  for start in range(0, total, batch_size):
    yield start, min(total, start + batch_size)


def _chunked_generate(
    generator: TunixChatGenerator,
    *,
    prompts: Sequence[str],
    temperature: float,
    seed: int,
    batch_size: int,
    max_prompt_length: int | None = None,
) -> list[str]:
  outputs: list[str] = []
  for batch_index, (start, end) in enumerate(_batched_range(len(prompts), batch_size)):
    batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts[start:end]]
    outputs.extend(
        generator.generate_many(
            batch_messages,
            temperature=temperature,
            seed=seed + batch_index,
            max_prompt_length=max_prompt_length,
        )
    )
  return outputs


def _generate_mt_bench(
    generator: TunixChatGenerator,
    *,
    questions: Sequence[dict[str, Any]],
    answer_file: Path,
    model_id: str,
    seed: int,
) -> int:
  rows = []
  for index, question in enumerate(questions):
    messages: list[dict[str, str]] = []
    turns = []
    temperature = MT_BENCH_TEMPERATURE.get(question["category"], 0.7)
    for turn_index, user_turn in enumerate(question["turns"]):
      messages.append({"role": "user", "content": user_turn})
      answer = generator.generate(
          messages,
          temperature=temperature,
          seed=seed + index * 31 + turn_index,
      )
      turns.append(answer)
      messages.append({"role": "assistant", "content": answer})
    rows.append(
        {
            "question_id": question["question_id"],
            "answer_id": f"{model_id}-{index:04d}",
            "model_id": model_id,
            "choices": [{"index": 0, "turns": turns}],
            "tstamp": time.time(),
        }
    )
  return _write_jsonl(answer_file, rows)


def _alpaca_prompt(record: dict[str, Any]) -> str:
  prompt = record["instruction"]
  if record.get("input"):
    prompt = f"{prompt}\n\n{record['input']}"
  return prompt


def _generate_alpacaeval(
    generator: TunixChatGenerator,
    *,
    prompts: Sequence[dict[str, Any]],
    output_file: Path,
    model_name: str,
    seed: int,
    batch_size: int,
) -> int:
  rows = []
  if output_file.exists():
    rows = json.loads(output_file.read_text())
  prompt_strings = [_alpaca_prompt(record) for record in prompts]
  for batch_index, (start, end) in enumerate(
      _batched_range(len(prompt_strings), batch_size)
  ):
    if end <= len(rows):
      continue
    answers = _chunked_generate(
        generator,
        prompts=prompt_strings[start:end],
        temperature=0.0,
        seed=seed + batch_index,
        batch_size=batch_size,
    )
    for record, answer in zip(prompts[start:end], answers, strict=True):
      rows.append(
          {
              "dataset": record.get("dataset"),
              "instruction": record["instruction"],
              "output": answer,
              "generator": model_name,
          }
      )
    _ensure_parent(output_file)
    output_file.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
  return len(rows)


def _empty_markdown_metadata(token_len: int) -> dict[str, Any]:
  return {
      "token_len": token_len,
      "header_count": {"h1": 0, "h2": 0, "h3": 0, "h4": 0, "h5": 0, "h6": 0},
      "list_count": {"ordered": 0, "unordered": 0},
      "bold_count": {"**": 0, "__": 0},
  }


def _generate_arena_hard(
    generator: TunixChatGenerator,
    *,
    questions: Sequence[dict[str, Any]],
    answer_file: Path,
    model_name: str,
    seed: int,
    batch_size: int,
) -> int:
  rows = _load_jsonl(answer_file) if answer_file.exists() else []
  prompt_strings = [question["prompt"] for question in questions]
  for batch_index, (start, end) in enumerate(
      _batched_range(len(prompt_strings), batch_size)
  ):
    if end <= len(rows):
      continue
    answers = _chunked_generate(
        generator,
        prompts=prompt_strings[start:end],
        temperature=0.0,
        seed=seed + batch_index,
        batch_size=batch_size,
        max_prompt_length=ARENA_HARD_PROMPT_LENGTH_BUDGET,
    )
    for index, (question, answer) in enumerate(
        zip(questions[start:end], answers, strict=True),
        start=start,
    ):
      rows.append(
          {
              "uid": question["uid"],
              "ans_id": f"{model_name}-{index:04d}",
              "model": model_name,
              "messages": [
                  {"role": "user", "content": question["prompt"]},
                  {"role": "assistant", "content": {"answer": answer}},
              ],
              "tstamp": time.time(),
              "metadata": _empty_markdown_metadata(generator.count_tokens(answer)),
          }
      )
    _write_jsonl(answer_file, rows)
  return len(rows)


def _generate_ifeval(
    generator: TunixChatGenerator,
    *,
    prompts: Sequence[dict[str, Any]],
    output_file: Path,
    seed: int,
    batch_size: int,
) -> int:
  rows = _load_jsonl(output_file) if output_file.exists() else []
  prompt_strings = [record["prompt"] for record in prompts]
  for batch_index, (start, end) in enumerate(
      _batched_range(len(prompt_strings), batch_size)
  ):
    if end <= len(rows):
      continue
    answers = _chunked_generate(
        generator,
        prompts=prompt_strings[start:end],
        temperature=0.0,
        seed=seed + batch_index,
        batch_size=batch_size,
    )
    for record, answer in zip(prompts[start:end], answers, strict=True):
      rows.append(
          {
              "key": record["key"],
              "prompt": record["prompt"],
              "response": answer,
          }
      )
    _write_jsonl(output_file, rows)
  return len(rows)


def _generate_ifbench(
    generator: TunixChatGenerator,
    *,
    prompts: Sequence[dict[str, Any]],
    output_file: Path,
    seed: int,
    batch_size: int,
) -> int:
  return _generate_ifeval(
      generator,
      prompts=prompts,
      output_file=output_file,
      seed=seed,
      batch_size=batch_size,
  )


def _collect_generation_plan(
    assets: dict[str, dict[str, str]],
    questions_limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
  output: dict[str, list[dict[str, Any]]] = {}
  if "mt_bench" in assets:
    output["mt_bench"] = _load_jsonl(
        Path(assets["mt_bench"]["question_path"]), questions_limit
    )
  if "alpacaeval2" in assets:
    alpaca_rows = json.loads(
        Path(assets["alpacaeval2"]["reference_output_path"]).read_text()
    )
    if questions_limit is not None:
      alpaca_rows = alpaca_rows[:questions_limit]
    output["alpacaeval2"] = alpaca_rows
  if "arena_hard" in assets:
    output["arena_hard"] = _load_jsonl(
        Path(assets["arena_hard"]["question_path"]), questions_limit
    )
  if "ifeval" in assets:
    output["ifeval"] = _load_jsonl(Path(assets["ifeval"]["input_path"]), questions_limit)
  if "ifbench" in assets:
    output["ifbench"] = _load_jsonl(
        Path(assets["ifbench"]["input_path"]), questions_limit
    )
  return output


def _shell_join(parts: Sequence[str]) -> str:
  return " ".join(shlex_quote(part) for part in parts)


def shlex_quote(value: str) -> str:
  import shlex

  return shlex.quote(value)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def _find_latest_matching_file(directory: Path, pattern: str) -> Path | None:
  candidates = sorted(
      directory.glob(pattern),
      key=lambda item: item.stat().st_mtime,
      reverse=True,
  )
  return candidates[0] if candidates else None


def _extract_mt_bench_scores(
    workspace: Path,
    methods: Sequence[str],
) -> dict[str, float | None]:
  scores = {method: None for method in methods}
  judgment_dir = workspace / "data/mt_bench/model_judgment"
  judgment_file = _find_latest_matching_file(judgment_dir, "*_single.jsonl")
  if judgment_file is None:
    return scores
  totals = {method: 0.0 for method in methods}
  counts = {method: 0 for method in methods}
  latest_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
  for row in _load_jsonl(judgment_file):
    method = row.get("model")
    score = row.get("score")
    if method not in totals or score in (None, -1):
      continue
    key = (
        row.get("question_id"),
        method,
        row.get("turn"),
        tuple(row.get("judge", [])),
    )
    latest_rows[key] = row
  for row in latest_rows.values():
    method = row.get("model")
    score = row.get("score")
    if method not in totals or score in (None, -1):
      continue
    totals[method] += float(score)
    counts[method] += 1
  for method in methods:
    if counts[method]:
      scores[method] = totals[method] / counts[method]
  return scores


def _extract_alpacaeval_scores(
    workspace: Path,
    methods: Sequence[str],
) -> dict[str, float | None]:
  scores = {method: None for method in methods}
  results_root = workspace / "results"
  if not results_root.exists():
    return scores
  for method in methods:
    leaderboard_file = _find_latest_matching_file(
        results_root / method, "**/leaderboard.csv"
    )
    if leaderboard_file is None:
      continue
    rows = _read_csv_rows(leaderboard_file)
    if not rows:
      continue
    row = rows[0]
    for candidate in rows:
      if method in candidate.values():
        row = candidate
        break
    value = row.get("length_controlled_winrate")
    if value is not None and value != "":
      scores[method] = float(value)
  return scores


def _extract_arena_hard_scores(
    workspace: Path,
    methods: Sequence[str],
) -> dict[str, float | None]:
  scores = {method: None for method in methods}
  results_root = workspace / "results"
  result_file = _find_latest_matching_file(results_root, "arena_hard_*.txt")
  if result_file is None:
    return scores
  pattern = re.compile(
      r"\b("
      + "|".join(re.escape(method) for method in methods)
      + r")\b\s+([0-9]+(?:\.[0-9]+)?)"
  )
  for line in result_file.read_text(encoding="utf-8").splitlines():
    match = pattern.search(line)
    if match is None:
      continue
    scores[match.group(1)] = float(match.group(2))
  return scores


def _compute_instruction_following_scores(
    strict_rows: Sequence[dict[str, Any]],
    loose_rows: Sequence[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, float]:
  def _prompt_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    if not rows:
      return 0.0
    return sum(bool(row["follow_all_instructions"]) for row in rows) / len(rows)

  def _instruction_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    total = 0
    correct = 0
    for row in rows:
      follow_list = row.get("follow_instruction_list", [])
      total += len(follow_list)
      correct += sum(bool(value) for value in follow_list)
    if total <= 0:
      return 0.0
    return correct / total

  return {
      f"{prefix}_prompt_strict": _prompt_accuracy(strict_rows),
      f"{prefix}_prompt_loose": _prompt_accuracy(loose_rows),
      f"{prefix}_instruction_strict": _instruction_accuracy(strict_rows),
      f"{prefix}_instruction_loose": _instruction_accuracy(loose_rows),
  }


def _compute_ifeval_scores(
    strict_rows: Sequence[dict[str, Any]],
    loose_rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
  return _compute_instruction_following_scores(
      strict_rows,
      loose_rows,
      prefix="ifeval",
  )


def _run_ifeval_evaluator(
    *,
    assets: dict[str, str],
    model_name: str,
    offline_eval_venv: Path,
) -> dict[str, float]:
  python_path = offline_eval_venv / "bin/python"
  if not python_path.exists():
    raise SystemExit(
        "IFEval offline venv is missing python: "
        f"{python_path}. Create the offline env first."
    )
  response_file = Path(assets["response_dir"]) / f"{model_name}.jsonl"
  output_dir = Path(assets["result_dir"]) / model_name
  output_dir.mkdir(parents=True, exist_ok=True)
  env = dict(os.environ)
  env["PYTHONPATH"] = (
      str(Path(assets["official_root"]).parent)
      + os.pathsep
      + env.get("PYTHONPATH", "")
  )
  subprocess.run(
      [
          str(python_path),
          "-c",
          (
              "import nltk; "
              "nltk.download('punkt', quiet=True); "
              "nltk.download('punkt_tab', quiet=True)"
          ),
      ],
      check=True,
      env=env,
  )
  subprocess.run(
      [
          str(python_path),
          "-m",
          "instruction_following_eval.evaluation_main",
          f"--input_data={assets['input_path']}",
          f"--input_response_data={response_file}",
          f"--output_dir={output_dir}",
      ],
      check=True,
      cwd=str(Path(assets["official_root"]).parent),
      env=env,
  )
  strict_rows = _load_jsonl(output_dir / "eval_results_strict.jsonl")
  loose_rows = _load_jsonl(output_dir / "eval_results_loose.jsonl")
  scores = _compute_ifeval_scores(strict_rows, loose_rows)
  summary_path = output_dir / "ifeval_summary.json"
  summary_path.write_text(json.dumps(scores, indent=2, sort_keys=True))
  return scores


def _run_ifbench_evaluator(
    *,
    assets: dict[str, str],
    model_name: str,
    offline_eval_venv: Path,
) -> dict[str, float]:
  python_path = offline_eval_venv / "bin/python"
  if not python_path.exists():
    raise SystemExit(
        "IFBench offline venv is missing python: "
        f"{python_path}. Create the offline env first."
    )
  response_file = Path(assets["response_dir"]) / f"{model_name}.jsonl"
  output_dir = Path(assets["result_dir"]) / model_name
  output_dir.mkdir(parents=True, exist_ok=True)
  env = dict(os.environ)
  env["PYTHONPATH"] = (
      str(Path(assets["official_root"]))
      + os.pathsep
      + env.get("PYTHONPATH", "")
  )
  subprocess.run(
      [
          str(python_path),
          "-m",
          "run_eval",
          f"--input_data={assets['input_path']}",
          f"--input_response_data={response_file}",
          f"--output_dir={output_dir}",
      ],
      check=True,
      cwd=str(Path(assets["official_root"])),
      env=env,
  )
  strict_path = _find_latest_matching_file(output_dir, "*eval_results_strict.jsonl")
  loose_path = _find_latest_matching_file(output_dir, "*eval_results_loose.jsonl")
  if strict_path is None or loose_path is None:
    raise SystemExit(
        "IFBench evaluator did not produce the expected strict/loose outputs in "
        f"{output_dir}"
    )
  strict_rows = _load_jsonl(strict_path)
  loose_rows = _load_jsonl(loose_path)
  scores = _compute_instruction_following_scores(
      strict_rows,
      loose_rows,
      prefix="ifbench",
  )
  summary_path = output_dir / "ifbench_summary.json"
  summary_path.write_text(json.dumps(scores, indent=2, sort_keys=True))
  return scores


def _extract_ifeval_scores(
    workspace: Path,
    methods: Sequence[str],
) -> dict[str, dict[str, float | None]]:
  scores = {
      method: {
          "ifeval_prompt_strict": None,
          "ifeval_prompt_loose": None,
          "ifeval_instruction_strict": None,
          "ifeval_instruction_loose": None,
      }
      for method in methods
  }
  results_root = workspace / "results"
  for method in methods:
    summary_path = results_root / method / "ifeval_summary.json"
    if not summary_path.exists():
      continue
    row = json.loads(summary_path.read_text())
    for key in scores[method]:
      value = row.get(key)
      if value is not None:
        scores[method][key] = float(value)
  return scores


def _extract_ifbench_scores(
    workspace: Path,
    methods: Sequence[str],
) -> dict[str, dict[str, float | None]]:
  scores = {
      method: {
          "ifbench_prompt_strict": None,
          "ifbench_prompt_loose": None,
          "ifbench_instruction_strict": None,
          "ifbench_instruction_loose": None,
      }
      for method in methods
  }
  results_root = workspace / "results"
  for method in methods:
    summary_path = results_root / method / "ifbench_summary.json"
    if not summary_path.exists():
      continue
    row = json.loads(summary_path.read_text())
    for key in scores[method]:
      value = row.get(key)
      if value is not None:
        scores[method][key] = float(value)
  return scores


def _collect_existing_scores(
    *,
    assets: dict[str, dict[str, str]],
    methods: Sequence[str],
) -> dict[str, dict[str, float | None]]:
  scores = {
      method: {
          "mt_bench_avg_score": None,
          "alpacaeval2_lc_win_rate": None,
          "arena_hard_score": None,
          "ifeval_prompt_strict": None,
          "ifeval_prompt_loose": None,
          "ifeval_instruction_strict": None,
          "ifeval_instruction_loose": None,
          "ifbench_prompt_strict": None,
          "ifbench_prompt_loose": None,
          "ifbench_instruction_strict": None,
          "ifbench_instruction_loose": None,
      }
      for method in methods
  }
  if "mt_bench" in assets:
    mt_scores = _extract_mt_bench_scores(
        Path(assets["mt_bench"]["workspace"]), methods
    )
    for method, value in mt_scores.items():
      scores[method]["mt_bench_avg_score"] = value
  if "alpacaeval2" in assets:
    alpaca_scores = _extract_alpacaeval_scores(
        Path(assets["alpacaeval2"]["workspace"]), methods
    )
    for method, value in alpaca_scores.items():
      scores[method]["alpacaeval2_lc_win_rate"] = value
  if "arena_hard" in assets:
    arena_scores = _extract_arena_hard_scores(
        Path(assets["arena_hard"]["workspace"]), methods
    )
    for method, value in arena_scores.items():
      scores[method]["arena_hard_score"] = value
  if "ifeval" in assets:
    ifeval_scores = _extract_ifeval_scores(
        Path(assets["ifeval"]["workspace"]), methods
    )
    for method, row in ifeval_scores.items():
      scores[method].update(row)
  if "ifbench" in assets:
    ifbench_scores = _extract_ifbench_scores(
        Path(assets["ifbench"]["workspace"]), methods
    )
    for method, row in ifbench_scores.items():
      scores[method].update(row)
  return scores


def _apply_question_limit_to_assets(
    *,
    assets: dict[str, dict[str, str]],
    benchmark_questions: dict[str, list[dict[str, Any]]],
    question_limit: int | None,
) -> None:
  if question_limit is None:
    return
  if "mt_bench" in assets:
    _write_jsonl(
        Path(assets["mt_bench"]["question_path"]),
        benchmark_questions["mt_bench"],
    )
  if "arena_hard" in assets:
    _write_jsonl(
        Path(assets["arena_hard"]["question_path"]),
        benchmark_questions["arena_hard"],
    )
  if "ifeval" in assets:
    _write_jsonl(
        Path(assets["ifeval"]["input_path"]),
        benchmark_questions["ifeval"],
    )
  if "ifbench" in assets:
    _write_jsonl(
        Path(assets["ifbench"]["input_path"]),
        benchmark_questions["ifbench"],
    )


def _collect_score_assets(output_root: Path) -> dict[str, dict[str, str]]:
  assets: dict[str, dict[str, str]] = {}
  if (output_root / "mt_bench").exists():
    assets["mt_bench"] = _prepare_mt_bench_assets(output_root)
  if (output_root / "alpacaeval2").exists():
    assets["alpacaeval2"] = _prepare_alpacaeval_assets(output_root)
  if (output_root / "arena_hard").exists():
    assets["arena_hard"] = _prepare_arena_hard_assets(output_root)
  if (output_root / "ifeval").exists():
    assets["ifeval"] = _prepare_ifeval_assets(output_root)
  if (output_root / "ifbench").exists():
    assets["ifbench"] = _prepare_ifbench_assets(output_root)
  return assets


def _write_stage_b_commands(
    *,
    output_root: Path,
    assets: dict[str, dict[str, str]],
    runs: Sequence[dict[str, Any]],
    judge_model_family: str,
) -> Path:
  command_path = output_root / "stage_b_commands.sh"
  model_list = [run["variant"] for run in runs]
  commands = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      "",
      f'export MT_BENCH_JUDGE_VENV="${{MT_BENCH_JUDGE_VENV:-{DEFAULT_MT_BENCH_JUDGE_VENV}}}"',
      f'export OPENAI_JUDGE_VENV="${{OPENAI_JUDGE_VENV:-{DEFAULT_OPENAI_JUDGE_VENV}}}"',
      'export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is required for judge stage}"',
      "",
      "# Judge family label for bookkeeping only.",
      f'export JUDGE_MODEL_FAMILY="{judge_model_family}"',
      f'export MT_BENCH_JUDGE_MODEL="${{MT_BENCH_JUDGE_MODEL:-{DEFAULT_MT_BENCH_JUDGE_MODEL}}}"',
      f'export ALPACAEVAL_ANNOTATOR_CONFIG="${{ALPACAEVAL_ANNOTATOR_CONFIG:-{DEFAULT_ALPACAEVAL_ANNOTATOR}}}"',
      "",
  ]
  if "mt_bench" in assets:
    mt_workspace = Path(assets["mt_bench"]["workspace"])
    commands.extend(
        [
            "# MT-Bench official single-answer grading via FastChat.",
            "mkdir -p " + shlex_quote(str(mt_workspace / "results")),
            (
                "cd "
                + shlex_quote(str(mt_workspace))
                + " && "
                + _shell_join(
                    [
                        "bash",
                        "-lc",
                        (
                            'source "$MT_BENCH_JUDGE_VENV"/bin/activate && '
                            "printf '\\n' | python -m fastchat.llm_judge.gen_judgment "
                            "--bench-name mt_bench "
                            "--judge-file data/judge_prompts.jsonl "
                            '--judge-model "$MT_BENCH_JUDGE_MODEL" '
                            "--mode single "
                            f"--model-list {' '.join(model_list)}"
                        ),
                    ]
                )
            ),
            (
                "cd "
                + shlex_quote(str(mt_workspace))
                + " && "
                + _shell_join(
                    [
                        "bash",
                        "-lc",
                        (
                            'source "$MT_BENCH_JUDGE_VENV"/bin/activate && '
                            "python - <<'PY'\n"
                            "import json\n"
                            "import os\n"
                            "from collections import defaultdict\n"
                            "from pathlib import Path\n"
                            "judge_model = os.environ['MT_BENCH_JUDGE_MODEL']\n"
                            "path = Path('data/mt_bench/model_judgment') / f'{judge_model}_single.jsonl'\n"
                            "totals = defaultdict(float)\n"
                            "counts = defaultdict(int)\n"
                            "for line in path.read_text(encoding='utf-8').splitlines():\n"
                            "  if not line.strip():\n"
                            "    continue\n"
                            "  row = json.loads(line)\n"
                            "  score = row.get('score')\n"
                            "  if score in (None, -1):\n"
                            "    continue\n"
                            "  totals[row['model']] += float(score)\n"
                            "  counts[row['model']] += 1\n"
                            "lines = [f'MT-Bench judge: {judge_model}']\n"
                            f"for model in {repr(model_list)}:\n"
                            "  avg = totals[model] / counts[model] if counts[model] else float('nan')\n"
                            "  lines.append(f'{model}: {avg:.4f}')\n"
                            "text = '\\n'.join(lines) + '\\n'\n"
                            "Path('results').mkdir(parents=True, exist_ok=True)\n"
                            "summary_path = Path('results') / f'mt_bench_{judge_model}.txt'\n"
                            "summary_path.write_text(text, encoding='utf-8')\n"
                            "print(text, end='')\n"
                            "PY"
                        ),
                    ]
                )
            ),
            "",
        ]
    )
  if "alpacaeval2" in assets:
    commands.extend(
        [
            "# AlpacaEval 2 weighted length-controlled win rate via alpaca_eval.",
            "for MODEL_NAME in " + " ".join(model_list) + "; do",
            (
                "  MODEL_NAME=\"$MODEL_NAME\" "
                + _shell_join(
                    [
                        "bash",
                        "-lc",
                        (
                            'source "$OPENAI_JUDGE_VENV"/bin/activate && '
                            "python - <<'PY'\n"
                            "import os\n"
                            "from alpaca_eval import evaluate\n"
                            "model_name = os.environ['MODEL_NAME']\n"
                            "evaluate(\n"
                            f"    model_outputs='{assets['alpacaeval2']['model_output_dir']}/' + model_name + '.json',\n"
                            f"    reference_outputs='{assets['alpacaeval2']['reference_output_path']}',\n"
                            "    annotators_config=os.environ['ALPACAEVAL_ANNOTATOR_CONFIG'],\n"
                            "    name=model_name,\n"
                            f"    output_path='{assets['alpacaeval2']['workspace']}/results/' + model_name,\n"
                            ")\n"
                            "PY"
                        ),
                    ]
                )
            ),
            "done",
            "",
        ]
    )
  if "arena_hard" in assets:
    arena_workspace = Path(assets["arena_hard"]["workspace"])
    arena_repo = ARENA_HARD_CLONE
    commands.extend(
        [
            "# Arena-Hard official GPT-4.1 judge + score aggregation.",
            "mkdir -p " + shlex_quote(str(output_root / "arena_hard" / "results")),
            (
                "rsync -a "
                + shlex_quote(str(arena_workspace / "data") + "/")
                + " "
                + shlex_quote(str(arena_repo / "data") + "/")
            ),
            (
                "cd "
                + shlex_quote(str(arena_repo))
                + " && "
                + _shell_join(
                    [
                        "bash",
                        "-lc",
                        (
                            'source "$OPENAI_JUDGE_VENV"/bin/activate && '
                            "python gen_judgment.py "
                            f"--setting-file {output_root / 'arena_hard' / 'arena-hard-v2.0.generated.yaml'} "
                            f"--endpoint-file {output_root / 'arena_hard' / 'api_config.generated.yaml'}"
                        ),
                    ]
                )
            ),
            (
                "cd "
                + shlex_quote(str(arena_repo))
                + " && "
                + _shell_join(
                    [
                        "bash",
                        "-lc",
                        (
                            'source "$OPENAI_JUDGE_VENV"/bin/activate && '
                            "python show_result.py "
                            "--benchmark arena-hard-v2.0 "
                            "--judge-names gpt-4.1 "
                            "--control-features markdown length "
                            '--category hard_prompt | tee "'
                            + str(output_root / "arena_hard" / "results" / "arena_hard_gpt-4.1.txt")
                            + '"'
                        ),
                    ]
                )
            ),
            "",
        ]
    )
  command_path.write_text("\n".join(commands))
  command_path.chmod(0o755)
  return command_path


def _write_arena_hard_generated_configs(
    *,
    output_root: Path,
    runs: Sequence[dict[str, Any]],
) -> dict[str, str]:
  arena_root = output_root / "arena_hard"
  setting_path = arena_root / "arena-hard-v2.0.generated.yaml"
  api_path = arena_root / "api_config.generated.yaml"
  model_list = "\n".join(f"  - {run['variant']}" for run in runs)
  setting_path.write_text(
      "\n".join(
          [
              "judge_model: gpt-4.1",
              "temperature: 0.0",
              "max_tokens: 16000",
              "bench_name: arena-hard-v2.0",
              "reference: null",
              "",
              "regex_patterns:",
              r"  - \[\[([AB<>=]+)\]\]",
              r"  - \[([AB<>=]+)\]",
              "",
              'prompt_template: "<|User Prompt|>\\n{QUESTION}\\n\\n<|The Start of Assistant A\'s Answer|>\\n{ANSWER_A}\\n<|The End of Assistant A\'s Answer|>\\n\\n<|The Start of Assistant B\'s Answer|>\\n{ANSWER_B}\\n<|The End of Assistant B\'s Answer|>"',
              "",
              "model_list:",
              model_list,
              "",
          ]
      )
  )
  api_path.write_text(
      "\n".join(
          [
              "gpt-4.1:",
              "    model: gpt-4.1",
              "    endpoints: null",
              "    api_type: openai",
              "    parallel: 16",
              "    max_tokens: 16000",
              "    temperature: 0.0",
              "",
          ]
      )
  )
  return {"setting_path": str(setting_path), "api_path": str(api_path)}


def main() -> None:
  args = _parse_args()
  repo_root = Path(args.repo_root).resolve()
  output_root = (
      Path(args.output_root).resolve()
      if args.output_root
      else repo_root / "runs/results" / f"clean_benchmarks_{args.run_ts}"
  )
  output_root.mkdir(parents=True, exist_ok=True)
  summary_path = output_root / "benchmark_summary.json"
  previous_summary = (
      json.loads(summary_path.read_text()) if summary_path.exists() else None
  )

  methods = [_normalize_method(method) for method in args.methods]
  benchmarks = [_normalize_benchmark(name) for name in args.benchmarks]
  runs = _discover_clean_runs(repo_root, args.run_ts, methods)
  score_methods = [run["variant"] for run in runs]

  assets: dict[str, dict[str, str]] = {}
  if "mt_bench" in benchmarks:
    assets["mt_bench"] = _prepare_mt_bench_assets(output_root)
  if "alpacaeval2" in benchmarks:
    assets["alpacaeval2"] = _prepare_alpacaeval_assets(output_root)
  if "arena_hard" in benchmarks:
    assets["arena_hard"] = _prepare_arena_hard_assets(output_root)
  if "ifeval" in benchmarks:
    assets["ifeval"] = _prepare_ifeval_assets(output_root)
  if "ifbench" in benchmarks:
    assets["ifbench"] = _prepare_ifbench_assets(output_root)

  benchmark_questions = _collect_generation_plan(assets, args.question_limit)
  _apply_question_limit_to_assets(
      assets=assets,
      benchmark_questions=benchmark_questions,
      question_limit=args.question_limit,
  )
  score_assets = _collect_score_assets(output_root)
  generation_summary: dict[str, Any] = {
      "run_ts": args.run_ts,
      "judge_model_family": args.judge_model_family,
      "output_root": str(output_root),
      "question_limit": args.question_limit,
      "scores": _collect_existing_scores(assets=score_assets, methods=score_methods),
      "runs": runs,
      "benchmarks": dict(previous_summary.get("benchmarks", {}))
      if previous_summary
      else {},
  }

  if "arena_hard" in benchmarks:
    arena_config_paths = _write_arena_hard_generated_configs(
        output_root=output_root,
        runs=runs,
    )
  else:
    arena_config_paths = {"setting_path": None, "api_path": None}
  stage_b_path = _write_stage_b_commands(
      output_root=output_root,
      assets=assets,
      runs=runs,
      judge_model_family=args.judge_model_family,
  )

  for benchmark in benchmarks:
    generation_summary["benchmarks"][benchmark] = {
        "workspace": assets[benchmark]["workspace"],
        "question_count": len(benchmark_questions[benchmark]),
        "models": {},
    }

  if not args.skip_generation:
    for run in runs:
      generator = TunixChatGenerator(
          config_path=args.config_path,
          exported_model_path=run["exported_model_path"],
          max_prompt_length=args.max_prompt_length,
          max_generation_steps=args.max_generation_steps,
          top_k=args.top_k,
          top_p=args.top_p,
      )
      try:
        if "mt_bench" in benchmarks:
          answer_file = (
              Path(assets["mt_bench"]["answer_dir"]) / f"{run['variant']}.jsonl"
          )
          if not answer_file.exists() or args.force:
            count = _generate_mt_bench(
                generator,
                questions=benchmark_questions["mt_bench"],
                answer_file=answer_file,
                model_id=run["variant"],
                seed=args.seed,
            )
          else:
            count = len(_load_jsonl(answer_file))
          generation_summary["benchmarks"]["mt_bench"]["models"][run["variant"]] = {
              "answer_file": str(answer_file),
              "num_outputs": count,
          }

        if "alpacaeval2" in benchmarks:
          output_file = (
              Path(assets["alpacaeval2"]["model_output_dir"])
              / f"{run['variant']}.json"
          )
          if not output_file.exists() or args.force:
            count = _generate_alpacaeval(
                generator,
                prompts=benchmark_questions["alpacaeval2"],
                output_file=output_file,
                model_name=run["variant"],
                seed=args.seed,
                batch_size=args.generation_batch_size,
            )
          else:
            count = len(json.loads(output_file.read_text()))
          generation_summary["benchmarks"]["alpacaeval2"]["models"][run["variant"]] = {
              "output_file": str(output_file),
              "num_outputs": count,
          }

        if "arena_hard" in benchmarks:
          answer_file = (
              Path(assets["arena_hard"]["answer_dir"]) / f"{run['variant']}.jsonl"
          )
          if not answer_file.exists() or args.force:
            count = _generate_arena_hard(
                generator,
                questions=benchmark_questions["arena_hard"],
                answer_file=answer_file,
                model_name=run["variant"],
                seed=args.seed,
                batch_size=1,
            )
          else:
            count = len(_load_jsonl(answer_file))
          generation_summary["benchmarks"]["arena_hard"]["models"][run["variant"]] = {
              "answer_file": str(answer_file),
              "num_outputs": count,
          }

        if "ifeval" in benchmarks:
          output_file = (
              Path(assets["ifeval"]["response_dir"]) / f"{run['variant']}.jsonl"
          )
          existing_count = len(_load_jsonl(output_file)) if output_file.exists() else 0
          if args.force or existing_count != len(benchmark_questions["ifeval"]):
            if args.force and output_file.exists():
              output_file.unlink()
            count = _generate_ifeval(
                generator,
                prompts=benchmark_questions["ifeval"],
                output_file=output_file,
                seed=args.seed,
                batch_size=args.generation_batch_size,
            )
          else:
            count = existing_count
          ifeval_scores = _run_ifeval_evaluator(
              assets=assets["ifeval"],
              model_name=run["variant"],
              offline_eval_venv=Path(args.offline_eval_venv).resolve(),
          )
          generation_summary["benchmarks"]["ifeval"]["models"][run["variant"]] = {
              "response_file": str(output_file),
              "num_outputs": count,
              **ifeval_scores,
          }

        if "ifbench" in benchmarks:
          output_file = (
              Path(assets["ifbench"]["response_dir"]) / f"{run['variant']}.jsonl"
          )
          existing_count = len(_load_jsonl(output_file)) if output_file.exists() else 0
          if args.force or existing_count != len(benchmark_questions["ifbench"]):
            if args.force and output_file.exists():
              output_file.unlink()
            count = _generate_ifbench(
                generator,
                prompts=benchmark_questions["ifbench"],
                output_file=output_file,
                seed=args.seed,
                batch_size=args.generation_batch_size,
            )
          else:
            count = existing_count
          ifbench_scores = _run_ifbench_evaluator(
              assets=assets["ifbench"],
              model_name=run["variant"],
              offline_eval_venv=Path(args.offline_eval_venv).resolve(),
          )
          generation_summary["benchmarks"]["ifbench"]["models"][run["variant"]] = {
              "response_file": str(output_file),
              "num_outputs": count,
              **ifbench_scores,
          }
      finally:
        generator.close()

  generation_summary["stage_b"] = {
      "command_file": str(stage_b_path),
      "arena_hard_generated_setting": arena_config_paths["setting_path"],
      "arena_hard_generated_api_config": arena_config_paths["api_path"],
  }
  generation_summary["scores"] = _collect_existing_scores(
      assets=score_assets, methods=score_methods
  )

  summary_path.write_text(json.dumps(generation_summary, indent=2, sort_keys=True))

  print(f"Wrote benchmark summary to {summary_path}")
  print(f"Wrote stage-B commands to {stage_b_path}")
  if args.skip_generation:
    print("Generation skipped; workspaces and judge plans are ready.")
  else:
    for benchmark in benchmarks:
      print(
          f"{benchmark}: {generation_summary['benchmarks'][benchmark]['question_count']} prompts"
      )


if __name__ == "__main__":
  main()
