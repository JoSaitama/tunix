# %%
import argparse
import contextlib
import os
from pprint import pprint
import subprocess
from typing import Any, Dict, Optional, Sequence

import datasets as datasets_lib
import fsspec
import grain
import jax
import pandas as pd
import re
from tqdm.auto import tqdm
import transformers
from tunix.generate import mappings

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer

try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  from etils import ecolab

  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload="tunix",
      behavior="preferred",
      cell_autoreload=True,
  )

  file_open = gfile.Open

  NOTEBOOK_ENV = "g3"
except Exception:
  NOTEBOOK_ENV = "git"
  cm = contextlib.nullcontext()

  file_open = fsspec.open

with cm:
  from tunix.models.qwen2 import model as qwen2_lib
  from tunix.models.qwen2 import params as qwen2_params_lib
  from tunix.generate import sampler as sampler_lib
  from tunix.utils import math_utils
# %%

REMOTE_PREFIXES = ("gs://", "gcs://", "s3://", "http://", "https://", "hf://")


def _is_remote(path: str) -> bool:
  return path.startswith(REMOTE_PREFIXES)


def _is_gcs(path: str) -> bool:
  return path.startswith("gs://") or path.startswith("gcs://")


def _is_hf_repo_id(path_or_id: str) -> bool:
  if os.path.isabs(path_or_id) or path_or_id.startswith("."):
    return False
  return ("/" in path_or_id) and (not _is_remote(path_or_id)) and (not os.path.exists(path_or_id))


def _check_gcp_auth() -> bool:
  credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
  if credentials_path and os.path.exists(credentials_path):
    return True
  try:
    subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    return True
  except Exception:
    return False

# Only used for Math500
def extract_answer_robust(passage: str) -> str:
  if not passage:
    return ""

  # Pattern 1: Look for \boxed{...} with proper matching braces
  # This handles nested braces like \boxed{\frac{1}{2}}
  stack = []
  i = passage.find("\\boxed")
  if i != -1:
    i += 6  # Skip '\boxed'
    # Skip whitespace
    while i < len(passage) and passage[i].isspace():
      i += 1
    if i < len(passage) and passage[i] == "{":
      i += 1
      start = i
      brace_count = 1
      while i < len(passage) and brace_count > 0:
        if passage[i] == "{":
          brace_count += 1
        elif passage[i] == "}":
          brace_count -= 1
        i += 1
      if brace_count == 0:
        answer = passage[start : i - 1]
        return answer.strip()

  # Pattern 2: Lenient matching - extract up to common terminators
  patterns = [
      r"\\boxed\{([^}]+)\}",  # Standard
      r"boxed\{([^}]+)\}",  # Missing backslash
      r"\\boxed\s*\{(.+?)(?:\.\s|\)\.|\.$)",  # Ends with period
      r"final answer is.*?\\boxed\{([^}]+)",  # "final answer is"
      r"answer is.*?\\boxed\{([^}]+)",
  ]

  for pattern in patterns:
    matches = re.findall(pattern, passage, re.IGNORECASE | re.DOTALL)
    if matches:
      answer = matches[-1].strip()
      # Clean up
      answer = answer.rstrip(".,;:)")
      # Try to fix common LaTeX issues
      if "\\frac" in answer:
        # Count braces - each \frac needs 2 pairs
        open_braces = answer.count("{")
        close_braces = answer.count("}")
        if open_braces > close_braces:
          answer += "}" * (open_braces - close_braces)
      return answer

  # Pattern 3: Super lenient - just find anything after boxed{
  super_lenient = r"boxed\s*\{([^\n]{1,200})"
  matches = re.findall(super_lenient, passage, re.IGNORECASE)
  if matches:
    answer = matches[-1]
    # Find the first reasonable endpoint
    for char in [".", ")", "\n", "The ", "Thus", "Therefore"]:
      if char in answer:
        answer = answer[: answer.index(char)]
        break
    return answer.strip().rstrip(".,;:)")

  return ""
# %%

# only used for AIME-2024
THOUGHT_DELIMITER_END = "</think>"
def evaluate_correctness(response: Any, ground_truths: Any) -> bool:
  """Evaluate the correctness of a response."""
  if response is None or response == "":
    print(f"{response=} {ground_truths=} IS NOT CORRECT")
    return False
  if THOUGHT_DELIMITER_END in response:
    response = response.split(THOUGHT_DELIMITER_END)
    model_solution = response[1]
    print(f"{model_solution=} after THOUGHT_DELIMITER_END in evaluate_correctness")
  else:
    print(f"{response=} in evaluate_correctness")
    model_solution = response
  model_answer = math_utils.extract_answer(model_solution)
  if model_answer is None:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  if ground_truths is None:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  # Convert single answer to list for uniform processing
  if isinstance(ground_truths, str | float | int):
    ground_truths = [ground_truths]
  # Process each ground truth
  processed_ground_truths = []
  for truth in ground_truths:
    truth = str(truth)
    if "\\boxed" in truth:
      processed_truth = math_utils.extract_answer(truth)
      if processed_truth is not None:
        processed_ground_truths.append(processed_truth)
    else:
      processed_ground_truths.append(truth)
  print(f"{processed_ground_truths=} in evaluate_correctness")
  if not processed_ground_truths:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  # Check against all possible correct answers
  for ground_truth in processed_ground_truths:
    is_correct = math_utils.grade_answer_mathd(
        model_answer, ground_truth
    ) or math_utils.grade_answer_sympy(model_answer, ground_truth)
    if is_correct:
      print(f" {model_answer=} {ground_truth=} IS CORRECT")
      return True
  print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
  return False
# %%

class Qwen25MathEvaluator:

  def __init__(
      self,
      model_config,
      model_version: str,
      model_path: str,
      dataset: str,
      mesh_config=None,
      max_prompt_length: int = 1024,  # Increased from 512
      max_generation_steps: int = 1024,  # Increased from 512
      sampler_type: str = "vanilla",  # vanilla, vllm, or sglang-jax
      tokenizer_source: str | None = None,
      hf_token: str | None = None,
  ):
    self.model_config = model_config
    self.model_version = model_version
    self.model_path = model_path
    self.dataset = dataset
    self.max_prompt_length = max_prompt_length
    self.max_generation_steps = max_generation_steps
    self.sampler_type = sampler_type
    self.tokenizer_source = tokenizer_source
    self.hf_token = hf_token

    if mesh_config is None:
      mesh_config = [[1, max(1, jax.device_count())], ["fsdp", "tp"]]
    mesh_size = mesh_config[0][0] * mesh_config[0][1]
    if mesh_size != max(1, jax.device_count()):
      raise ValueError(
          f"mesh size mismatch: {mesh_config[0][0]}x{mesh_config[0][1]}={mesh_size}, "
          f"but jax.device_count()={jax.device_count()}."
      )
    self.mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))
    self.tokenizer = None
    self.model = None
    self.sampler = None
    self.eos_token_ids: list[int] | None = None

    print(f"Initializing {self.model_version} evaluator")
    print(f"Model path: {model_path}")
    print(f"Mesh config: {mesh_config}")
    print(f"Available devices: {jax.devices()}")

  def _resolve_eos_token_ids(self) -> list[int]:
    if self.tokenizer is None:
      raise RuntimeError("Tokenizer must be loaded before resolving EOS tokens.")

    candidate_ids = []

    eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
      candidate_ids.append(int(eos_token_id))

    # Keep compatibility with chat tokenizers that expose legacy "<|im_end|>".
    for tok in ("<|im_end|>", "<｜end▁of▁sentence｜>"):
      tok_ids = self.tokenizer.encode(tok, add_special_tokens=False)
      if len(tok_ids) == 1:
        candidate_ids.append(int(tok_ids[0]))

    eos_ids = list(dict.fromkeys(candidate_ids))
    if not eos_ids:
      raise RuntimeError(
          "Failed to infer EOS token id(s). Check tokenizer special tokens."
      )
    return eos_ids

  def load_model(self):
    print("Loading model components...")

    print("Loading tokenizer...")

    # Huggingface API doesn't work with gcs, OSS loads from model directly
    tokenizer_source = self.tokenizer_source
    if tokenizer_source is None:
      tokenizer_source = self.model_version if NOTEBOOK_ENV != "g3" else self.model_path
    tokenizer_kwargs = {"trust_remote_code": True}
    if self.hf_token:
      try:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, token=self.hf_token, **tokenizer_kwargs
        )
      except TypeError:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, use_auth_token=self.hf_token, **tokenizer_kwargs
        )
    else:
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    self.eos_token_ids = self._resolve_eos_token_ids()
    print(f"Using EOS token ids: {self.eos_token_ids}")

    print("Setting up model config...")


    print("Loading model from safe tensors...")
    with self.mesh:
      self.model = qwen2_params_lib.create_model_from_safe_tensors(
          file_dir=self.model_path, config=self.model_config, mesh=self.mesh
      )

    print("Model loaded successfully!")
    print("Creating sampler...")
    cache_config = sampler_lib.CacheConfig(
        cache_size=self.max_prompt_length + self.max_generation_steps + 100,
        num_layers=self.model_config.num_layers,
        num_kv_heads=self.model_config.num_kv_heads,
        head_dim=self.model_config.head_dim,
    )

    if self.sampler_type == "vanilla":
      self.sampler_vanilla = sampler_lib.Sampler(
          transformer=self.model,
          tokenizer=self.tokenizer,
          cache_config=cache_config,
      )
    elif self.sampler_type == "sglang-jax":
      from tunix.google.stubs import sglang_jax_sampler_stub as sglang_jax_sampler  # pylint: disable=g-import-not-at-top

      mapping_config = mappings.MappingConfig.build(
          mapping_obj=None,
          model=self.model,
          backend="sglang_jax",
      )
      self.sampler_sglang = sglang_jax_sampler.SglangJaxSampler(
          tokenizer=self.tokenizer,
          config=sglang_jax_sampler.SglangJaxConfig(
              mesh=self.mesh,
              context_length=self.max_prompt_length
              + self.max_generation_steps
              + 100,
              model_version=self.model_version,
              mem_fraction_static=0.4,
              init_with_random_weights=False,
              disable_radix_cache=True,
              enable_deterministic_sampling=False,
              mapping_config=mapping_config,
          ),
      )
    else:
      raise ValueError(f"Unsupported sampler type: {self.sampler_type}")

    print("Sampler created successfully!")

    return {
        "model": self.model,
        "tokenizer": self.tokenizer,
        "sampler": self.sampler,
        "config": self.model_config,
    }

  def load_dataset(self, split: str = "test") -> grain.MapDataset:
    print(f"Loading {self.dataset} dataset (split: {split})...")

    def preprocess_fn(example, idx):
      return {
          "question": example["problem"],
          "answer": example["answer"],
          "data_source": "math",
          }

    with file_open(self.dataset, "rb") as test_f:
      if self.dataset.endswith("jsonl"):
        test_df = pd.read_json(test_f, lines=True)
      elif self.dataset.endswith("json"):
        test_df = pd.read_json(test_f)
      else:
        test_df = pd.read_parquet(test_f)

    test_ds = Dataset.from_pandas(test_df).map(preprocess_fn, with_indices=True)


    print(f"Loaded {len(test_ds)} examples")
    print("Example data:")
    pprint(test_ds[0])

    def process_item(item):
      question = item["question"]
      answer = item["answer"]

      if "aime_2024" in self.dataset:
        instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        prompt = f"{question} {instruction}"
      else:
        instruction = "Please reason step by step. Your final answer must appear inside \\boxed{...} and nothing else."
        prompt = f"{instruction} {question}"
      prompt = self.tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}],
          tokenize=False, add_generation_prompt=True)

      return {
          "prompt": prompt,
          "question": question,
          "answer": answer,
      }

    dataset = grain.MapDataset.source(test_ds).map(process_item)
    print("\n" + "=" * 60)
    print("DEBUG: First formatted prompt:")
    first_item = dataset[0]
    print(first_item["prompt"])
    print("=" * 60 + "\n")

    return dataset

  def generate(
      self,
      prompts: list[str],
      temperature: float = 0.6,
      top_k: int = 50,
      top_p: float = 0.95,
      seed: int | None = None,
  ) -> list[str]:
    if self.tokenizer is None:
      raise RuntimeError(
          "Model components not loaded. Call load_model() first."
      )
    max_length = max(len(self.tokenizer.encode(p)) for p in prompts)
    cache_size = self.max_prompt_length + self.max_generation_steps + 100
    safe_gen_length = min(
        self.max_generation_steps,
        cache_size - max_length - 100,  # 100 token buffer
    )
    if safe_gen_length < 256:
      print(
          f"WARNING: Short generation length ({safe_gen_length} tokens) due to"
          f" long prompt ({max_length} tokens)"
      )

    stop_token_ids = self.eos_token_ids or self._resolve_eos_token_ids()

    # Generate
    if self.sampler_type == "vanilla":
      out_data = self.sampler_vanilla(
          input_strings=prompts,
          max_generation_steps=safe_gen_length,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p,
          echo=False,
          eos_tokens=stop_token_ids,
          seed=jax.random.PRNGKey(seed) if seed is not None else None,
      )
    elif self.sampler_type == "sglang-jax":
      out_data = self.sampler_sglang(
          input_strings=prompts,
          max_generation_steps=safe_gen_length,
          max_prompt_length=self.max_prompt_length,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          seed=seed,
          echo=False,
          pad_output=True,
      )
    else:
      raise ValueError(f"Unsupported sampler type: {self.sampler_type}")
    return out_data.text

  def evaluate(
      self,
      batch_size: int = 8,
      num_batches: int | None = None,
      temperature: float = 0.6,
      top_k: Optional[int] = 50,
      top_p: Optional[float] = 0.95,
      num_passes: int = 1,
      debug_first_n: int = 3,  # NEW: Debug first N examples
  ) -> Dict[str, Any]:
    print("=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches: {num_batches or 'all'}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print(f"  Passes per question: {num_passes}")
    print(f"  Debug first N examples: {debug_first_n}")
    print("=" * 60)

    # Load dataset
    dataset = self.load_dataset()

    # Create batched dataset
    if num_batches is not None:
      dataset = dataset.batch(batch_size)[:num_batches]
    else:
      dataset = dataset.batch(batch_size)

    correct = 0
    total = 0
    results = []
    debug_count = 0

    # Evaluate batch by batch
    for batch_idx, batch in enumerate(tqdm(dataset, desc="Evaluating")):
      prompts = batch["prompt"]

      questions = batch["question"]
      answers = batch["answer"]

      responses_collection = [[] for _ in range(len(prompts))]
      for pass_idx in range(num_passes):
        batch_response = self.generate(
            prompts=prompts,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=pass_idx,
        )
        for i, r in enumerate(batch_response):
          responses_collection[i].append(r)

      for prompt, question, answer, responses in zip(
          prompts, questions, answers, responses_collection
      ):
        is_correct = False
        extracted_answers = []
        answer_correct = []
        for response in responses:
          # Grade answer using both methods from utils.py
          if "aime_2024" in self.dataset:
            is_correct = evaluate_correctness(response, answer)
          else:
            model_answer = extract_answer_robust(response)
            extracted_answers.append(model_answer)

            if model_answer is None:
              continue
            # Grade answer using both methods from utils.py
            is_correct = math_utils.grade_answer_mathd(
                model_answer, answer
            ) or math_utils.grade_answer_sympy(model_answer, answer)

          answer_correct.append(is_correct)

          if is_correct:
            break

        if is_correct:
          correct += 1

        should_debug = debug_count < debug_first_n

        if should_debug:
          print(f"\n{'='*60}")
          print(f"DEBUG Example {debug_count + 1}/{debug_first_n}")
          print(f"Question: {question[:]}")
          print("=" * 60 + "\n")
          print(f"Ground truth: {answer}")
          print("=" * 60 + "\n")
          print(f"Prompt (first 300 chars): {prompt[:]}")
          if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            print(f"Prompt length: {len(self.tokenizer.encode(prompt))} tokens")
          print("=" * 60 + "\n")
          for i, (response, ans, cor) in enumerate(
              zip(responses, extracted_answers, answer_correct)
          ):
            print(f"Response {i}: {response}")
            print("=" * 120 + "\n")
            print(f"\nExtracted answer{i}: {ans}")
            print(f"Is correct: {cor}")
          print(f"Final result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
          print(
              f"Running accuracy: {correct}/{total+1} ="
              f" {(correct/(total+1)*100):.2f}%"
          )
          debug_count += 1

        total += 1

        # Store result
        results.append({
            "question": question,
            "answer": answer,
            "responses": responses,
            "extracted_answers": extracted_answers,
            "correct": is_correct,
        })

        # Print progress
        if total % 10 == 0:
          current_acc = (correct / total * 100) if total > 0 else 0
          print(f"\nProgress: {correct}/{total} = {current_acc:.2f}%")

    # Calculate final metrics
    accuracy = (correct / total * 100) if total > 0 else 0

    eval_results = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_passes": num_passes,
        "detailed_results": results,
    }

    return eval_results
# %%

if NOTEBOOK_ENV == "g3":
  DATA_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev"
  MODEL_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev"
else:
  DATA_PATH_PREFIX = "gs://tunix/data"
  MODEL_PATH_PREFIX = "gs://tunix/models"

MATH_500_DATA_PATH = os.path.join(DATA_PATH_PREFIX, "MATH-500/test.jsonl")
AIME_2024_DATA_PATH = os.path.join(
    DATA_PATH_PREFIX, "HuggingFaceH4/aime_2024/train-00000-of-00001.parquet"
)
MODEL_MAPPING = {
    "Qwen/Qwen2.5-1.5B-Instruct": (
        qwen2_lib.ModelConfig.qwen2p5_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "qwen2_5/torch/1.5b-it"),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (
        qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B"),
    ),
    "agentica-org/DeepScaleR-1.5B-Preview": (
        qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "DeepScaleR-1.5B-Preview"),
    ),
}
DATASET_MAPPING = {
    "math500": MATH_500_DATA_PATH,
    "aime2024": AIME_2024_DATA_PATH,
}
DEFAULT_MODEL_VERSION = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET = "math500"


def _resolve_mesh_config(mesh_fsdp: int | None, mesh_tp: int | None):
  device_count = max(1, jax.device_count())
  if mesh_fsdp is None and mesh_tp is None:
    return [[1, device_count], ["fsdp", "tp"]]
  if mesh_fsdp is None or mesh_tp is None:
    raise ValueError("`--mesh-fsdp` and `--mesh-tp` must be set together.")
  if mesh_fsdp * mesh_tp != device_count:
    raise ValueError(
        f"mesh size mismatch: {mesh_fsdp}x{mesh_tp}={mesh_fsdp * mesh_tp}, "
        f"but jax.device_count()={device_count}."
    )
  return [[mesh_fsdp, mesh_tp], ["fsdp", "tp"]]


def _resolve_paths(args):
  model_config, default_model_path = MODEL_MAPPING[args.model_version]
  model_path = args.model_path if args.model_path else default_model_path
  dataset_path = args.dataset_path if args.dataset_path else DATASET_MAPPING[args.dataset]
  tokenizer_source = args.tokenizer_source
  if tokenizer_source is None:
    if not _is_remote(model_path):
      tokenizer_source = model_path
    elif NOTEBOOK_ENV == "g3":
      tokenizer_source = model_path
    else:
      tokenizer_source = args.model_version
  return model_config, model_path, dataset_path, tokenizer_source


def _dataset_profile(dataset_name: str, dataset_path: str):
  path = (dataset_path or "").lower()
  if dataset_name == "aime2024" or "aime_2024" in path:
    return {
        "max_prompt_length": 2048,
        "max_generation_steps": 32768,
        "batch_size": 1,
        "temperature": 0.6,
        "top_k": -1,
        "top_p": 0.95,
        "debug_first_n": 3,
    }
  return {
      "max_prompt_length": 1024,
      "max_generation_steps": 1024,
      "batch_size": 8,
      "temperature": 0.6,
      "top_k": 50,
      "top_p": 0.95,
      "debug_first_n": 3,
  }


def _apply_dataset_defaults(args, dataset_path: str):
  profile = _dataset_profile(args.dataset, dataset_path)
  if args.max_prompt_length is None:
    args.max_prompt_length = profile["max_prompt_length"]
  if args.max_generation_steps is None:
    args.max_generation_steps = profile["max_generation_steps"]
  if args.batch_size is None:
    args.batch_size = profile["batch_size"]
  if args.temperature is None:
    args.temperature = profile["temperature"]
  if args.top_k is None:
    args.top_k = profile["top_k"]
  if args.top_p is None:
    args.top_p = profile["top_p"]
  if args.debug_first_n is None:
    args.debug_first_n = profile["debug_first_n"]


def _run_preflight(args, model_path: str, dataset_path: str, tokenizer_source: str):
  errors = []
  if not _is_remote(model_path) and not os.path.exists(model_path):
    errors.append(f"Missing local model path: {model_path}")
  if not _is_remote(dataset_path) and not os.path.exists(dataset_path):
    errors.append(f"Missing local dataset path: {dataset_path}")
  if any(_is_gcs(p) for p in (model_path, dataset_path)) and not _check_gcp_auth():
    errors.append(
        "GCS path detected but no Application Default Credentials found. "
        "Run `gcloud auth application-default login` "
        "or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`."
    )
  if _is_hf_repo_id(tokenizer_source) and not args.hf_token:
    print(
        "INFO: Hugging Face repo id is used without token. "
        "Public models may still work; gated/private models require `HF_TOKEN`."
    )
  if args.require_hf_token and not args.hf_token:
    errors.append(
      "`--require-hf-token` was set but no token was provided. "
      "Use `--hf-token` or export `HF_TOKEN`."
    )
  if errors:
    raise RuntimeError("\n".join(errors))


def parse_args(argv: Sequence[str] | None = None):
  parser_ = argparse.ArgumentParser(description="Evaluate DeepScaler/Qwen math models.")
  parser_.add_argument("--model-version", default=DEFAULT_MODEL_VERSION, choices=list(MODEL_MAPPING.keys()))
  parser_.add_argument("--model-path", default=None)
  parser_.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASET_MAPPING.keys()))
  parser_.add_argument("--dataset-path", default=None)
  parser_.add_argument("--tokenizer-source", default=None)
  parser_.add_argument("--sampler-type", default="vanilla", choices=["vanilla", "sglang-jax"])
  parser_.add_argument("--max-prompt-length", type=int, default=None)
  parser_.add_argument("--max-generation-steps", type=int, default=None)
  parser_.add_argument("--batch-size", type=int, default=None)
  parser_.add_argument("--num-batches", type=int, default=None)
  parser_.add_argument("--temperature", type=float, default=None)
  parser_.add_argument("--top-k", type=int, default=None, help="Set <0 to disable top-k.")
  parser_.add_argument("--top-p", type=float, default=None)
  parser_.add_argument("--num-passes", type=int, default=1)
  parser_.add_argument("--debug-first-n", type=int, default=None)
  parser_.add_argument("--mesh-fsdp", type=int, default=None)
  parser_.add_argument("--mesh-tp", type=int, default=None)
  parser_.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
  parser_.add_argument("--require-hf-token", action="store_true")
  parser_.add_argument("--skip-preflight", action="store_true")
  parser_.add_argument("--smoke-test", action="store_true")
  return parser_.parse_args(argv)


def _apply_smoke_test(args):
  if not args.smoke_test:
    return
  args.batch_size = 1
  args.num_batches = 1
  args.debug_first_n = 1
  args.max_generation_steps = min(args.max_generation_steps, 256)
  args.max_prompt_length = min(args.max_prompt_length, 512)


def run_eval(args):
  model_config, model_path, dataset_path, tokenizer_source = _resolve_paths(args)
  _apply_dataset_defaults(args, dataset_path=dataset_path)
  _apply_smoke_test(args)
  if not args.skip_preflight:
    _run_preflight(args, model_path=model_path, dataset_path=dataset_path, tokenizer_source=tokenizer_source)
  mesh_config = _resolve_mesh_config(args.mesh_fsdp, args.mesh_tp)

  print(f"NOTEBOOK_ENV: {NOTEBOOK_ENV}")
  print(f"model version: {args.model_version}")
  print(f"model path: {model_path}")
  print(f"dataset path: {dataset_path}")
  print(f"mesh config: {mesh_config}")
  print(f"smoke test: {args.smoke_test}")

  evaluator = Qwen25MathEvaluator(
      model_config=model_config,
      model_version=args.model_version,
      model_path=model_path,
      dataset=dataset_path,
      mesh_config=mesh_config,
      max_prompt_length=args.max_prompt_length,
      max_generation_steps=args.max_generation_steps,
      sampler_type=args.sampler_type,
      tokenizer_source=tokenizer_source,
      hf_token=args.hf_token,
  )
  evaluator.load_model()
  print("\nStarting evaluation...")
  results = evaluator.evaluate(
      batch_size=args.batch_size,
      num_batches=args.num_batches,
      temperature=args.temperature,
      top_k=None if args.top_k is not None and args.top_k < 0 else args.top_k,
      top_p=args.top_p,
      num_passes=args.num_passes,
      debug_first_n=args.debug_first_n,
  )

  print("\n" + "=" * 60)
  print("Evaluation Results")
  print("=" * 60)
  print(f"Model: {model_path}")
  print(f"Dataset: {dataset_path}")
  print(f"Correct: {results['correct']}/{results['total']}")
  print(f"Accuracy: {results['accuracy']:.2f}%")
  print("=" * 60)


def main(argv: Sequence[str] | None = None):
  args = parse_args(argv)
  run_eval(args)


if __name__ == "__main__":
  main()
