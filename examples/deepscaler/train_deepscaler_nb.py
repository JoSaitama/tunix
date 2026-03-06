# %%

# [WIP] Reproduction of [Deepscaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) with Single-turn Agentic framework.

import argparse
import contextlib
import importlib.util
import os
import subprocess
from typing import Sequence

from flax import nnx
import grain
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
import pandas as pd
import qwix

import datasets as datasets_lib
import transformers

try:
  from etils import ecolab
  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload="tunix",
      behavior="preferred",
      cell_autoreload=True,
  )
except Exception:
  cm = contextlib.nullcontext()

with cm:
  from tunix.models.qwen2 import params as params_lib
  from tunix.models.qwen2 import model as model_lib
  from tunix.sft import metrics_logger
  from tunix.rl.agentic.agents import model_agent
  from tunix.rl.agentic.environments import task_environment
  from tunix.rl.agentic.rewards import reward
  from tunix.rl.agentic.trajectory import trajectory_collect_engine
  from tunix.rl.agentic.parser.chat_template_parser import parser
  from tunix.rl.experimental.agentic_grpo_learner import GRPOConfig, GRPOLearner
  from tunix.rl import rl_cluster as rl_cluster_lib
  from tunix.rl.rollout import base_rollout
  from tunix.sft import utils as sft_utils
  from tunix.utils import math_rewards
  from tunix.utils import compat

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer

# ====== Data ======
TRAIN_FRACTION = 1.0

# ====== Reproducibility ======
SEED = 42

# ====== LoRA ======
RANK = 64
ALPHA = 64.0
TRAIN_WITH_LORA = False
TRAIN_DTYPE = "fp32"  # one of "fp32" or "bf16"
REWARD_ADVANTAGE_DTYPE = "fp32"  # one of "fp32" or "bf16"

# ====== Sharding ======
MESH = [(2, 4), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 2048
TOTAL_GENERATION_STEPS = 8192
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
NUM_GENERATIONS = 2

# === other GRPO configs ===
NUM_ITERATIONS = 1
BETA = 0.001
EPSILON = 0.2
FAST_PATH_DEFAULT_ROLLOUT_PROMPT_BATCH_SIZE = 4

# ====== Training ======
BATCH_SIZE = 32
MINI_BATCH_SIZE = 32
NUM_BATCHES = 100
NUM_TEST_BATCHES = 50
EVAL_EVERY_N_STEPS = 1000
NUM_EPOCHS = 100
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 1e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = int(0.1 * MAX_STEPS)
MAX_GRAD_NORM = 0.1
OFFLOAD_TO_CPU = False

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Rollout ======
ROLLOUT_ENGINE = "vanilla"  # one of "vanilla", "vllm" or "sglang-jax"
ROLLOUT_VLLM_HBM_UTILIZATION = 0.2
ROLLOUT_VLLM_TPU_BACKEND_TYPE = "jax"
ROLLOUT_VLLM_SERVER_MODE = False
ROLLOUT_VLLM_ASYNC_SCHEDULING = False
ROLLOUT_VLLM_SWAP_SPACE_SIZE_GB = 4.0
ROLLOUT_DP = -1
ROLLOUT_TP = -1
ROLLOUT_SGLANG_JAX_CONTEXT_LENGTH = None
ROLLOUT_SGLANG_JAX_MEM_FRACTION_STATIC = 0.2
ROLLOUT_SGLANG_JAX_DISABLE_RADIX_CACHE = True
ROLLOUT_SGLANG_JAX_ENABLE_DETERMINISTIC_SAMPLING = False
ROLLOUT_SGLANG_JAX_CHUNKED_PREFILL_SIZE = -1
ROLLOUT_SGLANG_JAX_PAGE_SIZE = 64
ROLLOUT_SGLANG_JAX_DTYPE = "auto"
ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE = "auto"

REMOTE_PREFIXES = ("gs://", "gcs://", "s3://", "http://", "https://", "hf://")


def _detect_env_and_file_open():
  try:
    from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
    return "g3", gfile.Open
  except Exception:
    import fsspec
    return "git", fsspec.open


NOTEBOOK_ENV, file_open = _detect_env_and_file_open()

if NOTEBOOK_ENV == "g3":
  DATA_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/rl/data/"
  MODEL_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/"
  CKPT_DIR_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev/"
else:
  DATA_PATH_PREFIX = "gs://tunix/rl/data"
  MODEL_PATH_PREFIX = "gs://tunix/rl/models"
  CKPT_DIR_PREFIX = "gs://tunix/rl/checkpoints"

CKPT_DIR = os.path.join(CKPT_DIR_PREFIX, "deepscaler_ckpt/01")
MODEL_VERSION = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = os.path.join(MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B")
DEEPSCALER_DATA_PATH = os.path.join(
    DATA_PATH_PREFIX, "DeepScaleR-Preview-Dataset/deepscaler.json"
)
AIME_2024_DATA_PATH = os.path.join(
    DATA_PATH_PREFIX, "HuggingFaceH4/aime_2024/train-00000-of-00001.parquet"
)

show_hbm_usage = sft_utils.show_hbm_usage
ModelAgent = model_agent.ModelAgent
TaskEnvironment = task_environment.TaskEnvironment
TrajectoryCollectEngine = trajectory_collect_engine.TrajectoryCollectEngine
is_two_reward = reward.is_two_reward


def _is_remote(path: str) -> bool:
  return path.startswith(REMOTE_PREFIXES)


def _is_gcs(path: str) -> bool:
  return path.startswith("gs://") or path.startswith("gcs://")


def _is_hf_repo_id(path_or_id: str) -> bool:
  if os.path.isabs(path_or_id) or path_or_id.startswith("."):
    return False
  return ("/" in path_or_id) and (not _is_remote(path_or_id)) and (not os.path.exists(path_or_id))


def _load_tokenizer(source: str, hf_token: str | None):
  if hf_token:
    try:
      return AutoTokenizer.from_pretrained(source, token=hf_token)
    except TypeError:
      return AutoTokenizer.from_pretrained(source, use_auth_token=hf_token)
  return AutoTokenizer.from_pretrained(source)


def _resolve_mesh(mesh_fsdp: int | None, mesh_tp: int | None):
  device_count = max(1, jax.device_count())
  if mesh_fsdp is None and mesh_tp is None:
    return (1, device_count)
  if mesh_fsdp is None or mesh_tp is None:
    raise ValueError("`--mesh-fsdp` and `--mesh-tp` must be set together.")
  mesh_size = mesh_fsdp * mesh_tp
  if mesh_size != device_count:
    raise ValueError(
        f"mesh size mismatch: {mesh_fsdp}x{mesh_tp}={mesh_size}, "
        f"but jax.device_count()={device_count}. Use values whose product equals device count."
    )
  return (mesh_fsdp, mesh_tp)


def _build_training_mesh(mesh_shape: tuple[int, int]) -> jax.sharding.Mesh:
  return jax.make_mesh(
      mesh_shape,
      ("fsdp", "tp"),
      axis_types=(jax.sharding.AxisType.Auto,) * 2,
  )


def _build_rollout_mesh(
    rollout_engine: str,
    training_mesh: jax.sharding.Mesh,
    rollout_tp_override: int | None = None,
) -> jax.sharding.Mesh:
  # sglang-jax expects rollout computations under a data/tensor mesh context.
  # Keep actor/reference mesh unchanged so the vanilla training path is intact.
  if rollout_engine != "sglang_jax":
    return training_mesh

  ordered_devices = sorted(training_mesh.devices.flatten().tolist(), key=lambda d: d.id)
  if rollout_tp_override is not None and rollout_tp_override > 0:
    if rollout_tp_override > len(ordered_devices):
      raise ValueError(
          f"`--rollout-tp`={rollout_tp_override} exceeds available devices "
          f"({len(ordered_devices)}) for sglang-jax rollout."
      )
    ordered_devices = ordered_devices[:rollout_tp_override]
  try:
    # Use JAX's mesh builder to preserve TPU physical topology ordering.
    from jax._src import mesh_utils as jax_mesh_utils  # pylint: disable=g-import-not-at-top
    rollout_devices = jax_mesh_utils.create_device_mesh(
        (1, len(ordered_devices)),
        devices=ordered_devices,
        contiguous_submeshes=False,
        allow_split_physical_axes=True,
    )
  except Exception:
    rollout_devices = np.array(ordered_devices, dtype=object).reshape((1, len(ordered_devices)))
  return jax.sharding.Mesh(
      rollout_devices,
      ("data", "tensor"),
      axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
  )


def _normalize_rollout_engine(rollout_engine: str) -> str:
  normalized = rollout_engine.strip().lower().replace("-", "_")
  if normalized not in ("vanilla", "vllm", "sglang_jax"):
    raise ValueError(
        "`--rollout-engine` must be one of: vanilla, vllm, sglang_jax "
        f"(or alias sglang-jax). Got: {rollout_engine!r}"
    )
  return normalized


def _to_jax_train_dtype(dtype_name: str) -> jnp.dtype:
  if dtype_name == "fp32":
    return jnp.float32
  if dtype_name == "bf16":
    return jnp.bfloat16
  raise ValueError(
      "`--train-dtype` must be one of: fp32, bf16. "
      f"Got: {dtype_name!r}"
  )


def _normalize_sglang_dtype(dtype_name: str) -> str:
  normalized = dtype_name.strip().lower()
  if normalized == "fp32":
    return "float32"
  if normalized == "bf16":
    return "bfloat16"
  if normalized in ("auto", "half", "float16", "bfloat16", "float", "float32"):
    return normalized
  raise ValueError(
      "`--rollout-sglang-jax-dtype` must be one of: "
      "auto, float32, bfloat16, float16, half, float, fp32, bf16. "
      f"Got: {dtype_name!r}"
  )


def _normalize_sglang_kv_cache_dtype(dtype_name: str) -> str:
  normalized = dtype_name.strip().lower()
  if normalized in ("auto", "bf16", "fp8_e5m2", "fp8_e4m3"):
    return normalized
  raise ValueError(
      "`--rollout-sglang-jax-kv-cache-dtype` must be one of: "
      "auto, bf16, fp8_e5m2, fp8_e4m3. "
      f"Got: {dtype_name!r}"
  )


def _resolve_rollout_model_source(args) -> str:
  if args.rollout_model_source:
    return args.rollout_model_source
  return args.model_path


def _is_module_available(module_name: str) -> bool:
  return importlib.util.find_spec(module_name) is not None


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


def _ensure_parent(path: str):
  if _is_remote(path):
    return
  os.makedirs(path, exist_ok=True)


def _resolve_tokenizer_source(args) -> str:
  if args.tokenizer_source:
    return args.tokenizer_source
  if not _is_remote(args.model_path):
    return args.model_path
  if NOTEBOOK_ENV == "g3":
    return args.model_path
  return args.model_version


def _run_preflight(args):
  errors = []
  rollout_engine = _normalize_rollout_engine(args.rollout_engine)
  rollout_model_source = _resolve_rollout_model_source(args)

  for required_input in (args.train_dataset_path, args.test_dataset_path):
    if not _is_remote(required_input) and not os.path.exists(required_input):
      errors.append(f"Missing local input path: {required_input}")

  if not _is_remote(args.model_path) and not os.path.exists(args.model_path):
    errors.append(f"Missing local model path: {args.model_path}")

  if (
      rollout_engine in ("vllm", "sglang_jax")
      and not _is_remote(rollout_model_source)
      and not _is_hf_repo_id(rollout_model_source)
      and not os.path.exists(rollout_model_source)
  ):
    errors.append(
        f"Missing local rollout model path/source: {rollout_model_source}"
    )

  if any(
      _is_gcs(path)
      for path in (
          args.model_path,
          args.train_dataset_path,
          args.test_dataset_path,
          rollout_model_source,
      )
  ):
    if not _check_gcp_auth():
      errors.append(
          "GCS path detected but no Application Default Credentials found. "
          "Run `gcloud auth application-default login` "
          "or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`."
      )

  tokenizer_source = _resolve_tokenizer_source(args)
  if _is_hf_repo_id(tokenizer_source) and not args.hf_token:
    print(
        "INFO: Hugging Face repo id is used without token. "
        "Public models may still work; gated/private models require `HF_TOKEN`."
    )

  if (
      rollout_engine in ("vllm", "sglang_jax")
      and _is_hf_repo_id(rollout_model_source)
      and not args.hf_token
  ):
    print(
        "INFO: rollout model source is a Hugging Face repo id without token. "
        "Public models may still work; gated/private models require `HF_TOKEN`."
    )

  if rollout_engine == "vllm" and not _is_module_available("vllm"):
    errors.append(
        "Rollout engine `vllm` was selected, but `vllm` is not installed. "
        "Install vLLM/tpu-inference in this runtime or use `--rollout-engine vanilla`."
    )
  if rollout_engine == "vllm":
    if args.rollout_dp == 0 or args.rollout_dp < -1:
      errors.append(
          "`--rollout-dp` must be -1 (auto) or a positive integer."
      )
    if args.rollout_tp == 0 or args.rollout_tp < -1:
      errors.append(
          "`--rollout-tp` must be -1 (auto) or a positive integer."
      )
    if not (0.0 < args.rollout_vllm_hbm_utilization <= 1.0):
      errors.append(
          "`--rollout-vllm-hbm-utilization` must be in (0, 1]."
      )
    if args.rollout_vllm_swap_space_size_gb < 0:
      errors.append(
          "`--rollout-vllm-swap-space-size-gb` must be >= 0."
      )
  if rollout_engine == "sglang_jax" and not _is_module_available("sgl_jax"):
    errors.append(
        "Rollout engine `sglang_jax` was selected, but `sgl_jax` is not installed. "
        "Install sglang-jax in this runtime or use `--rollout-engine vanilla`."
    )
  if rollout_engine == "sglang_jax":
    try:
      _normalize_sglang_dtype(args.rollout_sglang_jax_dtype)
    except ValueError as exc:
      errors.append(str(exc))
    try:
      _normalize_sglang_kv_cache_dtype(args.rollout_sglang_jax_kv_cache_dtype)
    except ValueError as exc:
      errors.append(str(exc))
    if (
        args.rollout_sglang_jax_context_length is not None
        and args.rollout_sglang_jax_context_length <= 0
    ):
      errors.append(
          "`--rollout-sglang-jax-context-length` must be > 0 if set."
      )
    if not (0.0 < args.rollout_sglang_jax_mem_fraction_static <= 1.0):
      errors.append(
          "`--rollout-sglang-jax-mem-fraction-static` must be in (0, 1]."
      )
    if (
        args.rollout_sglang_jax_chunked_prefill_size == 0
        or args.rollout_sglang_jax_chunked_prefill_size < -1
    ):
      errors.append(
          "`--rollout-sglang-jax-chunked-prefill-size` must be -1 "
          "(disable) or a positive integer."
      )
    if args.rollout_sglang_jax_page_size <= 0:
      errors.append("`--rollout-sglang-jax-page-size` must be > 0.")

  _ensure_parent(args.checkpoint_dir)
  _ensure_parent(args.metrics_log_dir)

  if errors:
    raise RuntimeError("\n".join(errors))


def create_datasets(tokenizer, train_ds_path: str, test_ds_path: str):
  def preprocess_fn(example, index):
    del index
    return {
        "question": example["problem"],
        "ground_truth": example["answer"],
        "data_source": "math",
    }

  with file_open(train_ds_path) as train_f, file_open(test_ds_path, "rb") as test_f:
    train_df = pd.read_json(train_f)
    test_df = pd.read_parquet(test_f)

  train_ds = Dataset.from_pandas(train_df).map(preprocess_fn, with_indices=True)
  test_ds = Dataset.from_pandas(test_df).map(preprocess_fn, with_indices=True)

  def process_item(item):
    question = item["question"]
    answer = item["answer"]
    instruction = "Let's think step by step, and put your final answer within \\boxed{}."
    prompt = f"{question} {instruction}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompts": prompt,
        "question": question,
        "answer": answer,
    }

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  test_ds = grain.MapDataset.source(test_ds).map(process_item)
  return train_ds, test_ds


def get_lora_model(base_model, model_mesh, rank: int, alpha: float):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
          ".*attn_vec_einsum"
      ),
      rank=rank,
      alpha=alpha,
  )
  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)

  with compat.set_mesh(model_mesh):
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


def parse_args(argv: Sequence[str] | None = None):
  parser_ = argparse.ArgumentParser(description="Train DeepScaler with Tunix GRPO.")
  parser_.add_argument("--model-path", default=MODEL_PATH)
  parser_.add_argument("--model-version", default=MODEL_VERSION)
  parser_.add_argument("--tokenizer-source", default=None)
  parser_.add_argument(
      "--train-dtype",
      default=TRAIN_DTYPE,
      choices=["fp32", "bf16"],
      help="Train model dtype for actor/reference weights.",
  )
  parser_.add_argument(
      "--reward-advantage-dtype",
      default=REWARD_ADVANTAGE_DTYPE,
      choices=["fp32", "bf16"],
      help="Dtype for reward and advantage tensors.",
  )
  parser_.add_argument("--train-dataset-path", default=DEEPSCALER_DATA_PATH)
  parser_.add_argument("--test-dataset-path", default=AIME_2024_DATA_PATH)
  parser_.add_argument("--checkpoint-dir", default=CKPT_DIR)
  parser_.add_argument("--metrics-log-dir", default="/tmp/tensorboard/grpo")
  parser_.add_argument("--batch-size", type=int, default=BATCH_SIZE)
  parser_.add_argument("--mini-batch-size", type=int, default=MINI_BATCH_SIZE)
  parser_.add_argument("--train-micro-batch-size", type=int, default=1)
  parser_.add_argument("--num-batches", type=int, default=NUM_BATCHES)
  parser_.add_argument("--num-test-batches", type=int, default=NUM_TEST_BATCHES)
  parser_.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
  parser_.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
  parser_.add_argument("--max-steps", type=int, default=None)
  parser_.add_argument("--eval-every-n-steps", type=int, default=EVAL_EVERY_N_STEPS)
  parser_.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
  parser_.add_argument("--b1", type=float, default=B1)
  parser_.add_argument("--b2", type=float, default=B2)
  parser_.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
  parser_.add_argument("--warmup-steps", type=int, default=None)
  parser_.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
  parser_.add_argument(
      "--offload-to-cpu",
      action=argparse.BooleanOptionalAction,
      default=OFFLOAD_TO_CPU,
      help="Whether to offload actor/reference/rollout models to CPU between stages.",
  )
  parser_.add_argument("--mesh-fsdp", type=int, default=None)
  parser_.add_argument("--mesh-tp", type=int, default=None)
  parser_.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
  parser_.add_argument("--total-generation-steps", type=int, default=TOTAL_GENERATION_STEPS)
  parser_.add_argument("--temperature", type=float, default=TEMPERATURE)
  parser_.add_argument("--top-p", type=float, default=TOP_P)
  parser_.add_argument("--top-k", type=int, default=TOP_K)
  parser_.add_argument(
      "--rollout-engine",
      default=ROLLOUT_ENGINE,
      choices=["vanilla", "vllm", "sglang_jax", "sglang-jax"],
      help="Rollout backend. `sglang-jax` is accepted as an alias of `sglang_jax`.",
  )
  parser_.add_argument(
      "--rollout-model-source",
      default=None,
      help=(
          "Model path or repo id used by non-vanilla rollout backends. "
          "Defaults to --model-path."
      ),
  )
  parser_.add_argument(
      "--rollout-vllm-hbm-utilization",
      type=float,
      default=ROLLOUT_VLLM_HBM_UTILIZATION,
  )
  parser_.add_argument(
      "--rollout-vllm-tpu-backend-type",
      default=ROLLOUT_VLLM_TPU_BACKEND_TYPE,
      help='vLLM TPU backend type, e.g. "jax", "torchax", "pytorch_xla".',
  )
  parser_.add_argument(
      "--rollout-vllm-server-mode",
      action=argparse.BooleanOptionalAction,
      default=ROLLOUT_VLLM_SERVER_MODE,
  )
  parser_.add_argument(
      "--rollout-vllm-async-scheduling",
      action=argparse.BooleanOptionalAction,
      default=ROLLOUT_VLLM_ASYNC_SCHEDULING,
  )
  parser_.add_argument(
      "--rollout-vllm-swap-space-size-gb",
      type=float,
      default=ROLLOUT_VLLM_SWAP_SPACE_SIZE_GB,
  )
  parser_.add_argument("--rollout-dp", type=int, default=ROLLOUT_DP)
  parser_.add_argument("--rollout-tp", type=int, default=ROLLOUT_TP)
  parser_.add_argument(
      "--rollout-sglang-jax-context-length",
      type=int,
      default=ROLLOUT_SGLANG_JAX_CONTEXT_LENGTH,
      help="Context length for sglang-jax rollout. Defaults to rollout KV cache size.",
  )
  parser_.add_argument(
      "--rollout-sglang-jax-mem-fraction-static",
      type=float,
      default=ROLLOUT_SGLANG_JAX_MEM_FRACTION_STATIC,
  )
  parser_.add_argument(
      "--rollout-sglang-jax-disable-radix-cache",
      action=argparse.BooleanOptionalAction,
      default=ROLLOUT_SGLANG_JAX_DISABLE_RADIX_CACHE,
  )
  parser_.add_argument(
      "--rollout-sglang-jax-enable-deterministic-sampling",
      action=argparse.BooleanOptionalAction,
      default=ROLLOUT_SGLANG_JAX_ENABLE_DETERMINISTIC_SAMPLING,
  )
  parser_.add_argument(
      "--rollout-sglang-jax-chunked-prefill-size",
      type=int,
      default=ROLLOUT_SGLANG_JAX_CHUNKED_PREFILL_SIZE,
  )
  parser_.add_argument(
      "--rollout-sglang-jax-page-size",
      type=int,
      default=ROLLOUT_SGLANG_JAX_PAGE_SIZE,
  )
  parser_.add_argument(
      "--rollout-sglang-jax-dtype",
      default=ROLLOUT_SGLANG_JAX_DTYPE,
      choices=[
          "auto",
          "float32",
          "bfloat16",
          "float16",
          "half",
          "float",
          "fp32",
          "bf16",
      ],
      help="sglang-jax rollout model dtype.",
  )
  parser_.add_argument(
      "--rollout-sglang-jax-kv-cache-dtype",
      default=ROLLOUT_SGLANG_JAX_KV_CACHE_DTYPE,
      choices=["auto", "bf16", "fp8_e5m2", "fp8_e4m3"],
      help="sglang-jax KV cache dtype.",
  )
  parser_.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
  parser_.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
  parser_.add_argument(
      "--grpo-max-concurrency",
      type=int,
      default=None,
      help=(
          "Override GRPO rollout orchestration concurrency. "
          "For sglang_jax, default remains 1 when not set."
      ),
  )
  parser_.add_argument(
      "--enable-rollout-fast-path",
      action="store_true",
      help=(
          "Enable batched rollout fast-path for sglang_jax. "
          "This bypasses the orchestrator producer path for training rollout."
      ),
  )
  parser_.add_argument(
      "--rollout-prompt-batch-size",
      type=int,
      default=None,
      help=(
          "Prompt batch size per rollout generate call in fast-path mode. "
          "If unset and fast-path is enabled, defaults to "
          f"{FAST_PATH_DEFAULT_ROLLOUT_PROMPT_BATCH_SIZE}."
      ),
  )
  parser_.add_argument("--beta", type=float, default=BETA)
  parser_.add_argument("--epsilon", type=float, default=EPSILON)
  parser_.add_argument("--save-interval-steps", type=int, default=SAVE_INTERVAL_STEPS)
  parser_.add_argument("--max-to-keep", type=int, default=MAX_TO_KEEP)
  parser_.add_argument("--train-with-lora", action="store_true", default=TRAIN_WITH_LORA)
  parser_.add_argument("--lora-rank", type=int, default=RANK)
  parser_.add_argument("--lora-alpha", type=float, default=ALPHA)
  parser_.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
  parser_.add_argument("--require-hf-token", action="store_true")
  parser_.add_argument("--enable-wandb", action="store_true")
  parser_.add_argument("--skip-preflight", action="store_true")
  parser_.add_argument("--smoke-test", action="store_true")
  return parser_.parse_args(argv)


def _build_runtime_values(args):
  num_batches = args.num_batches
  num_test_batches = args.num_test_batches
  num_epochs = args.num_epochs
  eval_every_n_steps = args.eval_every_n_steps
  total_generation_steps = args.total_generation_steps
  max_prompt_length = args.max_prompt_length
  max_steps = args.max_steps

  if args.smoke_test:
    num_batches = 1
    num_test_batches = 1
    num_epochs = 1
    eval_every_n_steps = 1
    total_generation_steps = min(total_generation_steps, 256)
    max_prompt_length = min(max_prompt_length, 512)
    if max_steps is None:
      max_steps = 1

  if max_steps is None:
    max_steps = int(num_batches * args.num_iterations * args.train_fraction * num_epochs)
  max_steps = max(1, max_steps)
  default_warmup_steps = int(0.1 * max_steps)
  requested_warmup_steps = (
      args.warmup_steps if args.warmup_steps is not None else default_warmup_steps
  )
  warmup_steps = max(0, min(requested_warmup_steps, max_steps - 1))

  return {
    "num_batches": num_batches,
    "num_test_batches": num_test_batches,
    "num_epochs": num_epochs,
    "eval_every_n_steps": eval_every_n_steps,
    "total_generation_steps": total_generation_steps,
    "max_prompt_length": max_prompt_length,
    "max_steps": max_steps,
    "warmup_steps": warmup_steps,
  }


def _build_rollout_config(
    args,
    runtime: dict[str, int],
    tokenizer,
    rollout_engine: str,
) -> base_rollout.RolloutConfig:
  kv_cache_size = (
      runtime["max_prompt_length"] + runtime["total_generation_steps"] + 256
  )
  common_kwargs = dict(
      max_tokens_to_generate=runtime["total_generation_steps"],
      max_prompt_length=runtime["max_prompt_length"],
      kv_cache_size=kv_cache_size,
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=args.top_k,
      eos_tokens=[tokenizer.encode("<|im_end|>")[0]],
  )
  if rollout_engine == "vanilla":
    return base_rollout.RolloutConfig(**common_kwargs)

  rollout_model_source = _resolve_rollout_model_source(args)
  if rollout_engine == "vllm":
    return base_rollout.RolloutConfig(
        **common_kwargs,
        data_parallel_size=args.rollout_dp,
        tensor_parallel_size=args.rollout_tp,
        rollout_vllm_model_version=rollout_model_source,
        rollout_vllm_hbm_utilization=args.rollout_vllm_hbm_utilization,
        rollout_vllm_tpu_backend_type=args.rollout_vllm_tpu_backend_type,
        rollout_vllm_server_mode=args.rollout_vllm_server_mode,
        rollout_vllm_async_scheduling=args.rollout_vllm_async_scheduling,
        rollout_vllm_swap_space_size_gb=args.rollout_vllm_swap_space_size_gb,
    )

  if rollout_engine == "sglang_jax":
    context_length = (
        args.rollout_sglang_jax_context_length
        if args.rollout_sglang_jax_context_length is not None
        else kv_cache_size
    )
    return base_rollout.RolloutConfig(
        **common_kwargs,
        rollout_sglang_jax_model_version=rollout_model_source,
        rollout_sglang_jax_context_length=context_length,
        rollout_sglang_jax_mem_fraction_static=args.rollout_sglang_jax_mem_fraction_static,
        rollout_sglang_jax_disable_radix_cache=args.rollout_sglang_jax_disable_radix_cache,
        rollout_sglang_jax_enable_deterministic_sampling=args.rollout_sglang_jax_enable_deterministic_sampling,
        rollout_sglang_jax_chunked_prefill_size=args.rollout_sglang_jax_chunked_prefill_size,
        rollout_sglang_jax_page_size=args.rollout_sglang_jax_page_size,
        rollout_sglang_jax_dtype=_normalize_sglang_dtype(
            args.rollout_sglang_jax_dtype
        ),
        rollout_sglang_jax_kv_cache_dtype=_normalize_sglang_kv_cache_dtype(
            args.rollout_sglang_jax_kv_cache_dtype
        ),
    )

  raise ValueError(f"Unsupported rollout engine: {rollout_engine}")


def run_training(args):
  if not args.enable_wandb:
    os.environ.setdefault("WANDB_DISABLED", "true")

  if args.require_hf_token and not args.hf_token:
    raise RuntimeError(
        "`--require-hf-token` was set but no token was provided. "
        "Use `--hf-token` or export `HF_TOKEN`."
    )

  if not args.skip_preflight:
    _run_preflight(args)

  runtime = _build_runtime_values(args)
  rollout_engine = _normalize_rollout_engine(args.rollout_engine)
  rollout_model_source = _resolve_rollout_model_source(args)
  if args.rollout_prompt_batch_size is not None and args.rollout_prompt_batch_size <= 0:
    raise ValueError("`--rollout-prompt-batch-size` must be a positive integer.")
  if args.enable_rollout_fast_path and rollout_engine != "sglang_jax":
    raise ValueError(
        "`--enable-rollout-fast-path` is only supported when "
        "`--rollout-engine sglang_jax` is used."
    )
  if args.rollout_prompt_batch_size is not None and not args.enable_rollout_fast_path:
    print(
        "WARNING: `--rollout-prompt-batch-size` is ignored unless "
        "`--enable-rollout-fast-path` is set."
    )
  train_model_dtype = _to_jax_train_dtype(args.train_dtype)
  reward_advantage_dtype = _to_jax_train_dtype(args.reward_advantage_dtype)
  os.environ["TUNIX_REWARD_ADVANTAGE_DTYPE"] = args.reward_advantage_dtype
  mesh_shape = _resolve_mesh(args.mesh_fsdp, args.mesh_tp)
  training_mesh = _build_training_mesh(mesh_shape)
  sglang_rollout_tp = args.rollout_tp if rollout_engine == "sglang_jax" and args.rollout_tp > 0 else None
  rollout_mesh = _build_rollout_mesh(
      rollout_engine,
      training_mesh,
      rollout_tp_override=sglang_rollout_tp,
  )
  print(f"NOTEBOOK_ENV: {NOTEBOOK_ENV}")
  print(f"mesh shape: {mesh_shape}")
  print(f"checkpoint dir: {args.checkpoint_dir}")
  print(f"metrics log dir: {args.metrics_log_dir}")
  print(f"smoke test: {args.smoke_test}")
  print(f"rollout engine: {rollout_engine}")
  if rollout_engine != "vanilla":
    print(f"rollout model source: {rollout_model_source}")
    print(f"rollout mesh shape: {tuple(rollout_mesh.shape.values())}")
    print(f"rollout mesh axes: {tuple(rollout_mesh.shape.keys())}")

  tokenizer_source = _resolve_tokenizer_source(args)
  print(f"tokenizer source: {tokenizer_source}")
  tokenizer = _load_tokenizer(tokenizer_source, args.hf_token)
  chat_parser = parser.QwenChatTemplateParser(tokenizer)

  train_dataset, test_dataset = create_datasets(
      tokenizer=tokenizer,
      train_ds_path=args.train_dataset_path,
      test_ds_path=args.test_dataset_path,
  )

  train_dataset = train_dataset.batch(args.batch_size)[: runtime["num_batches"]]
  if args.train_fraction == 1.0:
    train_dataset = train_dataset.repeat(runtime["num_epochs"])
  else:
    train_dataset = train_dataset[: int(len(train_dataset) * args.train_fraction)]
    train_dataset = train_dataset.repeat(runtime["num_epochs"])
  test_dataset = test_dataset.batch(args.batch_size)[: runtime["num_test_batches"]]

  del test_dataset
  show_hbm_usage()

  config = model_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()
  print("model path:", args.model_path)
  print("train model dtype:", train_model_dtype)
  print("reward/advantage dtype:", reward_advantage_dtype)
  qwen2_ref = params_lib.create_model_from_safe_tensors(
      args.model_path, config, training_mesh, dtype=train_model_dtype
  )

  if args.train_with_lora:
    qwen2_actor = get_lora_model(
        qwen2_ref, training_mesh, rank=args.lora_rank, alpha=args.lora_alpha
    )
  else:
    qwen2_actor = params_lib.create_model_from_safe_tensors(
        args.model_path, config, training_mesh, dtype=train_model_dtype
    )
  show_hbm_usage()

  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=args.save_interval_steps,
      max_to_keep=args.max_to_keep,
  )
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=args.metrics_log_dir,
      flush_every_n_steps=20,
  )

  optimizer = optax.adamw(
      learning_rate=optax.schedules.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=args.learning_rate,
          warmup_steps=runtime["warmup_steps"],
          decay_steps=runtime["max_steps"],
          end_value=0.0,
      ),
      b1=args.b1,
      b2=args.b2,
      weight_decay=args.weight_decay,
  )
  if args.max_grad_norm is not None and args.max_grad_norm > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=args.max_grad_norm),
        optimizer,
    )

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: training_mesh,
          rl_cluster_lib.Role.REFERENCE: training_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      rollout_engine=rollout_engine,
      offload_to_cpu=args.offload_to_cpu,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=runtime["eval_every_n_steps"],
          max_steps=runtime["max_steps"],
          mini_batch_size=args.mini_batch_size,
          train_micro_batch_size=args.train_micro_batch_size,
          metrics_logging_options=metrics_logging_options,
          checkpoint_root_directory=args.checkpoint_dir,
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=_build_rollout_config(
          args=args,
          runtime=runtime,
          tokenizer=tokenizer,
          rollout_engine=rollout_engine,
      ),
  )

  if args.grpo_max_concurrency is not None and args.grpo_max_concurrency <= 0:
    raise ValueError("`--grpo-max-concurrency` must be a positive integer.")

  grpo_max_concurrency = 1 if rollout_engine == "sglang_jax" else 8
  if rollout_engine == "sglang_jax":
    if args.grpo_max_concurrency is not None:
      grpo_max_concurrency = args.grpo_max_concurrency
    print(
        "sglang_jax rollout sets agentic max_concurrency="
        f"{grpo_max_concurrency} to control concurrent model calls."
    )
    if args.enable_rollout_fast_path:
      resolved_rollout_prompt_batch_size = (
          args.rollout_prompt_batch_size
          if args.rollout_prompt_batch_size is not None
          else FAST_PATH_DEFAULT_ROLLOUT_PROMPT_BATCH_SIZE
      )
      print(
          "rollout fast-path enabled: "
          f"rollout_prompt_batch_size={resolved_rollout_prompt_batch_size}"
      )
      if args.grpo_max_concurrency is not None:
        print(
            "NOTE: `--grpo-max-concurrency` is ignored when "
            "`--enable-rollout-fast-path` is enabled."
        )

  grpo_config = GRPOConfig(
      num_generations=args.num_generations,
      num_iterations=args.num_iterations,
      beta=args.beta,
      epsilon=args.epsilon,
      system_prompt="",
      max_concurrency=grpo_max_concurrency,
      enable_rollout_fast_path=args.enable_rollout_fast_path,
      rollout_prompt_batch_size=args.rollout_prompt_batch_size,
  )

  with compat.set_mesh(training_mesh):
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=qwen2_actor,
        reference=qwen2_ref,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

  grpo_trainer = GRPOLearner(
      rl_cluster=rl_cluster,
      reward_fns=[math_rewards.math_reward],
      algo_config=grpo_config,
      chat_parser=chat_parser,
  )
  grpo_trainer.train(train_dataset)


def main(argv: Sequence[str] | None = None):
  args = parse_args(argv)
  run_training(args)


if __name__ == "__main__":
  main()
