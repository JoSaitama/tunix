#!/usr/bin/env python3
# Auto-converted from examples/grpo_gemma.ipynb for CLI debugging.

import subprocess
import sys


import importlib.util

if importlib.util.find_spec('tensorflow') is None:
  print("Installing required packages...")
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'dotenv'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ipywidgets'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tensorflow'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tensorflow_datasets'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tensorboardX'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'grain'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/jax-ml/jax'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/google/tunix'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/google/qwix'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-q', 'flax', '-y'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/google/flax'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'], check=True)
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy>2'], check=True)

import os

# Do not hardcode secrets in notebooks/scripts.
# Provide these via environment variables, Colab secrets, or a local .env file.
# Example:
#   export HF_TOKEN=...
#   export KAGGLE_USERNAME=...
#   export KAGGLE_KEY=...

import kagglehub

try:
  from google.colab import userdata
  USE_COLAB = True

  subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-q', 'wandb', '-y'], check=True)

  os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
  os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
  os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
except:
  USE_COLAB = False

  from dotenv import load_dotenv
  load_dotenv()
  print("Using env vars to login")

  import nest_asyncio
  nest_asyncio.apply()
  print("nest_asyncio applied")

  # Only using wandb on TPU VM because it has strange bugs on Colab
  subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'wandb'], check=True)
  import wandb
  # Check if WANDB_API_KEY is set before logging in
  if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
      wandb.login(key=os.environ["WANDB_API_KEY"])
  else:
      print("WANDB_API_KEY not found. Skipping wandb login.")

if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
  kagglehub.login()

if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
    hf_token = os.environ["HF_TOKEN"]
    subprocess.run(['hf', 'auth', 'login', '--token', hf_token], check=True)
else:
    print("HF_TOKEN not found. Skipping Hugging Face login.")
import functools
from pprint import pprint
import re
import sys

import csv
import json
import shutil

from flax import nnx
import grain
import humanize
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
import kagglehub
import numpy as np
import optax
from orbax import checkpoint as ocp
from pathlib import Path
import qwix
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = .9

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
# Adjust mesh based on your TPU memory and model size.
NUM_TPUS = len(jax.devices())
# if NUM_TPUS == 8:
#   MESH_COUNTS = (1, 4)
# elif NUM_TPUS == 1:
#   MESH_COUNTS = (1, 1)
# else:
#   raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

if NUM_TPUS == 8:
    # v4-8 / v6e-8
    MESH_COUNTS = (1, 8)
elif NUM_TPUS == 4:
    # v4-4 / v6e-4 (current setup)
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

MESH = [
    MESH_COUNTS,
    ("fsdp", "tp"),
]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (mu in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (beta) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.08
# Epsilon value for clipping (epsilon in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 3738
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 64

EVAL_EVERY_N_STEPS = 64  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": None, "top_k": 1, "top_p": None},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}
def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. First, think about the problem \
and provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
"""
def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def _load_from_tfds(data_dir: str, split: str):
  import tensorflow_datasets.text.gsm8k
  return tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )


def download_kaggle_dataset(target_dir="./data/gsm8k"):
  os.makedirs(target_dir, exist_ok=True)
  src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
  src = Path(src)
  dst = Path(target_dir)

  for csv_file in src.glob("*.csv"):  # match all CSV files
    shutil.copy2(csv_file, dst / csv_file.name)
    print(f"Copied {csv_file.name} -> {dst/csv_file.name}")
  return target_dir


def get_dataset(data_dir, split="train", source="tfds") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if source == "tfds":
    import tensorflow_datasets.text.gsm8k
    data = tfds.data_source(
        "gsm8k",
        split=split,
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )

  elif source == "kaggle":
    kaggle_dir = download_kaggle_dataset(data_dir)
    file_name = "main_" + split + ".csv"
    csv_path = os.path.join(kaggle_dir, file_name)  # adjust filename if needed

    data = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        data.append({
            "question": row["question"],
            "answer": row["answer"],
        })

  else:
    raise ValueError(f"Unknown source: {source}")

  def _as_text(v):
    return v if isinstance(v, str) else v.decode("utf-8")

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=42)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT,
                  question=_as_text(x["question"]),
              ),
              # passed to reward functions
              "question": _as_text(x["question"]),
              # passed to reward functions
              "answer": extract_hash_answer(_as_text(x["answer"])),
          }
      )
  )
  return dataset
source = input("Choose data source [tfds/kaggle]: ").strip().lower()  # SHOULD I LEAVE THIS OR KEEP IT SET TO KAGGLE

if source not in ("tfds", "kaggle"):
  print("Invalid choice. Defaulting to 'tfds'.")
  source = "tfds"

print(f"Using data source: {source}")

dataset = get_dataset(TRAIN_DATA_DIR, "train", source).batch(TRAIN_MICRO_BATCH_SIZE)[
    :NUM_BATCHES
]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test", source).batch(TRAIN_MICRO_BATCH_SIZE)[
    :NUM_TEST_BATCHES
]

dataset_lengths = (
    len(train_dataset),
    len(val_dataset) if val_dataset is not None else 0,
    len(test_dataset),
)
print(f"dataset contains {dataset_lengths} of batches")
for ele in train_dataset[:1]:
  pprint(ele)
ignore_patterns = [
    "*.pth",  # Ignore PyTorch .pth weight files
]
print(f"Downloading {MODEL_ID} from Hugging Face...")
local_model_path = snapshot_download(
    repo_id=MODEL_ID, ignore_patterns=ignore_patterns
)
print(f"Model successfully downloaded to: {local_model_path}")

EOS_TOKENS = []
generation_config_path = os.path.join(local_model_path, "generation_config.json")
if os.path.exists(generation_config_path):
  with open(generation_config_path, "r") as f:
    generation_configs = json.load(f)
  EOS_TOKENS = generation_configs.get("eos_token_id", [])
  print(f"Using EOS token IDs: {EOS_TOKENS}")
MODEL_CP_PATH = local_model_path

# Use a config that matches MODEL_ID (it vs pt)
if "gemma-3-270m" in MODEL_ID:
    model_config = (
        gemma_lib.ModelConfig.gemma3_270m_it()
        if MODEL_ID.endswith("-it")
        else gemma_lib.ModelConfig.gemma3_270m()
    )
elif "gemma-3-1b" in MODEL_ID:
    model_config = (
        gemma_lib.ModelConfig.gemma3_1b_it()
        if MODEL_ID.endswith("-it")
        else gemma_lib.ModelConfig.gemma3_1b_pt()
    )
else:
    raise ValueError(f"Unknown model id: {MODEL_ID}")

mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
with mesh:
    gemma3 = params_safetensors_lib.create_model_from_safe_tensors(
        MODEL_CP_PATH, model_config, mesh
    )
    nnx.display(gemma3)

def get_lora_model(base_model, mesh):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
          ".*attn_vec_einsum"
      ),
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model
# Policy model
lora_policy = get_lora_model(gemma3, mesh=mesh)
nnx.display(lora_policy)
tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
if tokenizer.eos_id() not in EOS_TOKENS:
  EOS_TOKENS.append(tokenizer.eos_id())
  print(f"Using EOS token IDs: {EOS_TOKENS}")
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

example = f"{reasoning_start}Let me think!{reasoning_end}{solution_start}2{solution_end}"
print("Example:", example)
print("Match:\t", match_format.search(example))
def match_format_exactly(prompts, completions, **kwargs):
  return [
      0 if match_format.search(response) is None else 3.0
      for response in completions
  ]
def match_format_approximately(prompts, completions, **kwargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.find(reasoning_start) == 0 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores
def check_answer(prompts, completions, answer, **kwargs):
  responses = completions

  extracted_responses = [
      guess.group(1) if r is not None and (guess := match_format.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  assert len(extracted_responses) == len(
      answer
  ), f"{extracted_responses} and {answer} have mismatching length"
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")
def check_numbers(prompts, completions, answer, **kwargs):
  question = kwargs["question"]
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  print("START ============================")
  print(f"Question:\t{question[0]}")
  print(f"Answer:\t{answer[0]}")
  print(f"Response:\t{responses[0]}")
  print(f"Extracted:\t{extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue
  return scores
def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""

  if isinstance(question, str):
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
    ]
  else:
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=q,
        )
        for q in question
    ]

  out_data = sampler(
      input_strings=input_batch,
      max_generation_steps=768,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed if seed is not None else None,
      eos_tokens=EOS_TOKENS,
  )

  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output
def evaluate(
    dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""

  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]

    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)
        print(f"Question:\t{questions[idx]}")
        print(f"Correct Answer:\t{answers[idx]}")
        print(f"Response:\t{response}")
        print("-" * 50)

    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1

          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except:
          print("SKIPPED")

        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1

        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break

      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1

      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return
# The evaluation might take up to couple of minutes to finish.

sampler = sampler_lib.Sampler(
    transformer=lora_policy,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{num_correct=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)
# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/content/tmp/tensorboard/grpo", flush_every_n_steps=20
)
# Logs
# NOTE: Jupyter magic %load_ext tensorboard (ignored in script)
# NOTE: Jupyter magic %tensorboard --logdir /tmp/content/tmp/tensorboard/grpo --port=0 (run tensorboard manually if needed)

# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )
TRAIN_MICRO_BATCH_SIZE = int(TRAIN_MICRO_BATCH_SIZE)
# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_tokens=EOS_TOKENS,
    ),
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=gemma3,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# GRPO Trainer
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    algo_config=grpo_config,
)
with mesh:
  grpo_trainer.train(train_dataset, val_dataset)
if not USE_COLAB: wandb.init()
# Load checkpoint first.

trained_ckpt_path = os.path.join(
    CKPT_DIR, "actor", str(MAX_STEPS), "model_params"
)

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_policy, nnx.LoRAParam),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    lora_policy,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_policy, nnx.LoRAParam),
        trained_lora_params,
    ),
)
# The evaluation might take up to couple of minutes to finish. Please be patient.

sampler = sampler_lib.Sampler(
    transformer=lora_policy,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{num_correct=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)
output_dir = f"./{MODEL_ID}-lora"
if USE_COLAB:
    output_dir = f"/tmp/content/{MODEL_ID}-lora"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

print(f"Saving merged LoRA model to {output_dir}")

# Use the save_lora_merged_model_as_safetensors function
gemma_params.save_lora_merged_model_as_safetensors(
    local_model_path=local_model_path,
    output_dir=output_dir,
    lora_model=lora_policy,
    rank=RANK,
    alpha=ALPHA,
)

print("\n" + "="*60)
print("Model saved successfully!")
print(f"Output directory: {output_dir}")
print("="*60)

print("\nSaved files:")
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
    print(f"  {f:<30} {size:>10.2f} MB")

# For Colab: Download as zip
if USE_COLAB:
    from google.colab import files
    shutil.make_archive(output_dir, 'zip', output_dir)
    files.download(f"{output_dir}.zip")
