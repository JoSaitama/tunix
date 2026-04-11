# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for Agentic RL Learners."""

from __future__ import annotations

import abc
import asyncio
from concurrent import futures
import contextlib
import dataclasses
import itertools
import os
import threading
import time
from typing import Any, Callable, Coroutine, Dict, Generic, Iterable, Iterator, List, Sequence, TypeVar

from absl import logging
import flax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import task_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.queue import data_queue as queue_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import utils as sft_utils

TrainingInputT = Dict[str, List[str] | ArrayLike]
RewardFn = Callable[..., List[float]]
MetricFn = Callable[..., rl_cluster_lib.MetricsT]
_DEFAULT_FAST_PATH_ROLLOUT_PROMPT_BATCH_SIZE = 4
_LOG_TRAJECTORY_DETAILS = os.environ.get(
    "TUNIX_LOG_TRAJECTORY_DETAILS", ""
).lower() in ("1", "true", "yes", "on")


def _phase_timing_enabled() -> bool:
  return os.environ.get("TUNIX_ENABLE_PHASE_TIMING", "").lower() in (
      "1",
      "true",
      "yes",
      "on",
  )


def _sglang_jax_safe_actor_chunking_enabled() -> bool:
  if os.environ.get(
      "TUNIX_DISABLE_SGLANG_JAX_SAFE_ACTOR_CHUNKING", ""
  ).lower() in ("1", "true", "yes", "on"):
    return False
  return os.environ.get(
      "TUNIX_ENABLE_SGLANG_JAX_SAFE_ACTOR_CHUNKING", ""
  ).lower() in ("1", "true", "yes", "on")


def _sglang_jax_rollout_release_enabled() -> bool:
  return os.environ.get(
      "TUNIX_ENABLE_SGLANG_JAX_ROLLOUT_RELEASE", ""
  ).lower() in ("1", "true", "yes", "on")


def _deepscaler_adaptive_actor_chunking_enabled() -> bool:
  return os.environ.get(
      "TUNIX_ENABLE_DEEPSCALER_ADAPTIVE_ACTOR_CHUNKING", ""
  ).lower() in ("1", "true", "yes", "on")


def _dynamic_actor_grad_acc_steps_enabled() -> bool:
  return os.environ.get(
      "TUNIX_ENABLE_DYNAMIC_ACTOR_GRAD_ACC_STEPS", ""
  ).lower() in ("1", "true", "yes", "on")


def _actor_prompt_group_coalesce_factor() -> int:
  raw_value = os.environ.get("TUNIX_ACTOR_PROMPT_GROUP_COALESCE", "").strip()
  if not raw_value:
    return 1
  try:
    value = int(raw_value)
  except ValueError as exc:
    raise ValueError(
        "TUNIX_ACTOR_PROMPT_GROUP_COALESCE must be a positive integer. "
        f"Received: {raw_value!r}"
    ) from exc
  if value <= 0:
    raise ValueError(
        "TUNIX_ACTOR_PROMPT_GROUP_COALESCE must be a positive integer. "
        f"Received: {value}"
    )
  return value


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  completion_bucket_len: jax.Array | None = None
  policy_version: jax.Array | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class AgenticRLConfig(algo_config_lib.AlgorithmConfig):
  """Base configuration for Agentic RL algorithms.

  Parameters:
    system_prompt: System prompt for the agent.
    max_concurrency: Maximum number of concurrent rollout engines.
    off_policy_steps: Number of off-policy steps can be accepted before a
      policy update.
    num_generations: Number of samples per prompt.
    num_iterations: Number of iterations per batch.
  """

  system_prompt: str = ""
  max_concurrency: int = 16
  off_policy_steps: int = 0
  num_generations: int = 1
  num_iterations: int = 1
  enable_rollout_fast_path: bool = False
  rollout_prompt_batch_size: int | None = None
  actor_generation_chunk_size: int | None = None


TConfig = TypeVar("TConfig", bound=AgenticRLConfig)


@dataclasses.dataclass(slots=True)
class _FastPathTrajectoryItem:
  """Minimal trajectory item shape consumed by `_batch_to_train_example`."""

  pair_index: int
  traj: dict[str, Any]


class AgenticRLLearner(abc.ABC, Generic[TConfig]):
  """Base class for Agentic RL Learners using asynchronous rollouts."""

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: TConfig,
      reward_fns: RewardFn | List[RewardFn],
      chat_parser: Any | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `AgenticRLLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: Configuration object.
      reward_fns: Reward functions.
      chat_parser: A parser to handle chat message formatting.
      metric_fns: Metric functions.
      data_shuffle_seed: Seed for data shuffling.
    """
    self.rl_cluster = rl_cluster
    self.algo_config = algo_config
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )
    self.metric_fns = metric_fns or []
    self.rl_cluster.actor_trainer.is_managed_externally = True
    if hasattr(self.rl_cluster, "critic_trainer"):
      self.rl_cluster.critic_trainer.is_managed_externally = True

    self._data_shuffle_seed = (
        jax.random.PRNGKey(data_shuffle_seed)
        if data_shuffle_seed is not None
        else None
    )

    self._training_config = self.rl_cluster.cluster_config.training_config

    self.rl_cluster.global_steps = (
        self.rl_cluster.actor_trainer.restored_global_step()
    )
    # Current iter steps for micro-batch based training.
    self._iter_steps = 0
    self._eval_iter_steps = 0

    # Sync weights if the actor model and rollout model are not sharing weights.
    self.should_sync_weights = not (
        rl_utils.is_sharing_weights(
            self.rl_cluster.actor_trainer.model,
            self.rl_cluster.rollout.model(),
        )
    )

    # Enable async rollout if trainer and rollout are not on the same mesh.
    # If they do, then doesn't make sense for the interleave because they will
    # have resource contention.
    self.can_enable_async_rollout = (
        self.rl_cluster.cluster_config.role_to_mesh[rl_cluster_lib.Role.ACTOR]
        != self.rl_cluster.cluster_config.role_to_mesh[
            rl_cluster_lib.Role.ROLLOUT
        ]
    )
    self.executor = futures.ThreadPoolExecutor(max_workers=1)
    self._last_iter_step = self.rl_cluster.actor_trainer.iter_steps

    self._rollout_micro_batch_size = (
        self._training_config.rollout_micro_batch_size
    )
    self._compute_logps_micro_batch_size = (
        self._training_config.compute_logps_micro_batch_size
    )
    sft_utils.show_hbm_usage(title="AgenticRLLearner init")

    self.chat_parser = chat_parser
    self.tokenizer = rl_cluster.tokenizer
    self.policy_version = 0
    self._rollout_sync_lock = agentic_utils.RolloutSyncLock()
    self._full_batch_size = 0

  def _resolve_rollout_prompt_batch_size(self) -> int:
    """Returns rollout prompt batch size for fast-path mode."""
    configured = self.algo_config.rollout_prompt_batch_size
    if configured is None:
      return _DEFAULT_FAST_PATH_ROLLOUT_PROMPT_BATCH_SIZE
    if configured <= 0:
      raise ValueError(
          "`rollout_prompt_batch_size` must be a positive integer when"
          " fast-path is enabled."
      )
    return configured

  @staticmethod
  def _resolve_adaptive_actor_generation_chunk_size(
      *,
      base_chunk_size: int,
      num_generations: int,
      completion_bucket_len: int,
  ) -> int:
    """Returns a larger actor chunk size for shorter completion buckets."""
    if completion_bucket_len <= 0:
      return base_chunk_size
    if completion_bucket_len <= 1024:
      max_chunk_size = min(num_generations, base_chunk_size * 4)
    elif completion_bucket_len <= 4096:
      max_chunk_size = min(num_generations, base_chunk_size * 2)
    else:
      max_chunk_size = base_chunk_size
    for candidate in range(max_chunk_size, base_chunk_size - 1, -1):
      if (
          candidate % base_chunk_size == 0
          and num_generations % candidate == 0
      ):
        return candidate
    return base_chunk_size

  def _resolve_actor_generation_chunk_size(
      self, completion_bucket_len: int | None = None
  ) -> int:
    """Returns actor-side generation chunk size for training updates."""
    configured = self.algo_config.actor_generation_chunk_size
    if configured is None:
      resolved = self._num_generations()
    else:
      if configured <= 0:
        raise ValueError(
            "`actor_generation_chunk_size` must be a positive integer when set."
        )
      if configured > self._num_generations():
        raise ValueError(
            "`actor_generation_chunk_size` must be <= num_generations. Received:"
            f" {configured} > {self._num_generations()}"
        )
      if self._num_generations() % configured != 0:
        raise ValueError(
            "`actor_generation_chunk_size` must divide num_generations exactly."
            f" Received: actor_generation_chunk_size={configured},"
            f" num_generations={self._num_generations()}"
        )
      resolved = configured
    rollout = getattr(getattr(self, "rl_cluster", None), "rollout", None)
    if (
        resolved > 1
        and rollout is not None
        and rollout.__class__.__name__ == "SglangJaxRollout"
        and _sglang_jax_safe_actor_chunking_enabled()
    ):
      # Keep the external experiment batch the same, but split actor updates
      # down to one generation at a time to reduce JIT executable size on TPU.
      return 1
    if (
        completion_bucket_len is not None
        and resolved > 1
        and rollout is not None
        and rollout.__class__.__name__ == "SglangJaxRollout"
        and _deepscaler_adaptive_actor_chunking_enabled()
    ):
      return self._resolve_adaptive_actor_generation_chunk_size(
          base_chunk_size=resolved,
          num_generations=self._num_generations(),
          completion_bucket_len=completion_bucket_len,
      )
    return resolved

  def _num_actor_chunks_per_prompt_group(self) -> int:
    """Returns how many actor chunks each prompt group is split into."""
    return self._num_generations() // self._resolve_actor_generation_chunk_size()

  @staticmethod
  def _merge_train_examples(train_examples: list[TrainExample]) -> TrainExample:
    """Concatenates per-sequence train examples into one actor batch."""
    if not train_examples:
      raise ValueError("Cannot merge an empty train example list.")
    if len(train_examples) == 1:
      return train_examples[0]
    return jax.tree.map(
        AgenticRLLearner._merge_train_example_leaf, *train_examples
    )

  @staticmethod
  def _merge_train_example_leaf(*xs):
    """Concatenates a TrainExample leaf while preserving optional fields."""
    if all(x is None for x in xs):
      return None
    if any(x is None for x in xs):
      raise ValueError(
          "TrainExample leaves must either all be arrays or all be None when "
          "merging actor batches."
      )
    return jnp.concatenate(xs, axis=0)

  def _split_train_examples_into_prompt_groups(
      self, train_examples: list[TrainExample]
  ) -> list[list[TrainExample]]:
    """Splits a flat per-sequence list into prompt groups of size G."""
    num_generations = self._num_generations()
    if len(train_examples) % num_generations != 0:
      raise ValueError(
          "Expected per-sequence train examples to be divisible by "
          f"num_generations={num_generations}. Received batch size="
          f"{len(train_examples)}."
      )
    return [
        train_examples[i : i + num_generations]
        for i in range(0, len(train_examples), num_generations)
    ]

  @staticmethod
  def _completion_bucket_len_for_prompt_group(
      prompt_group: list[TrainExample],
  ) -> int:
    """Returns the bucketed completion length shared by a prompt group."""
    if not prompt_group:
      raise ValueError("Cannot resolve completion bucket for an empty group.")
    bucket_lens = []
    for train_example in prompt_group:
      if train_example.completion_bucket_len is None:
        bucket_lens.append(int(train_example.completion_ids.shape[-1]))
        continue
      bucket_array = np.asarray(train_example.completion_bucket_len).reshape(-1)
      if bucket_array.size != 1:
        raise ValueError(
            "Each per-sequence TrainExample must carry exactly one "
            f"`completion_bucket_len`. Received shape="
            f"{np.asarray(train_example.completion_bucket_len).shape}."
        )
      bucket_lens.append(int(bucket_array[0]))
    if len(set(bucket_lens)) != 1:
      raise ValueError(
          "All examples from the same prompt group must share the same "
          f"completion bucket. Received: {bucket_lens}"
      )
    return bucket_lens[0]

  def _group_prompt_groups_by_completion_bucket(
      self,
      prompt_groups: list[list[TrainExample]],
  ) -> list[tuple[int, list[list[TrainExample]]]]:
    """Groups prompt groups by completion bucket without crossing group order."""
    grouped_prompt_groups: dict[int, list[list[TrainExample]]] = {}
    bucket_order: list[int] = []
    for prompt_group in prompt_groups:
      bucket_len = self._completion_bucket_len_for_prompt_group(prompt_group)
      if bucket_len not in grouped_prompt_groups:
        grouped_prompt_groups[bucket_len] = []
        bucket_order.append(bucket_len)
      grouped_prompt_groups[bucket_len].append(prompt_group)
    return [
        (bucket_len, grouped_prompt_groups[bucket_len])
        for bucket_len in bucket_order
    ]

  def _chunk_and_merge_train_micro_batch(
      self,
      train_micro_batch: list[TrainExample],
      prompts_per_train_micro_batch: int,
      completion_bucket_len: int | None = None,
  ) -> list[TrainExample]:
    """Splits a prompt micro-batch into smaller actor-side sequence chunks."""
    chunk_size = self._resolve_actor_generation_chunk_size(
        completion_bucket_len=completion_bucket_len
    )
    num_generations = self._num_generations()
    if chunk_size >= num_generations:
      return [self._merge_train_examples(train_micro_batch)]

    if len(train_micro_batch) % num_generations != 0:
      logging.warning(
          "Train micro-batch size %d is not divisible by num_generations=%d. "
          "Falling back to sequential actor chunking.",
          len(train_micro_batch),
          num_generations,
      )
      batch_chunk_size = max(1, prompts_per_train_micro_batch * chunk_size)
      return [
          self._merge_train_examples(train_micro_batch[i : i + batch_chunk_size])
          for i in range(0, len(train_micro_batch), batch_chunk_size)
      ]

    grouped_examples = [
        train_micro_batch[i : i + num_generations]
        for i in range(0, len(train_micro_batch), num_generations)
    ]
    merged_batches = []
    for generation_slice_start in range(0, num_generations, chunk_size):
      chunk_examples = []
      for prompt_group in grouped_examples:
        chunk_examples.extend(
            prompt_group[
                generation_slice_start : generation_slice_start + chunk_size
            ]
        )
      merged_batches.append(self._merge_train_examples(chunk_examples))
    return merged_batches

  def _build_fast_path_chat_messages(
      self, single_example: TrainingInputT
  ) -> list[dict[str, str]]:
    """Builds the single-turn chat messages used by rollout fast-path."""
    question = str(single_example["question"][0])
    return [
        {"role": "system", "content": self.algo_config.system_prompt},
        {"role": "user", "content": question},
    ]

  def _tokenize_chat_messages(
      self,
      messages: list[dict[str, str]],
      *,
      contains_first_msg: bool,
      contains_generation_msg: bool,
  ) -> list[int]:
    if self.tokenizer is None or self.chat_parser is None:
      raise ValueError(
          "rollout fast-path requires tokenizer and chat_parser to tokenize"
          " prompt/completion messages."
      )
    tokens, _ = agentic_utils.tokenize_and_generate_masks(
        messages,
        tokenizer=self.tokenizer,
        parser=self.chat_parser,
        contains_first_msg=contains_first_msg,
        contains_generation_msg=contains_generation_msg,
    )
    return tokens

  @staticmethod
  def _is_memory_exhausted_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "resource_exhausted" in msg
        or "out of memory" in msg
        or "oom" in msg
        or "memory exhausted" in msg
    )

  @staticmethod
  def _should_stop_production(
      stop_event: threading.Event | None,
  ) -> bool:
    return stop_event is not None and stop_event.is_set()

  async def _wait_for_fast_path_step_budget(
      self,
      next_prompt_index: int,
      stop_event: threading.Event | None,
  ) -> bool:
    """Blocks fast-path producer from getting more than one full batch ahead."""
    full_batch_size = getattr(self, "_full_batch_size", 0)
    if full_batch_size <= 0:
      return not self._should_stop_production(stop_event)

    max_steps = getattr(getattr(self, "_training_config", None), "max_steps", None)
    while True:
      global_steps = getattr(self.rl_cluster, "global_steps", 0)
      if not isinstance(global_steps, (int, np.integer)):
        global_steps = 0
      if isinstance(max_steps, (int, np.integer)) and global_steps >= max_steps:
        return False
      if next_prompt_index < (global_steps + 1) * full_batch_size:
        return True
      if self._should_stop_production(stop_event):
        return False
      await asyncio.sleep(0.01)

  def _profile_step_for_prompt_index(self, prompt_index: int) -> int:
    """Returns a stable profiling step for fast-path rollout timing."""
    full_batch_size = getattr(self, "_full_batch_size", 0)
    global_steps = getattr(self.rl_cluster, "global_steps", 0)
    if not isinstance(global_steps, (int, np.integer)):
      global_steps = 0
    if full_batch_size <= 0:
      return global_steps + 1
    return max(global_steps + 1, (prompt_index // full_batch_size) + 1)

  def _async_metrics_step_for_prompt_index(
      self,
      prompt_index: int,
      mode: rl_cluster_lib.Mode,
  ) -> int:
    """Returns a non-stale metrics step for async fast-path logging."""
    global_steps = getattr(self.rl_cluster, "global_steps", 0)
    if not isinstance(global_steps, (int, np.integer)):
      global_steps = 0
    if mode != rl_cluster_lib.Mode.TRAIN:
      return global_steps
    full_batch_size = getattr(self, "_full_batch_size", 0)
    if full_batch_size <= 0:
      return global_steps
    return max(global_steps, prompt_index // full_batch_size)

  def _maybe_get_rollout_completion_tokens(
      self,
      rollout_output: base_rollout.RolloutOutput,
      output_index: int,
  ) -> list[int] | None:
    """Extracts unpadded completion tokens from rollout output when available."""
    output_tokens = getattr(rollout_output, "tokens", None)
    if output_tokens is None:
      return None
    try:
      completion_tokens = np.asarray(output_tokens[output_index], dtype=np.int32)
    except (IndexError, TypeError, ValueError):
      return None

    if completion_tokens.ndim == 0:
      completion_tokens = completion_tokens.reshape(1)
    elif completion_tokens.ndim > 1:
      completion_tokens = completion_tokens.reshape(-1)

    pad_id_fn = getattr(getattr(self.rl_cluster, "rollout", None), "pad_id", None)
    if callable(pad_id_fn):
      pad_id = pad_id_fn()
      while completion_tokens.size and completion_tokens[-1] == pad_id:
        completion_tokens = completion_tokens[:-1]

    return completion_tokens.tolist()

  async def _producer_fast_path(
      self,
      dataset_iterator,
      train_data_queue,
      stop_event: threading.Event | None = None,
  ):
    """Produces training examples using batched rollout generate calls."""
    num_generations = self.algo_config.num_generations
    rollout_prompt_batch_size = self._resolve_rollout_prompt_batch_size()
    prompt_index = 0

    try:
      for full_batch in dataset_iterator:
        if self._should_stop_production(stop_event):
          return
        single_examples = list(
            self._create_micro_batch_iterator(iter([full_batch]), 1)
        )
        for prompt_slice in rl_utils.chunk_slices_by_size(
            stop=len(single_examples),
            step=rollout_prompt_batch_size,
        ):
          if self._should_stop_production(stop_event):
            return
          if not await self._wait_for_fast_path_step_budget(
              prompt_index, stop_event
          ):
            return
          chunk_examples = single_examples[prompt_slice]
          chunk_prompt_indices: list[int] = []
          chunk_chat_messages: list[list[dict[str, str]]] = []
          chunk_prompt_tokens: list[list[int]] = []

          for single_example in chunk_examples:
            messages = self._build_fast_path_chat_messages(single_example)
            prompt_tokens = self._tokenize_chat_messages(
                messages,
                contains_first_msg=True,
                contains_generation_msg=False,
            )
            chunk_prompt_indices.append(prompt_index)
            chunk_chat_messages.append(messages)
            chunk_prompt_tokens.append(prompt_tokens)
            prompt_index += 1

          try:
            rollout_start = time.perf_counter()
            # Run synchronous rollout generation in a worker thread to avoid
            # nested event-loop execution conflicts with uvloop-backed engines.
            rollout_output = await asyncio.to_thread(
                self._generate_with_rollout_lock,
                prompts=chunk_chat_messages,
                apply_chat_template=True,
                mode=rl_cluster_lib.Mode.TRAIN,
                multi_sampling=num_generations,
            )
          except Exception as e:
            if self._is_memory_exhausted_error(e):
              raise RuntimeError(
                  "Rollout fast-path failed due to memory pressure. "
                  f"Current rollout prompt batch size={rollout_prompt_batch_size}. "
                  "Try reducing `--rollout-prompt-batch-size` or "
                  "`--total-generation-steps`."
              ) from e
            raise
          self._buffer_phase_timing(
              "perf/profile/rollout_generate_time",
              time.perf_counter() - rollout_start,
              step=self._profile_step_for_prompt_index(
                  chunk_prompt_indices[0]
              ),
          )

          if self._should_stop_production(stop_event):
            return
          expected_outputs = len(chunk_examples) * num_generations
          if len(rollout_output.text) != expected_outputs:
            raise RuntimeError(
                "Rollout fast-path received an unexpected number of sampled "
                f"outputs: expected {expected_outputs}, got "
                f"{len(rollout_output.text)}."
            )

          for local_prompt_i, single_example in enumerate(chunk_examples):
            if self._should_stop_production(stop_event):
              return
            pair_index = chunk_prompt_indices[local_prompt_i]
            base = local_prompt_i * num_generations
            chat_messages = chunk_chat_messages[local_prompt_i]
            prompt_tokens = chunk_prompt_tokens[local_prompt_i]
            batch_results = []
            for generation_i in range(num_generations):
              completion_text = rollout_output.text[base + generation_i]
              completion_message = {
                  "role": "assistant",
                  "content": completion_text,
              }
              completion_tokens = self._maybe_get_rollout_completion_tokens(
                  rollout_output, base + generation_i
              )
              if completion_tokens is None:
                completion_tokens = self._tokenize_chat_messages(
                    [completion_message],
                    contains_first_msg=False,
                    contains_generation_msg=False,
                )
              batch_results.append(
                  _FastPathTrajectoryItem(
                      pair_index=pair_index,
                      traj={
                          "conversation_text": chat_messages
                          + [completion_message],
                          "prompt_tokens": prompt_tokens,
                          "conversation_tokens": completion_tokens,
                          "policy_version": self.policy_version,
                      },
                  )
              )
            try:
              train_examples = self._batch_to_train_example(
                  batch_results=batch_results,
                  cached_inputs_for_window=[single_example],
                  mode=rl_cluster_lib.Mode.TRAIN,
              )
              for _ in range(self.algo_config.num_iterations):
                for train_example in train_examples:
                  if self._should_stop_production(stop_event):
                    return
                  train_data_queue.put(train_example)
            except Exception as e:
              if not isinstance(e, RuntimeError):
                logging.exception(
                    "Exception in _producer_fast_path while processing batch: %s",
                    e,
                )
              raise
    finally:
      train_data_queue.put(None)

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_cluster_lib.Mode,
      step: int | None = None,
      **kwargs,
  ) -> jax.Array:
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      step: The current training step.
      **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A JAX array (shape `[num_prompts]`) of scalar rewards for each
      prompt-completion pair. The rewards are the sum across all the provided
      reward functions.

    Raises:
        RuntimeError: If 'r' reward is None, indicating a failure to obtain the
        result, or if the length of 'r' reward does not match the length of
        'prompts'.
    """
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)

    num_prompts = len(prompts)
    num_reward_fns = len(self.reward_fns)
    rewards = np.zeros((num_prompts, num_reward_fns))

    # Compute all rewards for each prompt-completion pair.
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kwargs)

      if r is None:
        raise RuntimeError(
            f"Failed to obtain result from {reward_fn.__name__}. Result is"
            " None."
        )
      if isinstance(r, list) and len(r) != len(prompts):
        raise RuntimeError(
            f"Length mismatch after {reward_fn.__name__}: "
            f"len(r)={len(r)}, len(prompts)={num_prompts}. "
            f"Content of r: {r}"
        )

      rewards[:, i] = np.array(r)

    # Sum rewards across all reward functions for each prompt.
    sum_rewards = np.nansum(rewards, axis=1)

    batch_metrics_to_log = {
        "rewards/sum": (sum_rewards, np.mean),
        "rewards/min": (np.min(rewards, axis=1), np.min),
        "rewards/max": (np.max(rewards, axis=1), np.max),
    }
    for i, reward_fn in enumerate(self.reward_fns):
      metric_name = f"rewards/{reward_fn.__name__}"
      batch_metrics_to_log[metric_name] = (rewards[:, i], np.mean)

    if step is not None:
      self.rl_cluster.buffer_metrics_async(
          batch_metrics_to_log, mode=mode, step=step
      )
    else:
      self.rl_cluster.buffer_metrics(batch_metrics_to_log, mode=mode)

    if _LOG_TRAJECTORY_DETAILS:
      for prompt, completion in zip(prompts, completions):
        trajectory_metrics = {
            "prompts": (prompt, None),
            "completions": (completion, None),
        }
        if step is not None:
          self.rl_cluster.buffer_metrics_async(
              trajectory_metrics, mode=mode, step=step
          )
        else:
          self.rl_cluster.buffer_metrics(trajectory_metrics, mode=mode)

    return jnp.array(sum_rewards)

  def _create_micro_batch_iterator(
      self,
      full_batch_iterator: Iterator[TrainingInputT],
      micro_batch_size: int,
  ) -> Iterator[TrainingInputT]:
    """Re-batches large inputs into an iterator of micro-batches.

    Args:
      full_batch_iterator: Iterator yielding large `TrainingInputT` batches.
      micro_batch_size: The desired size of the micro-batches.

    Yields:
      `TrainingInputT` dicts, each with `micro_batch_size` samples.
    """
    buffer = {}

    def get_buffer_len(buf: dict[str, list[Any]]) -> int:
      if not buf:
        return 0
      return len(next(iter(buf.values())))

    for large_batch in full_batch_iterator:
      for key, values in large_batch.items():
        if key not in buffer:
          buffer[key] = []

        if isinstance(values, (np.ndarray, jax.Array)):
          buffer[key].extend(list(values.flatten()))
        elif isinstance(values, (list, tuple)):
          buffer[key].extend(values)
        else:
          buffer[key].append(values)

      while get_buffer_len(buffer) >= micro_batch_size:
        micro_batch = {}
        for key in buffer:
          micro_batch_list_slice = buffer[key][:micro_batch_size]
          micro_batch[key] = np.array(micro_batch_list_slice)
          buffer[key] = buffer[key][micro_batch_size:]

        yield micro_batch

  def _generate_with_rollout_lock(
      self,
      *,
      prompts,
      apply_chat_template: bool,
      mode: rl_cluster_lib.Mode,
      multi_sampling: int = 1,
  ):
    """Runs rollout generation while holding the shared rollout lock."""
    self._rollout_sync_lock.acquire_rollout()
    try:
      return self.rl_cluster.generate(
          prompts=prompts,
          apply_chat_template=apply_chat_template,
          mode=mode,
          multi_sampling=multi_sampling,
      )
    finally:
      self._rollout_sync_lock.release_rollout()

  def _update_actor_with_rollout_lock(
      self, train_ds, eval_ds, skip_jit: bool
  ) -> None:
    """Runs actor updates as an exclusive section against rollout prefetch."""
    self._rollout_sync_lock.acquire_weight_sync()
    try:
      rollout = getattr(self.rl_cluster, "rollout", None)
      flush_cache = getattr(rollout, "flush_cache", None)
      if callable(flush_cache):
        flush_cache()
      release_memory_occupation = getattr(
          rollout, "release_memory_occupation", None
      )
      should_release_rollout = (
          callable(release_memory_occupation)
          and (
              rollout is None
              or rollout.__class__.__name__ != "SglangJaxRollout"
              or _sglang_jax_rollout_release_enabled()
          )
      )
      if should_release_rollout:
        release_memory_occupation()
      self.rl_cluster.update_actor(train_ds, eval_ds, skip_jit)
    finally:
      self._rollout_sync_lock.release_weight_sync()

  def _make_agent_env_pair(
      self, single_example: TrainingInputT, group_id: int | None = None
  ) -> tuple[model_agent.ModelAgent, task_environment.TaskEnvironment]:
    """Constructs an (agent, environment) pair for a single input sample.

    This is used to set up a rollout for one generation within a group.

    Args:
      single_example: A training input containing a single prompt.
      group_id: An identifier to group generations from the same original
        prompt.

    Returns:
      A tuple containing a configured `ModelAgent` and `TaskEnvironment`.
    """

    question_text = single_example["question"][0]
    # Embed original input to avoid materializing the dataset in producer.
    task = {"question": question_text, "original_input": single_example}
    if group_id is not None:
      task["group_id"] = group_id
    # Pass along other metadata from the original example.
    for key, value in single_example.items():
      if key not in ["prompts", "original_input"]:
        task[key] = value[0]
    agent = model_agent.ModelAgent(system_prompt=self.algo_config.system_prompt)
    # TODO: b/456528861 - Support both single-turn and multi-turn from config.
    env = task_environment.TaskEnvironment(
        task=task,
        reward_fn=reward.dummy_reward,
        max_steps=1,
    )
    return agent, env

  def _model_call(self, chat_lists, env: Any = None):
    """Calls model generation."""
    version = self.policy_version

    if env:
      env.task["policy_version"] = version
    result = self.rl_cluster.generate(
        prompts=chat_lists,
        apply_chat_template=True,
        mode=rl_cluster_lib.Mode.TRAIN,
    )
    return result.text[0]

  def _build_orchestrator(self) -> rollout_orchestrator.RolloutOrchestrator:
    """Builds and configures a RolloutOrchestrator for parallel rollouts."""
    engine_defaults = dict(
        model_call=self._model_call,
        final_reward_fn=reward.dummy_reward,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
    )
    return rollout_orchestrator.RolloutOrchestrator(
        engine_cls=trajectory_collect_engine.TrajectoryCollectEngine,
        engine_defaults=engine_defaults,
        max_concurrency=self.algo_config.max_concurrency,
        run_episodes_sequentially=(
            self.rl_cluster.cluster_config.rollout_engine == "sglang_jax"
        ),
        rollout_sync_lock=self._rollout_sync_lock,
    )

  async def _orchestrator_producer(
      self,
      orchestrator: rollout_orchestrator.RolloutOrchestrator,
      prompt_iterator: Iterable[TrainingInputT],
      num_generations: int = 1,
      collect_mode: str = "Token",
  ):
    """Generates trajectory groups using the orchestrator pattern.

    Args:
      orchestrator: The RolloutOrchestrator instance to use.
      prompt_iterator: An iterable yielding single `TrainingInputT` examples.
      num_generations: The number of episodes to run per agent-environment pair.
      collect_mode: The mode for trajectory collection (e.g., "Token").

    Yields:
      A tuple where the first element is a list of trajectory results for a
      group, and the second is a list containing the original `TrainingInputT`
      for that group.
    """

    def pairs_stream_generator():
      """Yield (agent, env) pairs with unique group_id per original prompt."""
      for i, single_example in enumerate(prompt_iterator):
        agent, env = self._make_agent_env_pair(single_example, group_id=i)
        yield agent, env

    # Start producers in the background.
    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pairs_stream_generator(),
            group_size=self.algo_config.num_generations,
            group_key=lambda i, env, traj: env.task["group_id"],
            num_episodes=num_generations,
            collect_mode=collect_mode,
        )
    )

    # Let the producer start and initialize its manager before consuming.
    await asyncio.sleep(0)

    # Consume full groups and yield them with their original input.
    async_generator = orchestrator.yield_batches(
        batch_size=self.algo_config.num_generations
    )
    try:
      async with contextlib.aclosing(async_generator) as stream:
        async for group in stream:
          if group:
            # Retrieve the original input embedded in the task.
            original_input = group[0].traj["original_input"]
            yield group, [original_input]
    except (GeneratorExit, asyncio.CancelledError):
      # This is the normal shutdown path for a generator.
      return
    finally:
      # Ensure the background producer task is cancelled and cleaned up.
      if not producer_task.done():
        producer_task.cancel()

        async def await_cancellation():
          with contextlib.suppress(asyncio.CancelledError):
            await producer_task

        cancellation_task = asyncio.create_task(await_cancellation())
        del cancellation_task

  def _batch_to_train_example(
      self,
      batch_results: list[Any],
      cached_inputs_for_window: list[TrainingInputT],
      mode: rl_cluster_lib.Mode,
  ) -> List[TrainExample]:
    """Converts a group of trajectories into a list of `TrainExample`s.

    Args:
      batch_results: A list of trajectory results from the orchestrator.
      cached_inputs_for_window: The original input data for this group.
      mode: The current mode (TRAIN or EVAL).

    Returns:
      A list of `TrainExample` instances, ready for training.
    """
    # Create a merged training_input where each field from the original input
    # is repeated G times to align with the G completions.
    num_generations = self.algo_config.num_generations
    micro_batches = [cached_inputs_for_window[0]] * num_generations
    training_input = rl_utils.merge_micro_batches(micro_batches)

    prompt_index = batch_results[0].pair_index
    step = self._async_metrics_step_for_prompt_index(prompt_index, mode)
    trajectory_ids = self._compute_trajectory_ids(training_input, prompt_index)
    assert "trajectory_ids" not in training_input
    training_input["trajectory_ids"] = trajectory_ids
    if _LOG_TRAJECTORY_DETAILS:
      for t_id in trajectory_ids:
        self.rl_cluster.buffer_metrics_async(
            {
                "trajectory_ids": (t_id, None),
            },
            mode=mode,
            step=step,
        )
    return self._process_results(
        results=batch_results,
        training_input=training_input,
        mode=mode,
        step=step,
    )

  @abc.abstractmethod
  def _process_results(
      self,
      results: List[Any],
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      step: int | None = None,
  ) -> List[TrainExample]:
    """Processes generation results, computes rewards and advantages."""
    pass

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Unused in AgenticRLLearner."""
    raise NotImplementedError(
        "_generate_and_compute_advantage is not used in AgenticRLLearner"
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, prompt_index: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch."""
    batch_size = len(example["prompts"]) // self.algo_config.num_generations
    if batch_size != 1:
      raise ValueError(
          "_compute_trajectory_ids expects inputs for a single prompt group,"
          f" but got batch_size={batch_size}"
      )
    row_offset = prompt_index
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.algo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.algo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _num_iterations(self) -> int:
    """Returns the number of iterations per batch."""
    return self.algo_config.num_iterations

  def _num_generations(self) -> int:
    """Returns the number of generations per prompt."""
    return self.algo_config.num_generations

  def _buffer_phase_timing(
      self,
      metric_name: str,
      duration_s: float,
      *,
      step: int,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> None:
    """Buffers phase timing metrics when profiling is explicitly enabled."""
    if not _phase_timing_enabled():
      return
    self.rl_cluster.buffer_metrics_async(
        {metric_name: (float(duration_s), np.sum)},
        mode=mode,
        step=step,
    )

  def _log_completed_rl_step(
      self,
      *,
      step_start_time: float | None,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> None:
    """Logs a low-overhead marker once an RL step has fully completed."""
    if step_start_time is None:
      return
    completed_step = self.rl_cluster.global_steps
    self.rl_cluster.log_scalar_immediately(
        "perf/profile/rl_step_complete_marker",
        1.0,
        mode=mode,
        step=completed_step,
    )
    self.rl_cluster.log_scalar_immediately(
        "perf/profile/rl_step_wall_time",
        float(time.perf_counter() - step_start_time),
        mode=mode,
        step=completed_step,
    )

  def _configure_actor_trainer_grad_acc_steps(
      self,
      *,
      actor_train_batch_count: int,
      bucket_group_count: int,
  ) -> None:
    """Aligns actor grad-acc steps with the actual actor train batches."""
    actor_trainer = getattr(getattr(self, "rl_cluster", None), "actor_trainer", None)
    if (
        actor_trainer is None
        or actor_train_batch_count <= 0
        or not _dynamic_actor_grad_acc_steps_enabled()
    ):
      return
    actor_trainer.config.gradient_accumulation_steps = actor_train_batch_count
    self.rl_cluster.log_scalar_immediately(
        "perf/profile/actor_train_batch_count",
        float(actor_train_batch_count),
        mode=rl_cluster_lib.Mode.TRAIN,
        step=self.rl_cluster.global_steps,
    )
    self.rl_cluster.log_scalar_immediately(
        "perf/profile/actor_bucket_group_count",
        float(bucket_group_count),
        mode=rl_cluster_lib.Mode.TRAIN,
        step=self.rl_cluster.global_steps,
    )
    self.rl_cluster.log_scalar_immediately(
        "perf/profile/actor_dynamic_grad_acc_steps",
        float(actor_train_batch_count),
        mode=rl_cluster_lib.Mode.TRAIN,
        step=self.rl_cluster.global_steps,
    )

  @staticmethod
  def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Runs a coroutine, handling existing event loops correctly."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      # asyncio.get_running_loop() raises RuntimeError if no loop is running.
      # If no loop is running, start a new one using asyncio.run().
      return asyncio.run(coro)
    else:
      # If a loop is already running, use it to run the coroutine.
      return loop.run_until_complete(coro)

  async def _producer(
      self,
      orchestrator,
      dataset_iterator,
      train_data_queue,
      stop_event: threading.Event | None = None,
  ):
    """Produces training examples from prompts in the dataset_iterator."""
    if self.algo_config.enable_rollout_fast_path:
      await self._producer_fast_path(
          dataset_iterator, train_data_queue, stop_event
      )
      return
    if orchestrator is None:
      raise ValueError("`orchestrator` must be provided for non-fast-path.")

    def _iterate_micro_batches():
      for item in dataset_iterator:
        if self._should_stop_production(stop_event):
          return
        for prompt in self._create_micro_batch_iterator(iter([item]), 1):
          yield prompt

    prompt_iterator = _iterate_micro_batches()
    try:
      async for batch, cached_inputs in self._orchestrator_producer(
          orchestrator=orchestrator,
          prompt_iterator=prompt_iterator,
          num_generations=self.algo_config.num_generations,
          collect_mode="Token",
      ):
        if self._should_stop_production(stop_event):
          break
        try:
          train_examples = self._batch_to_train_example(
              batch_results=batch,
              cached_inputs_for_window=cached_inputs,
              mode=rl_cluster_lib.Mode.TRAIN,
          )
          iterations = self.algo_config.num_iterations
          for _ in range(iterations):
            for train_example in train_examples:
              if self._should_stop_production(stop_event):
                return
              train_data_queue.put(train_example)
        except Exception as e:
          if not isinstance(e, RuntimeError):
            logging.exception(
                "Exception in _producer while processing batch: %s", e
            )
          raise
    finally:
      # Signal production is complete for this batch, even if errors occurred.
      train_data_queue.put(None)

  def _data_consumer_batch_generator(
      self, queue: queue_lib.AbstractDataQueue, batch_size: int
  ):
    """Yields micro-batches from a queue until a None is received."""
    item_iterator = iter(lambda: queue.get(block=True), None)
    while True:
      batch = list(itertools.islice(item_iterator, batch_size))
      if not batch:
        return  # The iterator is exhausted.
      yield batch

  def train(
      self,
      train_dataset: Iterable[TrainingInputT],
      eval_dataset: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """Main training loop for the AgenticRLLearner."""
    full_batch_iterator = iter(train_dataset)

    try:
      first_item = next(full_batch_iterator)
    except StopIteration:
      logging.warning("Training dataset is empty.")
      self.rl_cluster.close()
      return

    full_batch_size = len(first_item["prompts"])
    self._full_batch_size = full_batch_size
    # Initialize batch sizes.
    mini_batch_size = self._training_config.mini_batch_size or full_batch_size
    train_micro_batch_size = (
        self._training_config.train_micro_batch_size or mini_batch_size
    )
    actor_generation_chunk_size = self._resolve_actor_generation_chunk_size()
    actor_chunks_per_prompt_group = self._num_actor_chunks_per_prompt_group()
    actor_prompt_group_coalesce = _actor_prompt_group_coalesce_factor()
    prompts_per_actor_call = (
        train_micro_batch_size * actor_prompt_group_coalesce
    )
    self._rollout_micro_batch_size = self._rollout_micro_batch_size or 1
    self._compute_logps_micro_batch_size = (
        self._compute_logps_micro_batch_size or 1
    )
    for v, n in [
        (self._rollout_micro_batch_size, f"{self._rollout_micro_batch_size=}"),
        (
            self._compute_logps_micro_batch_size,
            f"{self._compute_logps_micro_batch_size=}",
        ),
        (mini_batch_size, f"{mini_batch_size=}"),
        (
            prompts_per_actor_call,
            f"{prompts_per_actor_call=} (train_micro_batch_size *"
            " actor_prompt_group_coalesce)",
        ),
    ]:
      rl_utils.check_divisibility(v, full_batch_size, n, f"{full_batch_size=}")
    grad_acc_steps = self._training_config.get_with_default(
        "gradient_accumulation_steps", 1
    )
    effective_actor_grad_acc_steps = grad_acc_steps * actor_chunks_per_prompt_group

    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"Training with {full_batch_size=}, {mini_batch_size=},"
        f" {train_micro_batch_size=}, {self._rollout_micro_batch_size=},"
        f" {self._compute_logps_micro_batch_size=},"
        f" actor_generation_chunk_size={actor_generation_chunk_size},"
        f" actor_chunks_per_prompt_group={actor_chunks_per_prompt_group},"
        f" actor_prompt_group_coalesce={actor_prompt_group_coalesce},"
        f" effective_actor_grad_acc_steps={effective_actor_grad_acc_steps}"
    )

    logging.info("Starting AgenticRLLearner training loop.")
    full_dataset_iterator = itertools.chain([first_item], full_batch_iterator)

    all_eval_prompts = (
        list(self._create_micro_batch_iterator(iter(eval_dataset), 1))
        if eval_dataset
        else []
    )

    training_config = self.rl_cluster.cluster_config.training_config

    train_data_queue = queue_lib.SimpleDataQueue(maxsize=0)
    producer_stop_event = threading.Event()

    # 1. Start producer thread to generate rollouts and training examples.
    orchestrator = None
    if not self.algo_config.enable_rollout_fast_path:
      orchestrator = self._build_orchestrator()
    else:
      logging.info(
          "Rollout fast-path enabled. Using batched generate producer with"
          " rollout_prompt_batch_size=%d.",
          self._resolve_rollout_prompt_batch_size(),
      )
    producer_future = self.executor.submit(
        self._run_async,
        self._producer(
            orchestrator,
            full_dataset_iterator,
            train_data_queue,
            producer_stop_event,
        ),
    )

    # 2. Consume training examples and train.
    train_data_gen = self._data_consumer_batch_generator(
        train_data_queue, train_micro_batch_size * self._num_generations()
    )
    train_data_iter = iter(train_data_gen)
    prompt_groups_since_last_sync = 0
    current_rl_step_start_time: float | None = None
    train_error: Exception | None = None
    try:
      while True:
        if self.rl_cluster.global_steps >= self._training_config.max_steps:
          producer_stop_event.set()
          logging.info(
              "Reached max_steps: %d >= %d",
              self.rl_cluster.global_steps,
              self._training_config.max_steps,
          )
          break
        try:
          if prompt_groups_since_last_sync == 0:
            current_rl_step_start_time = time.perf_counter()
          window_prompt_groups = []
          while len(window_prompt_groups) < prompts_per_actor_call:
            try:
              next_prompt_batch = next(train_data_iter)
            except StopIteration:
              break
            window_prompt_groups.extend(
                self._split_train_examples_into_prompt_groups(next_prompt_batch)
            )
          if not window_prompt_groups:
            break
        except StopIteration:
          break
        self._iter_steps += 1

        # Filter out examples that are too old (off-policy).
        filtered_prompt_groups = []
        filtered_policy_versions = []
        for prompt_group in window_prompt_groups:
          group_policy_version = prompt_group[0].policy_version
          if group_policy_version is None:
            filtered_prompt_groups.append(prompt_group)
            continue
          version_value = group_policy_version[0]
          filtered_policy_versions.append(version_value)
          if version_value == -1 or (
              self.policy_version - version_value
              <= self.algo_config.off_policy_steps
          ):
            filtered_prompt_groups.append(prompt_group)
        if not filtered_prompt_groups:
          logging.warning(
              "Skipping prompt-group window: all %d prompt groups are too old."
              " Current policy version: %d, data versions: %s,"
              " off_policy_steps: %d",
              len(window_prompt_groups),
              self.policy_version,
              str(filtered_policy_versions),
              self.algo_config.off_policy_steps,
          )
          continue
        window_prompt_groups = filtered_prompt_groups
        train_micro_batch = list(itertools.chain.from_iterable(window_prompt_groups))

        grouped_bucket_prompt_groups = self._group_prompt_groups_by_completion_bucket(
            window_prompt_groups
        )
        actor_train_batches = []
        for bucket_len, bucket_prompt_groups in grouped_bucket_prompt_groups:
          bucket_train_micro_batch = list(
              itertools.chain.from_iterable(bucket_prompt_groups)
          )
          actor_train_batches.extend(
              self._chunk_and_merge_train_micro_batch(
                  train_micro_batch=bucket_train_micro_batch,
                  prompts_per_train_micro_batch=len(bucket_prompt_groups),
                  completion_bucket_len=bucket_len,
              )
          )
        self._configure_actor_trainer_grad_acc_steps(
            actor_train_batch_count=len(actor_train_batches),
            bucket_group_count=len(grouped_bucket_prompt_groups),
        )

        # --- Evaluation Logic ---
        current_eval_dataset = None
        if (
            all_eval_prompts
            and self.rl_cluster.actor_trainer.train_steps
            % training_config.eval_every_n_steps
            == 0
        ):
          self._eval_iter_steps = 0
          eval_orchestrator = self._build_orchestrator()

          async def _eval_runner_async(current_eval_orchestrator):
            eval_examples = []
            async for batch, cached_inputs in self._orchestrator_producer(
                current_eval_orchestrator,
                all_eval_prompts,
                num_generations=self._num_generations(),
            ):
              train_examples = self._batch_to_train_example(
                  batch,
                  cached_inputs,
                  rl_cluster_lib.Mode.EVAL,
              )
              eval_examples.extend(train_examples)
            return eval_examples

          eval_future = self.executor.submit(
              self._run_async, _eval_runner_async(eval_orchestrator)
          )
          eval_examples = eval_future.result()
          self._eval_iter_steps += 1
          current_eval_dataset = eval_examples

        # --- Training Step ---
        profile_step = self.rl_cluster.global_steps + 1
        actor_update_start = time.perf_counter()
        self._update_actor_with_rollout_lock(
            actor_train_batches, current_eval_dataset, skip_jit
        )
        self._buffer_phase_timing(
            "perf/profile/actor_update_time",
            time.perf_counter() - actor_update_start,
            step=profile_step,
        )
        if hasattr(self.rl_cluster, "critic_trainer"):
          self.rl_cluster.update_critic(
              train_micro_batch, current_eval_dataset, skip_jit
          )

        # --- Weight Sync Logic ---
        prompt_groups_since_last_sync += len(window_prompt_groups)
        if prompt_groups_since_last_sync >= full_batch_size:
          if self.should_sync_weights:
            logging.info("Requesting sync lock to sync weights...")
            self._rollout_sync_lock.acquire_weight_sync()
            try:
              logging.info("Sync lock acquired. Syncing weights.")
              sync_start = time.perf_counter()
              self.rl_cluster.sync_weights()
              self._buffer_phase_timing(
                  "perf/profile/sync_weights_time",
                  time.perf_counter() - sync_start,
                  step=profile_step,
              )
              self.policy_version += 1
              logging.info(
                  "Weights synced. Policy version incremented to %d.",
                  self.policy_version,
              )
            finally:
              self._rollout_sync_lock.release_weight_sync()
              logging.info("Sync lock released.")
          else:
            self.rl_cluster.global_steps += 1
          self._log_completed_rl_step(
              step_start_time=current_rl_step_start_time,
              mode=rl_cluster_lib.Mode.TRAIN,
          )
          if (
              self.rl_cluster.global_steps < self._training_config.max_steps
              and hasattr(self.rl_cluster, "actor_trainer")
          ):
            # The next RL step re-enters actor training after rollout/ref work on
            # the same TPU slice. Drop cached JIT handles so JAX can release the
            # previous executable before the next step loads it again.
            self.rl_cluster.actor_trainer.clear_jitted_step_caches()
          prompt_groups_since_last_sync = 0
          current_rl_step_start_time = None
    except Exception as e:
      train_error = e
      raise
    finally:
      producer_stop_event.set()
      try:
        _ = producer_future.result()
      except Exception:
        if train_error is None:
          raise
        logging.exception(
            "Producer failed while shutting down after a training error."
        )
      finally:
        self.rl_cluster.close()
