"""Utilities for handling and loading datasets in tunix CLI."""
import ast
import functools
import importlib
import os
from typing import Any

import grain
import numpy as np
from tunix.generate import tokenizer_adapter
from tunix.sft.peft_trainer import TrainingInput

Tokenizer = tokenizer_adapter.Tokenizer
TokenizerAdapter = tokenizer_adapter.TokenizerAdapter


def apply_chat_template(x, tokenizer: TokenizerAdapter) -> dict[str, Any]:
  return {
      "prompts": tokenizer.apply_chat_template(
          x["prompt"], tokenize=False, add_generation_prompt=True
      ),
      **{k: v for k, v in x.items() if k != "prompt"},
  }


def parse_call_string(arg_string: str) -> tuple[list[Any], dict[str, Any]]:
  """Parses a string containing function call arguments and keyword arguments.

  Args:
    arg_string: A string representing the arguments of a function call,
      e.g., "'arg1', 123, kwarg1='value', kwarg2=456".

  Returns:
    A tuple containing two elements:
      - A list of positional arguments.
      - A dictionary of keyword arguments.

  Raises:
    ValueError: If the arg_string is not a valid argument syntax.
  """
  if not arg_string.strip():
    return [], {}

  fake_expression = f"dummy_func({arg_string})"
  try:
    tree = ast.parse(fake_expression)
  except SyntaxError as exc:
    raise ValueError(f"Invalid argument syntax: {arg_string}") from exc

  if not tree.body or not isinstance(tree.body[0], ast.Expr):
    raise ValueError(
        f"Internal error: Expected an expression node for '{arg_string}'"
    )

  call_node = tree.body[0].value
  if not isinstance(call_node, ast.Call):
    raise ValueError(f"Internal error: Expected a Call node for '{arg_string}'")

  parsed_args = []
  for node in call_node.args:
    parsed_args.append(ast.literal_eval(node))

  parsed_kwargs = {}
  for keyword in call_node.keywords:
    parsed_kwargs[keyword.arg] = ast.literal_eval(keyword.value)

  return parsed_args, parsed_kwargs


def get_dataset_from_module(
    specifier: str,
    tokenizer: TokenizerAdapter,
    apply_template: bool = True,
):
  """Get dataset from module.

  Examples of specifier:
    - "data.coding" # create_dataset is the default function
    - "data.coding:create_dataset"
    - "data.coding:get_my_dataset"
    - "data.coding:create_dataset(name='coding_v0')"
    - "data.coding:create_dataset('coding_v0', split='train')"
    - "/home/user/project/data/coding.py:get_dataset"

  Args:
    specifier: The specifier of the module.
    tokenizer: The tokenizer to apply to the dataset.

  Returns:
    The dataset.
  Raises:
    ImportError: If the module cannot be imported or loaded.
  """
  if "(" in specifier and ":" in specifier:
    specifier, args_part = specifier.rsplit("(", 1)
  else:
    args_part = ""
  if ":" in specifier:
    specifier, func_spec = specifier.rsplit(":", 1)
  else:
    func_spec = ""
  if os.path.exists(specifier) and specifier.endswith(".py"):
    module_name = os.path.splitext(os.path.basename(specifier))[0]
    spec = importlib.util.spec_from_file_location(module_name, specifier)
    module = importlib.util.module_from_spec(spec)

    if spec is None:
      raise ImportError(f"Failed to create spec for {specifier}")
    if spec.loader is None:
      raise ImportError(f"Failed to get loader for spec {specifier}")
    if module is None:
      raise ImportError(f"Failed to create module for {specifier}")

    try:
      spec.loader.exec_module(module)
    except Exception as e:
      raise ImportError(
          f"Failed to execute module {module_name} from {specifier}: {e}"
      ) from e
  else:
    try:
      module = importlib.import_module(specifier)
    except Exception as e:
      raise ImportError(f"Failed to import module {specifier}: {e}") from e
  args = []
  kwargs = {}
  if func_spec:
    func = getattr(module, func_spec)
    if args_part:
      args_part = args_part.rstrip(")")
      args, kwargs = parse_call_string(args_part)

  else:
    func = module.create_dataset
  dataset = func(*args, **kwargs)
  if apply_template:
    return dataset.map(
        functools.partial(apply_chat_template, tokenizer=tokenizer)
    )
  return dataset


def _get_sft_response_text(record: dict[str, Any]) -> str:
  for key in ("response", "chosen_response", "chosen_responses"):
    value = record.get(key)
    if value is None:
      continue
    if isinstance(value, str):
      return value
    raise ValueError(f"SFT response field '{key}' must be a string.")
  raise KeyError(
      "SFT dataset records must include one of: "
      "'response', 'chosen_response', 'chosen_responses'."
  )


def _build_sft_training_input(
    prompt_text: str,
    response_text: str,
    tokenizer: Tokenizer,
    max_target_length: int,
) -> TrainingInput | None:
  prompt_tokens = tokenizer.tokenize(prompt_text, add_eos=False)
  response_tokens = tokenizer.tokenize(response_text, add_eos=True)

  total_length = len(prompt_tokens) + len(response_tokens)
  if total_length > max_target_length:
    return None

  input_tokens = prompt_tokens.tolist()
  input_tokens.extend(response_tokens.tolist())

  input_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
  pad_size = max_target_length - total_length
  input_tokens.extend([tokenizer.pad_id()] * pad_size)
  input_mask.extend([0] * pad_size)

  return TrainingInput(
      input_tokens=np.array(input_tokens, dtype=prompt_tokens.dtype),
      input_mask=np.array(input_mask, dtype=np.bool_),
  )


def _record_to_sft_training_input(
    record: dict[str, Any],
    tokenizer: Tokenizer,
    max_target_length: int,
) -> TrainingInput | None:
  prompt_text = record["prompts"]
  response_text = _get_sft_response_text(record)
  return _build_sft_training_input(
      prompt_text=prompt_text,
      response_text=response_text,
      tokenizer=tokenizer,
      max_target_length=max_target_length,
  )


def get_sft_dataset_from_module(
    specifier: str,
    tokenizer: Tokenizer,
    max_target_length: int,
):
  """Loads an SFT dataset module and converts records into TrainingInput."""
  dataset = get_dataset_from_module(specifier, tokenizer)
  dataset = dataset.map(
      functools.partial(
          _record_to_sft_training_input,
          tokenizer=tokenizer,
          max_target_length=max_target_length,
      )
  )
  return dataset.filter(lambda x: x is not None)


def post_init_dataset(
    dataset,
    tokenizer: Tokenizer,
    batch_size: int,
    num_batches: int | None,
    max_prompt_length: int | None,
):
  """Applies post-initialization transformations to a dataset.

  This function filters, batches, and optionally limits the number of batches
  in a dataset.

  Args:
    dataset: The input dataset.
    tokenizer: The tokenizer used for prompt length filtering.
    batch_size: The size of each batch.
    num_batches: If not None, the maximum number of batches to yield.
    max_prompt_length: If not None and greater than 0, prompts longer than this
      will be filtered out.

  Returns:
    The processed dataset.
  """
  if max_prompt_length is not None and max_prompt_length > 0:

    def prompt_length_filter(x):
      tokens = tokenizer.tokenize(x["prompts"])
      return len(tokens) <= max_prompt_length

    dataset = dataset.filter(prompt_length_filter).to_iter_dataset()
  dataset = dataset.batch(batch_size)
  if num_batches is not None:
    if isinstance(dataset, grain.MapDataset):
      dataset = dataset[:num_batches]
    else:
      dataset = grain.experimental.LimitIterDataset(dataset, count=num_batches)
  return dataset
