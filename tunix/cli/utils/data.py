"""Utilities for handling and loading datasets in tunix CLI."""

import ast
import functools
import hashlib
import importlib
import json
import os
from typing import Any, Callable, Optional, Union

from absl import logging
import grain
from tunix.generate import tokenizer_adapter

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


def get_dataset_from_module(specifier: str, tokenizer: TokenizerAdapter):
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
  return dataset.map(
      functools.partial(apply_chat_template, tokenizer=tokenizer)
  )


def post_init_dataset(
    dataset,
    tokenizer: TokenizerAdapter,
    batch_size: int,
    num_batches: Optional[int],
    max_prompt_length: Optional[int],
    fraction: float = 1.0,
    num_epochs: int = 1,
    prompt_key: str = "prompts",
    custom_batch_fn: Optional[Callable] = None,
    require_complete_num_batches: bool = False,
    selection_manifest_path: Optional[str] = None,
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
    fraction: Fraction of the dataset to use (between 0.0 and 1.0), commonly
      used for splitting training and validation sets.
    num_epochs: Number of times to repeat the dataset.
  Returns:
    The processed dataset.
  """
  if require_complete_num_batches and num_batches is None:
    raise ValueError(
        "require_complete_num_batches requires an explicit num_batches value."
    )

  if require_complete_num_batches:
    input_size = len(dataset)
    valid_examples = []
    overlength_count = 0
    selection_hasher = hashlib.sha256()
    for index in range(input_size):
      example = dataset[index]
      prompt = example[prompt_key]
      if (
          max_prompt_length is not None
          and max_prompt_length > 0
          and len(tokenizer.encode(prompt)) > max_prompt_length
      ):
        overlength_count += 1
        continue
      valid_examples.append(example)

    required_size = num_batches * batch_size
    valid_size = len(valid_examples)
    if valid_size < required_size:
      raise ValueError(
          "Token-filtered dataset cannot supply the requested complete "
          f"batches: valid_rows={valid_size}, required_rows={required_size}, "
          f"num_batches={num_batches}, batch_size={batch_size}."
      )
    selected_examples = valid_examples[:required_size]
    for index, example in enumerate(selected_examples):
      identity = {
          "selected_index": index,
          "prompt": example[prompt_key],
          "question": example.get("question"),
          "answer": example.get("answer"),
      }
      selection_hasher.update(
          json.dumps(
              identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
          ).encode("utf-8")
      )
      selection_hasher.update(b"\n")
    manifest = {
        "input_rows": input_size,
        "overlength_rows": overlength_count,
        "valid_rows": valid_size,
        "selected_rows": required_size,
        "unused_valid_rows": valid_size - required_size,
        "batch_size": batch_size,
        "full_batches": num_batches,
        "remainder": valid_size % batch_size,
        "max_prompt_length": max_prompt_length,
        "selection_sha256": selection_hasher.hexdigest(),
    }
    logging.info("Strict dataset selection manifest: %s", manifest)
    if selection_manifest_path:
      manifest_path = os.path.abspath(selection_manifest_path)
      os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
      with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    dataset = grain.MapDataset.source(selected_examples)
    max_prompt_length = None

  if max_prompt_length is not None and max_prompt_length > 0:

    def prompt_length_filter(x):
      tokens = tokenizer.encode(x[prompt_key])
      return len(tokens) <= max_prompt_length

    dataset = dataset.filter(prompt_length_filter)

  if num_batches is not None and not require_complete_num_batches:
    target_size = min(num_batches * batch_size, len(dataset))
    dataset = dataset[:target_size]

  if fraction < 1.0 and fraction > 0.0:
    first_segment_size = int(len(dataset) * fraction)
    first_segment_dataset = dataset[:first_segment_size]
    second_segment_dataset = dataset[first_segment_size:]
  else:
    first_segment_dataset = dataset
    second_segment_dataset = None

  first_segment_dataset = (
      first_segment_dataset.repeat(num_epochs)
      .to_iter_dataset()
      .batch(batch_size, batch_fn=custom_batch_fn)
  )
  if second_segment_dataset is not None:
    second_segment_dataset = (
        second_segment_dataset.repeat(num_epochs)
        .to_iter_dataset()
        .batch(batch_size, batch_fn=custom_batch_fn)
    )

  return first_segment_dataset, second_segment_dataset
