"""Materialized prompt examples and deterministic batching helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np


Example = dict[str, Any]


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def unbatch(dataset: Iterable[Mapping[str, Any]]) -> list[Example]:
    """Materializes a finite batched dataset as prompt-level dictionaries."""
    examples: list[Example] = []
    for batch in dataset:
        if "prompts" not in batch:
            raise ValueError("dataset batch is missing 'prompts'")
        size = len(batch["prompts"])
        for index in range(size):
            examples.append(
                {key: _scalar(values[index]) for key, values in batch.items()}
            )
    return examples


def batch_examples(examples: Sequence[Example]) -> dict[str, np.ndarray]:
    if not examples:
        raise ValueError("cannot batch an empty example list")
    keys = tuple(examples[0])
    if any(tuple(example) != keys for example in examples[1:]):
        raise ValueError("all examples must have the same ordered keys")
    return {
        key: np.asarray([example[key] for example in examples]) for key in keys
    }


def cyclic_batches(
    examples: Sequence[Example],
    *,
    batch_size: int,
    num_batches: int,
) -> Iterator[dict[str, np.ndarray]]:
    if not examples:
        raise ValueError("examples must not be empty")
    if batch_size <= 0 or num_batches < 0:
        raise ValueError("invalid batch_size or num_batches")
    cursor = 0
    for _ in range(num_batches):
        chosen = [examples[(cursor + i) % len(examples)] for i in range(batch_size)]
        cursor = (cursor + batch_size) % len(examples)
        yield batch_examples(chosen)
