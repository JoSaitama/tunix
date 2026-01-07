from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Iterable

import grain
import kagglehub
import tensorflow_datasets as tfds

from .prompts import TEMPLATE, SYSTEM_PROMPT, extract_hash_answer


def _load_from_tfds(data_dir: str, split: str):
    import tensorflow_datasets.text.gsm8k

    return tfds.data_source(
        "gsm8k",
        split=split,
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )


def download_kaggle_dataset(target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
    src = Path(src)
    dst = Path(target_dir)

    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
        print(f"Copied {csv_file.name} -> {dst / csv_file.name}")
    return target_dir


def _as_text(value):
    return value if isinstance(value, str) else value.decode("utf-8")


def _format_example(question: str, answer: str):
    return {
        "prompts": TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
        "question": question,
        "answer": extract_hash_answer(answer),
    }


def _dataset_from_kaggle(data_dir: str, split: str):
    kaggle_dir = download_kaggle_dataset(data_dir)
    file_name = "main_" + split + ".csv"
    csv_path = os.path.join(kaggle_dir, file_name)

    data = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({"question": row["question"], "answer": row["answer"]})

    return data


def get_dataset(data_dir: str, split: str, source: str) -> grain.MapDataset:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if source == "tfds":
        data = _load_from_tfds(data_dir, split)
    elif source == "kaggle":
        data = _dataset_from_kaggle(data_dir, split)
    else:
        raise ValueError(f"Unknown source: {source}")

    dataset = (
        grain.MapDataset.source(data)
        .shuffle(seed=42)
        .map(
            lambda x: _format_example(
                _as_text(x["question"]),
                _as_text(x["answer"]),
            )
        )
    )
    return dataset


def batch_dataset(dataset: grain.MapDataset, batch_size: int, max_examples: int | None):
    batched = dataset.batch(batch_size)
    if max_examples is None:
        return batched
    max_batches = max_examples // batch_size
    return batched[:max_batches]
