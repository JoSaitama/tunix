from __future__ import annotations

import os
from typing import Any

import grain
from huggingface_hub import hf_hub_download
import pandas as pd

from .auth import get_hf_token
from .prompts import BOXED_INSTRUCTION


def _read_dataframe(data_path: str) -> pd.DataFrame:
    if data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    if data_path.endswith(".jsonl"):
        return pd.read_json(data_path, lines=True)
    if data_path.endswith(".json"):
        return pd.read_json(data_path)
    raise ValueError(
        f"Unsupported dataset file format for {data_path}. "
        "Use one of: .parquet, .jsonl, .json."
    )


def _resolve_data_path(data_path: str) -> str:
    if os.path.exists(data_path):
        return data_path
    if data_path.startswith(("gs://", "s3://", "hf://")):
        return data_path

    # Convenient fallback for AIME-2024 default path.
    if os.path.basename(data_path) == "train-00000-of-00001.parquet":
        local_path = hf_hub_download(
            repo_id="HuggingFaceH4/aime_2024",
            repo_type="dataset",
            filename="data/train-00000-of-00001.parquet",
            token=get_hf_token(),
        )
        print(
            "AIME dataset not found locally. Downloaded to:",
            local_path,
        )
        return local_path

    # Convenient fallback for DeepScaleR preview train dataset.
    if os.path.basename(data_path) == "deepscaler.json":
        local_path = hf_hub_download(
            repo_id="agentica-org/DeepScaleR-Preview-Dataset",
            repo_type="dataset",
            filename="deepscaler.json",
            token=get_hf_token(),
        )
        print(
            "DeepScaleR train dataset not found locally. Downloaded to:",
            local_path,
        )
        return local_path

    raise FileNotFoundError(f"Dataset path does not exist: {data_path}")


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _pick_field(row: pd.Series, fields: list[str]) -> str | None:
    for field in fields:
        if field in row:
            text = _coerce_text(row[field])
            if text is not None:
                return text
    return None


def _records_from_frame(df: pd.DataFrame) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        question = _pick_field(row, ["problem", "question", "prompt"])
        answer = _pick_field(row, ["answer", "solution", "final_answer"])
        if question is None or answer is None:
            continue
        records.append({"question": question, "answer": answer})
    if not records:
        raise ValueError(
            "No valid records found. Expected columns like "
            "`problem/question` and `answer/solution`."
        )
    return records


def _format_prompt(tokenizer, question: str) -> str:
    content = f"{question}\n{BOXED_INSTRUCTION}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def get_dataset(data_path: str, tokenizer) -> grain.MapDataset:
    resolved_path = _resolve_data_path(data_path)
    records = _records_from_frame(_read_dataframe(resolved_path))
    dataset = (
        grain.MapDataset.source(records)
        .shuffle(seed=42)
        .map(
            lambda x: {
                "prompts": _format_prompt(tokenizer, x["question"]),
                "question": x["question"],
                "answer": x["answer"],
            }
        )
    )
    return dataset


def batch_dataset(
    dataset: grain.MapDataset,
    batch_size: int,
    max_examples: int | None,
) -> grain.MapDataset:
    batched = dataset.batch(batch_size)
    if max_examples is None:
        return batched
    max_batches = max_examples // batch_size
    return batched[:max_batches]
