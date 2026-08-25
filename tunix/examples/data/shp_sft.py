"""SHP chosen-only dataset adapter for Tunix SFT."""

from __future__ import annotations

from grain import python as grain

from tunix.examples.data import shp_dpo


def _to_sft_record(row):
  return {
      "prompt": [{"role": "user", "content": row["prompt"]}],
      "response": row["chosen"],
  }


def create_dataset(
    split: str,
    limit: int | None = None,
    seed: int = 42,
    partition: str = "sft",
    sft_fraction: float = 0.5,
    subset: str = "all",
    eval_fraction: float = 0.0,
):
  """Loads SHP chosen responses for SFT."""
  dataset = shp_dpo._load_normalized_hf_dataset(
      split=split,
      limit=limit,
      seed=seed,
      partition=partition,
      sft_fraction=sft_fraction,
      subset=subset,
      eval_fraction=eval_fraction,
  )
  return grain.MapDataset.source(dataset).map(_to_sft_record)


if __name__ == "__main__":
  ds = shp_dpo._load_normalized_hf_dataset(split="train", limit=5)
  print(ds)
  print(ds.column_names)
  print(ds[0])
