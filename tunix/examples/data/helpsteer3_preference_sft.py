"""HelpSteer3-Preference chosen-only dataset adapter for Tunix SFT."""

from __future__ import annotations

from grain import python as grain

from tunix.examples.data import helpsteer3_preference_dpo


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
    domains: str | list[str] | tuple[str, ...] = "all",
    languages: str | list[str] | tuple[str, ...] = "all",
    min_preference_strength: int = 1,
    max_preference_strength: int | None = None,
    drop_identical_responses: bool = False,
):
  """Loads HelpSteer3-Preference chosen responses for SFT."""
  dataset = helpsteer3_preference_dpo._load_normalized_hf_dataset(
      split=split,
      limit=limit,
      seed=seed,
      partition=partition,
      sft_fraction=sft_fraction,
      subset=subset,
      eval_fraction=eval_fraction,
      domains=domains,
      languages=languages,
      min_preference_strength=min_preference_strength,
      max_preference_strength=max_preference_strength,
      drop_identical_responses=drop_identical_responses,
  )
  return grain.MapDataset.source(dataset).map(_to_sft_record)


if __name__ == "__main__":
  ds = helpsteer3_preference_dpo._load_normalized_hf_dataset(
      split="train",
      limit=5,
      domains="general",
  )
  print(ds)
  print(ds.column_names)
  print(ds[0])
