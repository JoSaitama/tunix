"""Safety preference pairs from the official PKU-SafeRLHF-30K splits."""

import logging
import random

from datasets import Dataset
from datasets import load_dataset
from examples.dpo.preference_noise import apply_prompt_response_mismatch_to_hf_dataset
from grain import python as grain

from tunix.examples.data import preference_common


_DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF-30K"


def _prompt_key(prompt: str) -> str:
  return " ".join(prompt.casefold().split())


def create_dataset(
    split: str,
    subset: str = "all",
    limit: int | None = None,
    shuffle_seed: int = 42,
    mismatch_scope: str = "none",
    mismatch_ratio: float = 0.0,
    mismatch_seed: int = 123,
    mismatch_mode: str = "response_pair",
):
  """Builds safety preference pairs with optional training-only mismatch."""
  if split not in ("train", "test"):
    raise ValueError(f"Expected train or test, got {split!r}.")
  subset = preference_common.normalize_subset(subset)
  if split == "test" and subset != "all":
    raise ValueError("The test split only supports subset='all'.")
  if limit is not None and limit <= 0:
    raise ValueError("limit must be positive.")
  mismatch_scope = preference_common.normalize_corruption_scope(mismatch_scope)
  preference_common.validate_ratio("mismatch_ratio", mismatch_ratio)
  apply_mismatch = mismatch_scope != "none" and mismatch_ratio > 0.0
  if apply_mismatch and (
      mismatch_scope != "global" or (split, subset) != ("train", "train")
  ):
    raise ValueError("PKU mismatch requires global scope on the training subset.")

  splits = load_dataset(_DATASET_NAME)
  seen = set()
  if split == "train":
    seen.update(_prompt_key(prompt or "") for prompt in splits["test"]["prompt"])
  rows = []
  for index, row in enumerate(splits[split]):
    prompt = (row["prompt"] or "").strip()
    key = _prompt_key(prompt)
    safe_0 = row["is_response_0_safe"]
    safe_1 = row["is_response_1_safe"]
    if (
        not key
        or key in seen
        or safe_0 is None
        or safe_1 is None
        or safe_0 == safe_1
    ):
      continue
    if not preference_common.prompt_is_in_subset(
        key, subset=subset, eval_fraction=0.1, seed=42
    ):
      continue
    chosen = (row["response_0" if safe_0 else "response_1"] or "").strip()
    rejected = (row["response_1" if safe_0 else "response_0"] or "").strip()
    if not chosen or not rejected or chosen == rejected:
      continue
    seen.add(key)
    rows.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source_dataset": _DATASET_NAME,
        "example_id": f"pku_saferlhf_30k_{split}_{index}",
    })

  if split == "train" and subset != "eval":
    random.Random(shuffle_seed).shuffle(rows)
  selected = rows[:limit]
  if not selected:
    raise ValueError(f"No PKU safe/unsafe pairs in {split}/{subset}.")
  logging.info(
      "PKU-SafeRLHF-30K %s/%s: selected %d of %d eligible unique prompts.",
      split, subset, len(selected), len(rows),
  )
  if apply_mismatch:
    selected = apply_prompt_response_mismatch_to_hf_dataset(
        Dataset.from_list(selected),
        ratio=mismatch_ratio,
        seed=mismatch_seed,
        mode=mismatch_mode,
    )
    logging.info(
        "PKU-SafeRLHF-30K mismatch: %d of %d training rows before length filtering.",
        int(len(selected) * mismatch_ratio), len(selected),
    )
  return grain.MapDataset.source(selected).map(
      preference_common.to_preference_record
  )
