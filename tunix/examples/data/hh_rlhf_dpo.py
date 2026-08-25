"""HH-RLHF preference-pair dataset adapter for Tunix DPO.

Raw dataset:
  Anthropic/hh-rlhf

Normalized HF columns before Tunix mapping:
  prompt, chosen, rejected

Final Tunix DPO record:
  prompt, chosen_responses, rejected_responses
"""

from __future__ import annotations

import re

from typing import Any

from datasets import load_dataset
from grain import python as grain

from examples.dpo.preference_noise import apply_prompt_response_mismatch_to_hf_dataset
from tunix.examples.data import preference_common


_DATASET_NAME = "Anthropic/hh-rlhf"


def _normalize_split_name(split: str) -> str:
  """Map UltraFeedback-style split names to this dataset's HF splits."""
  normalized = str(split).strip()
  if normalized == "train_prefs":
    return "train"
  if normalized == "test_prefs":
    return "test"
  return normalized


def _clean_text(value: Any) -> str:
  return str(value or "").strip()


_TURN_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):")


def _split_final_assistant_turn(transcript: str) -> tuple[str, str]:
  """Split a HH-RLHF full transcript into prompt and final assistant answer."""
  transcript = _clean_text(transcript)
  marker = "Assistant:"
  pos = transcript.rfind(marker)
  if pos < 0:
    return "", transcript.strip()

  prompt = transcript[: pos + len(marker)].strip()
  response = transcript[pos + len(marker):].strip()
  return prompt, response


def _parse_hh_prompt_messages(prompt_text: str) -> list[dict[str, str]]:
  """Parse HH 'Human:/Assistant:' transcript prefix into chat messages.

  The final empty Assistant marker is ignored, so the prompt normally ends with
  the last user turn. The chosen/rejected fields are used as the final assistant
  response by the DPO pipeline.
  """
  prompt_text = _clean_text(prompt_text)
  matches = list(_TURN_RE.finditer(prompt_text))
  if not matches:
    return [{"role": "user", "content": prompt_text}] if prompt_text else []

  messages: list[dict[str, str]] = []
  for i, match in enumerate(matches):
    speaker = match.group(1)
    content_start = match.end()
    content_end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt_text)
    content = prompt_text[content_start:content_end].strip()

    # Skip the final empty "Assistant:" marker.
    if not content:
      continue

    role = "user" if speaker == "Human" else "assistant"
    messages.append({"role": role, "content": content})

  return messages


def _to_prompt_chosen_rejected(row: dict[str, Any], idx: int) -> dict[str, Any]:
  """Convert HH full chosen/rejected transcripts to prompt/chosen/rejected."""
  chosen_full = _clean_text(row["chosen"])
  rejected_full = _clean_text(row["rejected"])

  chosen_prompt, chosen = _split_final_assistant_turn(chosen_full)
  rejected_prompt, rejected = _split_final_assistant_turn(rejected_full)

  prompt_text = chosen_prompt if chosen_prompt else rejected_prompt
  prompt_messages = _parse_hh_prompt_messages(prompt_text)

  if not prompt_messages or not chosen or not rejected:
    prompt_messages = [{"role": "user", "content": prompt_text or chosen_full}]
    chosen = chosen or chosen_full
    rejected = rejected or rejected_full

  return {
      "prompt": prompt_messages,
      "chosen": chosen,
      "rejected": rejected,
      "source_dataset": _DATASET_NAME,
      "example_id": f"hh_rlhf_{idx}",
  }


def _load_normalized_hf_dataset(
    split: str,
    limit: int | None = None,
    seed: int = 42,
    shuffle_seed: int | None = None,
    partition: str = "all",
    sft_fraction: float = 0.5,
    subset: str = "all",
    eval_fraction: float = 0.0,
):
  """Load HH-RLHF and return a HF dataset with prompt/chosen/rejected columns."""
  dataset = load_dataset(_DATASET_NAME, split=_normalize_split_name(split))
  dataset = dataset.map(
      _to_prompt_chosen_rejected,
      with_indices=True,
      load_from_cache_file=False,
  )

  keep_cols = ["prompt", "chosen", "rejected", "source_dataset", "example_id"]
  remove_cols = [c for c in dataset.column_names if c not in keep_cols]
  if remove_cols:
    dataset = dataset.remove_columns(remove_cols)

  dataset = dataset.filter(
      lambda row: bool(str(row["chosen"]).strip())
      and bool(str(row["rejected"]).strip())
      and str(row["chosen"]).strip() != str(row["rejected"]).strip()
  )

  partition = preference_common.normalize_partition(partition)
  subset = preference_common.normalize_subset(subset)

  if partition != "all":
    dataset = dataset.filter(
        lambda row: preference_common.prompt_is_in_partition(
            row["prompt"],
            partition=partition,
            sft_fraction=sft_fraction,
            seed=seed,
        )
    )

  if subset != "all":
    dataset = dataset.filter(
        lambda row: preference_common.prompt_is_in_subset(
            row["prompt"],
            subset=subset,
            eval_fraction=eval_fraction,
            seed=seed,
        )
    )

  effective_shuffle_seed = seed
  if split.startswith("train") and shuffle_seed is not None:
    effective_shuffle_seed = int(shuffle_seed)

  if limit is not None:
    dataset = dataset.shuffle(seed=effective_shuffle_seed).select(
        range(min(limit, len(dataset)))
    )
  elif split.startswith("train"):
    dataset = dataset.shuffle(seed=effective_shuffle_seed)

  return dataset


def create_dataset(
    split: str,
    limit: int | None = None,
    seed: int = 42,
    shuffle_seed: int | None = None,
    partition: str = "all",
    sft_fraction: float = 0.5,
    subset: str = "all",
    eval_fraction: float = 0.0,
    flip_scope: str = "none",
    flip_ratio: float = 0.0,
    flip_tail_fraction: float = 0.5,
    flip_seed: int = 123,
    no_pref_scope: str = "none",
    no_pref_ratio: float = 0.0,
    no_pref_tail_fraction: float = 0.5,
    no_pref_seed: int = 123,
    no_pref_mode: str = "duplicate_chosen",
    mismatch_scope: str = "none",
    mismatch_ratio: float = 0.0,
    mismatch_seed: int = 123,
    mismatch_mode: str = "response_pair",
):
  """Loads HH-RLHF preference pairs as a Grain dataset."""
  dataset = _load_normalized_hf_dataset(
      split=split,
      limit=limit,
      seed=seed,
      shuffle_seed=shuffle_seed,
      partition=partition,
      sft_fraction=sft_fraction,
      subset=subset,
      eval_fraction=eval_fraction,
  )

  flip_scope = preference_common.normalize_corruption_scope(flip_scope)
  no_pref_scope = preference_common.normalize_corruption_scope(no_pref_scope)
  mismatch_scope = preference_common.normalize_corruption_scope(mismatch_scope)
  no_pref_mode = preference_common.normalize_no_pref_mode(no_pref_mode)
  mismatch_mode = mismatch_mode.strip().lower().replace("-", "_")

  preference_common.validate_ratio("flip_ratio", flip_ratio)
  preference_common.validate_ratio("flip_tail_fraction", flip_tail_fraction)
  preference_common.validate_ratio("no_pref_ratio", no_pref_ratio)
  preference_common.validate_ratio("no_pref_tail_fraction", no_pref_tail_fraction)
  preference_common.validate_ratio("mismatch_ratio", mismatch_ratio)

  active_corruptions = [
      flip_scope != "none" and flip_ratio > 0.0,
      no_pref_scope != "none" and no_pref_ratio > 0.0,
      mismatch_scope != "none" and mismatch_ratio > 0.0,
  ]
  if sum(bool(x) for x in active_corruptions) > 1:
    raise ValueError(
        "Only one corruption type should be enabled at a time: "
        "flip, no_pref, or mismatch."
    )

  if flip_scope != "none" and flip_ratio > 0.0:
    flip_indices = preference_common.select_corruption_indices(
        len(dataset),
        scope=flip_scope,
        ratio=flip_ratio,
        tail_fraction=flip_tail_fraction,
        seed=flip_seed,
    )

    def _maybe_flip(row, idx):
      if idx in flip_indices:
        return {**row, "chosen": row["rejected"], "rejected": row["chosen"]}
      return row

    dataset = dataset.map(_maybe_flip, with_indices=True)

  if no_pref_scope != "none" and no_pref_ratio > 0.0:
    no_pref_indices = preference_common.select_corruption_indices(
        len(dataset),
        scope=no_pref_scope,
        ratio=no_pref_ratio,
        tail_fraction=no_pref_tail_fraction,
        seed=no_pref_seed,
    )

    def _maybe_no_pref(row, idx):
      if idx in no_pref_indices:
        if no_pref_mode == "duplicate_chosen":
          return {**row, "rejected": row["chosen"]}
        if no_pref_mode == "duplicate_rejected":
          return {**row, "chosen": row["rejected"]}
        raise ValueError(f"Unsupported no_pref_mode: {no_pref_mode!r}")
      return row

    dataset = dataset.map(_maybe_no_pref, with_indices=True)

  if mismatch_scope != "none" and mismatch_ratio > 0.0:
    if mismatch_scope != "global":
      raise ValueError(
          "prompt-response mismatch currently supports only global scope, "
          f"got {mismatch_scope!r}."
      )
    dataset = apply_prompt_response_mismatch_to_hf_dataset(
        dataset,
        ratio=mismatch_ratio,
        seed=mismatch_seed,
        mode=mismatch_mode,
    )

  return grain.MapDataset.source(dataset).map(preference_common.to_preference_record)


if __name__ == "__main__":
  ds = _load_normalized_hf_dataset(split="train", limit=5)
  print(ds)
  print(ds.column_names)
  for i in range(min(3, len(ds))):
    row = ds[i]
    print("=" * 80)
    print("prompt:", row["prompt"][:300].replace("\n", "\\n"))
    print("chosen:", row["chosen"][:300].replace("\n", "\\n"))
    print("rejected:", row["rejected"][:300].replace("\n", "\\n"))
