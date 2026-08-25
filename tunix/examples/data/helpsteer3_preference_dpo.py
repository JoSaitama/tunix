"""HelpSteer3-Preference preference-pair dataset adapter for Tunix DPO.

Raw dataset:
  nvidia/HelpSteer3, preference subset/config

Normalized HF columns before Tunix mapping:
  prompt, chosen, rejected

Final Tunix DPO record:
  prompt, chosen_responses, rejected_responses

Domain controls:
  domains="all"          -> General + STEM + Code + Multilingual
  domains="general"      -> General only
  domains="stem"         -> STEM only
  domains="code"         -> Code only
  domains="multilingual" -> Multilingual only
  domains="english"      -> General + STEM + Code, excluding Multilingual
  domains="general,stem" -> comma-separated custom subset

Notes:
  preference_score < 0 means Response 1 is preferred.
  preference_score > 0 means Response 2 is preferred.
  preference_score == 0 is excluded by default for clean DPO.
"""

from __future__ import annotations

import json
import os
from typing import Any

from datasets import load_dataset
from grain import python as grain

from examples.dpo.preference_noise import apply_prompt_response_mismatch_to_hf_dataset
from tunix.examples.data import preference_common


_DATASET_NAME = "nvidia/HelpSteer3"
_CONFIG_NAME = "preference"


def _normalize_split_name(split: str) -> str:
  normalized = str(split).strip().lower()
  if normalized in ("train", "train_prefs"):
    return "train"
  if normalized in ("validation", "val", "eval", "test", "test_prefs"):
    return "validation"
  return normalized


def _clean_text(value: Any) -> str:
  """Clean surrounding whitespace only. Do not truncate content."""
  return str(value or "").strip()


def _metadata_to_str(value: Any) -> str:
  """Store list/dict metadata safely as a string column."""
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  try:
    return json.dumps(value, ensure_ascii=False)
  except TypeError:
    return str(value)


def _env_override_arg(env_name: str, value: Any) -> Any:
  """Allow existing DPO launchers to control HelpSteer3 filters by env.

  If the caller does not explicitly pass a non-default value, environment
  variables such as HELPSTEER3_DOMAINS can override the adapter default.
  """
  env_value = os.environ.get(env_name, "").strip()
  if not env_value:
    return value

  if value is None:
    return env_value

  if isinstance(value, str) and value.strip().lower() in (
      "",
      "all",
      "env",
      "default",
  ):
    return env_value

  return value


def _first_existing_key(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
  for key in candidates:
    if key in row:
      return key
  raise KeyError(
      f"None of the candidate keys {candidates} found. "
      f"Available keys: {sorted(row.keys())}"
  )


def _parse_preference_score(value: Any) -> int:
  try:
    return int(value)
  except Exception as exc:
    raise ValueError(f"Invalid preference score value: {value!r}") from exc


def _normalize_domains(domains: str | list[str] | tuple[str, ...]) -> set[str] | None:
  """Normalize domain filter.

  Returns:
    None means keep all domains.
    Otherwise returns lowercase domain names to keep.
  """
  if domains is None:
    return None

  if isinstance(domains, str):
    raw_items = [
        item.strip().lower()
        for item in domains.replace(";", ",").split(",")
        if item.strip()
    ]
  else:
    raw_items = [str(item).strip().lower() for item in domains if str(item).strip()]

  if not raw_items or "all" in raw_items:
    return None

  expanded: set[str] = set()
  for item in raw_items:
    if item in ("general", "gen"):
      expanded.add("general")
    elif item in ("stem", "science"):
      expanded.add("stem")
    elif item in ("code", "coding", "programming"):
      expanded.add("code")
    elif item in ("multilingual", "multi", "multilang"):
      expanded.add("multilingual")
    elif item in ("english", "en"):
      # Paper-style English RM setting: General + STEM + Code.
      # We intentionally exclude Multilingual here.
      expanded.update(["general", "stem", "code"])
    else:
      raise ValueError(
          f"Unsupported domains value {item!r}. Supported: "
          "all, general, stem, code, multilingual, english, or comma-separated."
      )

  return expanded


def _normalize_languages(
    languages: str | list[str] | tuple[str, ...],
) -> set[str] | None:
  """Normalize optional language filter.

  Usually keep languages="all" for DTV experiments because HelpSteer3's
  language column may represent natural language for multilingual data and
  programming language for code data.
  """
  if languages is None:
    return None

  if isinstance(languages, str):
    raw_items = [
        item.strip().lower()
        for item in languages.replace(";", ",").split(",")
        if item.strip()
    ]
  else:
    raw_items = [str(item).strip().lower() for item in languages if str(item).strip()]

  if not raw_items or "all" in raw_items:
    return None

  aliases = {
      "en": "english",
      "eng": "english",
      "zh": "chinese",
      "cn": "chinese",
      "ko": "korean",
      "kr": "korean",
      "jp": "japanese",
      "ja": "japanese",
  }

  return {aliases.get(item, item) for item in raw_items}


def _load_raw_helpsteer3_preference(split: str):
  """Load the HelpSteer3 preference subset.

  The primary expected format is:
    load_dataset("nvidia/HelpSteer3", "preference", split=...)
  Fallbacks are kept to make the adapter robust to HF packaging changes.
  """
  hf_split = _normalize_split_name(split)

  try:
    return load_dataset(_DATASET_NAME, _CONFIG_NAME, split=hf_split)
  except Exception as first_exc:
    try:
      return load_dataset(_DATASET_NAME, data_dir=_CONFIG_NAME, split=hf_split)
    except Exception:
      try:
        return load_dataset(_DATASET_NAME, split=hf_split)
      except Exception as final_exc:
        raise RuntimeError(
            "Failed to load HelpSteer3 preference dataset using config, "
            "data_dir, and default loading patterns."
        ) from final_exc


def _context_to_prompt(context: Any) -> str:
  """Convert HelpSteer3 context into a single prompt string without truncation.

  HelpSteer3 context is usually a list of chat messages:
    [{"role": "user", "content": "..."}]
  For multi-turn context, we preserve all turns in a readable text format.
  """
  if isinstance(context, str):
    return context.strip()

  if isinstance(context, list):
    parts = []
    for msg in context:
      if isinstance(msg, dict):
        role = _clean_text(msg.get("role", ""))
        content = _clean_text(msg.get("content", ""))
        if role and content:
          parts.append(f"{role.capitalize()}: {content}")
        elif content:
          parts.append(content)
      else:
        text = _clean_text(msg)
        if text:
          parts.append(text)
    return "\n\n".join(parts).strip()

  if isinstance(context, dict):
    role = _clean_text(context.get("role", ""))
    content = _clean_text(context.get("content", ""))
    if role and content:
      return f"{role.capitalize()}: {content}".strip()
    if content:
      return content.strip()

  return _clean_text(context)


def _to_prompt_chosen_rejected(row: dict[str, Any], idx: int) -> dict[str, Any]:
  context_key = _first_existing_key(row, ("context", "prompt", "messages"))
  response_1_key = _first_existing_key(
      row, ("response_1", "response1", "response_a", "response_A")
  )
  response_2_key = _first_existing_key(
      row, ("response_2", "response2", "response_b", "response_B")
  )
  score_key = _first_existing_key(
      row,
      (
          "overall_preference",
          "preference_score",
          "preference_strength",
          "score",
          "overall_score",
      ),
  )

  prompt = _context_to_prompt(row.get(context_key, ""))
  response_1 = _clean_text(row.get(response_1_key, ""))
  response_2 = _clean_text(row.get(response_2_key, ""))
  raw_score = _parse_preference_score(row.get(score_key))

  if raw_score < 0:
    chosen, rejected = response_1, response_2
    preferred_response = "response_1"
  elif raw_score > 0:
    chosen, rejected = response_2, response_1
    preferred_response = "response_2"
  else:
    # Clean DPO excludes score=0 by default.
    # Kept only for future no-preference analysis.
    chosen, rejected = response_1, response_2
    preferred_response = "tie"

  domain = _clean_text(row.get("domain", "")).lower()
  language = _clean_text(row.get("language", "")).lower()

  return {
      "prompt": prompt,
      "chosen": chosen,
      "rejected": rejected,
      "source_dataset": _DATASET_NAME,
      "example_id": f"helpsteer3_preference_{domain}_{idx}",
      "domain": domain,
      "language": language,
      "raw_preference_score": raw_score,
      "preference_strength": abs(raw_score),
      "preferred_response": preferred_response,
      "individual_preferences": _metadata_to_str(
          row.get("individual_preferences", row.get("individual_preference", ""))
      ),
      "raw_context": _metadata_to_str(row.get(context_key, "")),
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
    domains: str | list[str] | tuple[str, ...] = "all",
    languages: str | list[str] | tuple[str, ...] = "all",
    min_preference_strength: int = 1,
    max_preference_strength: int | None = None,
    drop_identical_responses: bool = False,
):
  """Load HelpSteer3-Preference and return prompt/chosen/rejected columns."""
  domains = _env_override_arg("HELPSTEER3_DOMAINS", domains)
  languages = _env_override_arg("HELPSTEER3_LANGUAGES", languages)

  target_domains = _normalize_domains(domains)
  target_languages = _normalize_languages(languages)

  dataset = _load_raw_helpsteer3_preference(split)

  required_any_groups = [
      ("context", "prompt", "messages"),
      ("response_1", "response1", "response_a", "response_A"),
      ("response_2", "response2", "response_b", "response_B"),
      (
          "overall_preference",
          "preference_score",
          "preference_strength",
          "score",
          "overall_score",
      ),
  ]

  available = set(dataset.column_names)
  for group in required_any_groups:
    if not any(key in available for key in group):
      raise ValueError(
          f"HelpSteer3 dataset missing one of required columns {group}. "
          f"Available columns: {dataset.column_names}"
      )

  if "domain" in dataset.column_names and target_domains is not None:
    dataset = dataset.filter(
        lambda row: _clean_text(row.get("domain", "")).lower() in target_domains
    )

  if "language" in dataset.column_names and target_languages is not None:
    dataset = dataset.filter(
        lambda row: _clean_text(row.get("language", "")).lower() in target_languages
    )

  if min_preference_strength < 0:
    raise ValueError(
        f"min_preference_strength must be >= 0, got {min_preference_strength}."
    )
  if max_preference_strength is not None and max_preference_strength < 0:
    raise ValueError(
        f"max_preference_strength must be >= 0, got {max_preference_strength}."
    )
  if (
      max_preference_strength is not None
      and max_preference_strength < min_preference_strength
  ):
    raise ValueError(
        "max_preference_strength must be >= min_preference_strength, got "
        f"{max_preference_strength} < {min_preference_strength}."
    )

  score_col = next(
      key
      for key in (
          "overall_preference",
          "preference_score",
          "preference_strength",
          "score",
          "overall_score",
      )
      if key in dataset.column_names
  )

  def _strength_in_range(row):
    strength = abs(_parse_preference_score(row[score_col]))
    if strength < min_preference_strength:
      return False
    if max_preference_strength is not None and strength > max_preference_strength:
      return False
    return True

  dataset = dataset.filter(_strength_in_range)

  dataset = dataset.map(
      _to_prompt_chosen_rejected,
      with_indices=True,
      load_from_cache_file=False,
  )

  keep_cols = [
      "prompt",
      "chosen",
      "rejected",
      "source_dataset",
      "example_id",
      "domain",
      "language",
      "raw_preference_score",
      "preference_strength",
      "preferred_response",
      "individual_preferences",
      "raw_context",
  ]
  remove_cols = [c for c in dataset.column_names if c not in keep_cols]
  if remove_cols:
    dataset = dataset.remove_columns(remove_cols)

  dataset = dataset.filter(
      lambda row: bool(str(row["prompt"]).strip())
      and bool(str(row["chosen"]).strip())
      and bool(str(row["rejected"]).strip())
  )

  if drop_identical_responses:
    dataset = dataset.filter(
        lambda row: str(row["chosen"]).strip() != str(row["rejected"]).strip()
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
  normalized_split = _normalize_split_name(split)
  if normalized_split == "train" and shuffle_seed is not None:
    effective_shuffle_seed = int(shuffle_seed)

  if limit is not None:
    dataset = dataset.shuffle(seed=effective_shuffle_seed).select(
        range(min(limit, len(dataset)))
    )
  elif normalized_split == "train":
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
    domains: str | list[str] | tuple[str, ...] = "all",
    languages: str | list[str] | tuple[str, ...] = "all",
    min_preference_strength: int = 1,
    max_preference_strength: int | None = None,
    drop_identical_responses: bool = False,
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
  """Loads HelpSteer3-Preference pairs as a Grain dataset."""
  env_train_min_pref = os.environ.get("HELPSTEER3_TRAIN_MIN_PREFERENCE_STRENGTH", "").strip()
  env_min_pref = os.environ.get("HELPSTEER3_MIN_PREFERENCE_STRENGTH", "").strip()
  if str(subset).strip().lower() == "train" and env_train_min_pref:
    min_preference_strength = int(env_train_min_pref)
  elif env_min_pref:
    min_preference_strength = int(env_min_pref)

  dataset = _load_normalized_hf_dataset(
      split=split,
      limit=limit,
      seed=seed,
      shuffle_seed=shuffle_seed,
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
  for domains in ["all", "english", "general", "stem", "code", "multilingual"]:
    ds = _load_normalized_hf_dataset(
        split="train",
        limit=3,
        domains=domains,
    )
    print("=" * 80)
    print("domains:", domains)
    print(ds)
    print(ds.column_names)
    if len(ds) > 0:
      row = ds[0]
      print("domain:", row.get("domain", ""))
      print("language:", row.get("language", ""))
      print("raw_preference_score:", row.get("raw_preference_score", ""))
      print("preference_strength:", row.get("preference_strength", ""))
      print("preferred_response:", row.get("preferred_response", ""))
      print("prompt_len:", len(row["prompt"]))
      print("chosen_len:", len(row["chosen"]))
      print("rejected_len:", len(row["rejected"]))
      print("prompt_head:", row["prompt"][:200].replace("\n", "\\n"))
      print("chosen_head:", row["chosen"][:200].replace("\n", "\\n"))
      print("rejected_head:", row["rejected"][:200].replace("\n", "\\n"))
