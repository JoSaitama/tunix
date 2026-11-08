"""Fixed audit partitions and read-only rollout replay (no JAX/model imports)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def partition_indices(pool_size, split, candidate_count, heldout_count, seed):
    """Independent permutation streams; increasing a count preserves its prefix."""
    if not 0 < split < pool_size or not 0 < candidate_count <= split or not 0 < heldout_count <= pool_size - split:
        raise ValueError("invalid audit partition sizes")
    a = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    b = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    return a.permutation(split)[:candidate_count], split + b.permutation(pool_size - split)[:heldout_count]


def validate_partition(partition, pool_size, split, candidate_count, heldout_count):
    result = []
    for name, count, low, high in (("candidate_indices", candidate_count, 0, split),
                                   ("heldout_indices", heldout_count, split, pool_size)):
        values = partition.get(name)
        if not isinstance(values, list) or len(values) != count or any(
                isinstance(i, bool) or not isinstance(i, int) or not low <= i < high for i in values):
            raise ValueError(f"invalid {name}: count or source split mismatch")
        if len(set(values)) != count:
            raise ValueError(f"duplicate {name}")
        result.append(np.asarray(values, dtype=np.int64))
    return tuple(result)


def read_partition(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("data_partition", data)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


REPLAY_CONTRACT = (
    "step", "model_id", "lora_rank", "lora_alpha", "reward_mode", "heldout_reward_mode",
    "prompt_length", "completion_length", "temperature", "top_p", "top_k", "beta", "epsilon",
    "pool_prompts", "train_fraction", "dataset_shuffle_seed", "sample_seed", "rollout_seed",
)
PRODUCTION_FILES = ("tunix/rl/grpo/grpo_learner.py", "my_example/rewards.py",
                    "my_example/reward_rank_noise.py", "my_example/prompts.py")


def replay_metadata(root, output=None):
    root = Path(root).expanduser().resolve(strict=True)
    if output is not None:
        output = Path(output).expanduser().resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("replay input/output directories must not overlap")
    manifest = json.loads((root / "audit_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest.get("arguments"), dict) or not isinstance(manifest.get("data_partition"), dict):
        raise ValueError("replay requires an audit manifest with arguments and data_partition")
    if not manifest.get("restored_lora", {}).get("sha256"):
        raise ValueError("replay manifest lacks restored parameter SHA256")
    return root, manifest


def validate_replay_contract(manifest, args, repo):
    original = manifest["arguments"]
    differences = {key: [original.get(key), getattr(args, key)] for key in REPLAY_CONTRACT
                   if original.get(key) != getattr(args, key)}
    if differences:
        raise ValueError(f"replay configuration mismatch: {differences}")
    for name in PRODUCTION_FILES:
        recorded = manifest.get("audit_source_sha256", {}).get(name)
        if recorded != sha256_file(Path(repo) / name):
            raise ValueError(f"replay production source mismatch: {name}")


def load_rollout(path, expected_indices):
    """Read portable NPZ without pickle. Rewards/advantages are re-audited later."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("replay NPZ symlinks are not allowed")
    with np.load(path, allow_pickle=False) as data:
        host = {key: data[key].copy() for key in data.files}
    required = ("prompt_ids", "completion_ids", "completion_mask", "advantages", "clean_rewards",
                "observed_rewards", "noise_selected", "noise_effective", "correct", "text", "answers",
                "prompt_ids_hash", "source_indices", "rollout_seed", "ref_per_token_logps", "old_per_token_logps")
    if any(key not in host for key in required):
        raise ValueError("incomplete fixed rollout NPZ")
    for name in ("prompt_ids", "completion_ids"):
        if host[name].ndim != 2 or host[name].shape[0] != 16 or host[name].dtype.kind not in "iu":
            raise ValueError(f"invalid replay {name}")
    shape = host["completion_ids"].shape
    for name in ("completion_mask", "ref_per_token_logps", "old_per_token_logps"):
        if host[name].shape != shape or not np.isfinite(host[name]).all():
            raise ValueError(f"invalid replay {name}")
    if not np.isin(host["completion_mask"], [0, 1]).all():
        raise ValueError("replay mask must be binary")
    for name in ("advantages", "clean_rewards", "observed_rewards", "correct", "text", "answers",
                 "prompt_ids_hash", "source_indices"):
        if host[name].shape != (16,):
            raise ValueError(f"invalid replay vector {name}")
    for name in ("advantages", "clean_rewards", "observed_rewards"):
        if not np.isfinite(host[name]).all():
            raise ValueError(f"nonfinite replay {name}")
    for name in ("noise_selected", "noise_effective"):
        if host[name].shape != (4,) or host[name].dtype.kind != "b":
            raise ValueError(f"invalid replay {name}")
    if host["correct"].dtype.kind != "b":
        raise ValueError("replay correctness must be boolean")
    np.testing.assert_array_equal(host["source_indices"], np.repeat(expected_indices, 4))
    ids = host["prompt_ids_hash"].reshape(4, 4)
    if any(len(set(row.tolist())) != 1 for row in ids):
        raise ValueError("replay prompt identities are not contiguous four-answer groups")
    return host
