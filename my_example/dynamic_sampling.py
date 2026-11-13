# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Opt-in DAPO-style group filtering/refill for the frozen GRPO experiment.

Reference: Yu et al., DAPO, arXiv:2503.14476, section 3.2; Apache-2.0 verl recipe:
https://github.com/verl-project/verl-recipe/blob/
0a9360bb366d8bd71e26fb86da1acf077da9e6d9/dapo/dapo_ray_trainer.py

Only DS is adapted, NOT the full DAPO objective. The selection metric is the
existing observed dense reward (after rank noise). Generation/logps/advantages
reuse the original learner; selected answers are not regenerated. No hash
gradient compression or extra validation labels are used.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext

import numpy as np

UPSTREAM_COMMIT = "0a9360bb366d8bd71e26fb86da1acf077da9e6d9"


def eligible_groups(rewards) -> np.ndarray:
    """Official std>0 rule on finite [prompts, generations] observed rewards."""
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("DS expects nonempty [prompts, generations>=2] rewards")
    if not np.isfinite(values).all():
        raise ValueError("DS rewards must be finite")
    return values.std(axis=1) > 0


class CyclingBatches:
    """Repeat ONLY the supplied training partition in its frozen source order.

    No caching of the full dataset and no borrowing from validation/test.
    One-shot iterators are rejected: silently exhausting them would truncate the
    update budget. Rollouts are fresh on every visit, even across cycles.
    """

    def __init__(self, dataset):
        if iter(dataset) is dataset:
            raise ValueError("DS needs a re-iterable training dataset, not a one-shot iterator")
        self.dataset = dataset

    def __iter__(self):
        while True:
            found = False
            for batch in self.dataset:
                found = True
                yield batch
            if not found:
                raise ValueError("DS training partition is empty")


def concatenate_examples(examples, pad_id: int):
    """Join selected groups without assuming equal per-generation token lengths.

    Prompt fields are left-padded, completion fields right-padded. Masks/logps
    extend by zero; ID fields use the real tokenizer pad ID. Optional None
    fields stay None. This preserves the learner's prompt/completion convention.
    """
    import jax.numpy as jnp

    if not examples:
        raise ValueError("DS cannot concatenate an empty selected batch")
    values = {}
    for field in dataclasses.fields(examples[0]):
        leaves = [getattr(example, field.name) for example in examples]
        if all(leaf is None for leaf in leaves):
            values[field.name] = None
            continue
        if any(leaf is None for leaf in leaves):
            raise ValueError(f"DS inconsistent optional field: {field.name}")
        rank = leaves[0].ndim
        if rank < 1 or any(leaf.ndim != rank for leaf in leaves):
            raise ValueError(f"DS inconsistent array rank: {field.name}")
        shape = tuple(max(leaf.shape[i] for leaf in leaves) for i in range(1, rank))
        constant = pad_id if field.name in ("prompt_ids", "completion_ids") else 0
        is_prompt = field.name in ("prompt_ids", "prompt_mask")
        padded = [jnp.pad(leaf, [(0, 0)] + [
                    (n - leaf.shape[i+1], 0) if is_prompt else (0, n - leaf.shape[i+1])
                    for i, n in enumerate(shape)], constant_values=constant)
                  for leaf in leaves]
        values[field.name] = jnp.concatenate(padded, axis=0)
    return examples[0].replace(**values)


class DynamicSamplingMixin:
    """Synchronous same-policy refill, limited to the frozen one-update path."""

    def __init__(self, *args, ds_max_generation_batches=20, ds_artifact_dir=None, **kwargs):
        if ds_max_generation_batches < 1:
            raise ValueError("DS max generation batches must be positive")
        config, cluster = kwargs["algo_config"], kwargs["rl_cluster"]
        training = cluster.cluster_config.training_config
        if training.use_dynamic_batch_curation or os.environ.get("TUNIX_PODS_KEEP_COUNT", "").strip():
            raise ValueError("DS is standalone: cannot combine with PODS/DBC/DTV/fixed filters")
        if config.num_iterations != 1 or training.get_with_default("gradient_accumulation_steps", 1) != 1:
            raise ValueError("DS currently requires num_iterations=1 and gradient_accumulation_steps=1")
        self._ds_cap = int(ds_max_generation_batches)
        self._ds_dir = Path(ds_artifact_dir) if ds_artifact_dir else None
        self._ds_capture = False
        self._ds_rewards = None
        self._ds_rounds = self._ds_generated = self._ds_selected = self._ds_updates = 0
        self._ds_nonconstant = self._ds_candidate_batches = 0
        super().__init__(*args, **kwargs)
        if self.can_enable_async_rollout:
            raise ValueError("DS currently requires colocated synchronous actor/rollout")
        if self._last_iter_step or self.rl_cluster.actor_trainer.train_steps:
            raise ValueError("DS resume is not implemented; use a new run root (checkpoints remain available for diagnostics)")
        if self._ds_dir:
            self._ds_dir.mkdir(parents=True, exist_ok=True)
            if (self._ds_dir / "ds_selection.jsonl").exists():
                raise ValueError("DS selection log already exists; refusing to append a new session")
            self._write_json("run_metadata.json", {
                "method": "grpo_ds", "upstream_commit": UPSTREAM_COMMIT,
                "selection_metric": "observed_dense_reward_std_gt_zero",
                "num_generations": config.num_generations, "beta": config.beta,
                "max_steps": training.max_steps,
                "max_generation_batches_per_update": self._ds_cap,
                "experiment_seed": os.environ.get("TUNIX_EXPERIMENT_SEED"),
                "mismatch_fraction": os.environ.get("TUNIX_REWARD_RANK_NOISE_FRACTION", "0"),
                "mismatch_seed": os.environ.get("TUNIX_REWARD_RANK_NOISE_SEED", "0"),
                "argv": sys.argv,
                "adaptations": [
                    "DS component only: frozen dense reward, KL, clipping, optimizer and update budget",
                    "reuse original generation, all-G advantages and reference/old-policy logps",
                    "repeat training partition in frozen source order as needed; fresh answers each visit",
                    "same-policy sequential fixed-size generation batches; no cross-update answer cache",
                    "excess eligible groups discarded in source order after filling the target",
                    "reference/old logps computed for rejected groups too; no claimed inference saving",
                    "finite generation cap: fail without a partial actor update",
                    "no automatic resume: dataset cursor and rollout RNG replay not implemented",
                ],
            })
        print(f"GRPO+DS enabled: observed reward std>0; refill cap={self._ds_cap}")

    def train(self, train_ds, eval_ds=None, skip_jit=False):
        # RLLearner still determines 691 updates from the unchanged config.
        return super().train(CyclingBatches(train_ds), eval_ds, skip_jit)

    def _compute_rewards(self, prompts, completions, mode, **kwargs):
        rewards = super()._compute_rewards(prompts=prompts, completions=completions, mode=mode, **kwargs)
        if self._ds_capture:
            self._ds_rewards = (np.asarray(rewards).copy(), list(prompts))
        return rewards

    def _fill_training_batch(self, iterator, group_size):
        import jax
        from tunix.rl import rl_cluster as cluster_lib

        selected, kept = [], 0
        benchmark = getattr(self.rl_cluster, "_training_benchmark", None)
        target = int(self._training_config.train_micro_batch_size)
        started = time.monotonic()
        initial_actor_step = int(self.rl_cluster.actor_trainer.train_steps)
        for generation_batch in range(1, self._ds_cap + 1):
            batch = next(iterator)
            if len(batch["prompts"]) != target:
                raise ValueError("DS requires full fixed-size candidate prompt batches")
            if "trajectory_ids" in batch:
                raise ValueError("DS expects raw prompt batches without trajectory_ids")
            repeated = {key: np.repeat(np.asarray(value), group_size, axis=0)
                        for key, value in batch.items()}
            if any(len(value) != target * group_size for value in repeated.values()):
                raise ValueError("DS input fields must have a prompt-leading dimension")
            self._ds_candidate_batches += 1
            repeated["trajectory_ids"] = [
                f"ds_{self._ds_candidate_batches}_{i}_{r}"
                for i in range(target) for r in range(group_size)]
            capture = (getattr(benchmark, "segmented_xprof_python", None) is not None
                       and benchmark.ds_hbm_active())
            with (benchmark.hbm_segment(f"candidate-{generation_batch:02d}", candidate_batch=generation_batch)
                  if capture else nullcontext()):
                self._ds_capture, self._ds_rewards = True, None
                try:
                    example = super()._generate_and_compute_advantage(repeated, cluster_lib.Mode.TRAIN)
                    if self._ds_rewards is None:
                        raise RuntimeError("DS did not capture observed training rewards")
                    rewards, prompts = self._ds_rewards
                finally:
                    self._ds_capture, self._ds_rewards = False, None
                if rewards.shape != (target * group_size,):
                    raise ValueError("DS rewards do not match complete candidate groups")
                if any(len(set(prompts[i*group_size:(i+1)*group_size])) != 1 for i in range(target)):
                    raise ValueError("DS received a group with different prompts")
                if len(example.advantages) != len(rewards):
                    raise ValueError("DS prepared examples/rewards are misaligned")
                grouped = rewards.reshape(target, group_size)
                eligible = eligible_groups(grouped)
                chosen = np.flatnonzero(eligible)[:target-kept]
                if benchmark is not None:
                    benchmark.note_ds_candidate(prompts[::group_size], int(eligible.sum()), len(chosen))
                self._ds_rounds += 1
                self._ds_generated += target
                self._ds_nonconstant += int(eligible.sum())
                self._ds_selected += len(chosen)
                self._record_round(grouped, eligible, chosen, prompts, generation_batch)
                if len(chosen):
                    flat = (chosen[:, None] * group_size + np.arange(group_size)).reshape(-1)
                    selected.append(jax.tree.map(lambda x: x[flat], example))
                    kept += len(chosen)
                if int(self.rl_cluster.actor_trainer.train_steps) != initial_actor_step:
                    raise RuntimeError("Actor changed while DS was refilling a batch")
                if capture:
                    jax.block_until_ready((example, selected))
                # Drop references to rejected full batches before another generation.
                del example
            if kept == target:
                with (benchmark.hbm_segment("batch-assembly") if capture else nullcontext()):
                    combined = concatenate_examples(selected, self.rl_cluster.rollout.pad_id())
                    if capture:
                        jax.block_until_ready(combined)
                self._ds_updates += 1
                self._write_summary(status="prepared", generation_batches=generation_batch)
                if benchmark is None:
                    self.rl_cluster.buffer_metrics({
                        "ds/generation_batches": (generation_batch, np.mean),
                        "ds/generated_groups": (target * generation_batch, np.sum),
                        "ds/selected_groups": (target, np.sum),
                        "ds/selected_fraction": (1 / generation_batch, np.mean),
                        "ds/refill_seconds": (time.monotonic() - started, np.sum),
                    }, mode=cluster_lib.Mode.TRAIN)
                return combined
        self._write_summary(status="refill_cap_exceeded", generation_batches=self._ds_cap)
        raise RuntimeError(f"DS refill cap exceeded: {kept}/{target} eligible groups after {self._ds_cap} batches; no partial actor update")

    def _prepare_data(self, iterator, proceed_num_steps, sample_repeat, batch_repeat,
                      service_target_batch_size, data_queue, async_loading=False, mode=None):
        from tunix.rl import rl_cluster as cluster_lib

        if mode != cluster_lib.Mode.TRAIN:
            return super()._prepare_data(iterator=iterator, proceed_num_steps=proceed_num_steps,
                sample_repeat=sample_repeat, batch_repeat=batch_repeat,
                service_target_batch_size=service_target_batch_size, data_queue=data_queue,
                async_loading=async_loading, mode=mode)
        try:
            if (proceed_num_steps != 1 or batch_repeat != 1 or async_loading or
                sample_repeat != self.algo_config.num_generations or
                service_target_batch_size != self._training_config.train_micro_batch_size):
                raise ValueError("DS currently supports the frozen synchronous one-microbatch-per-update path only")
            combined = self._fill_training_batch(iterator, sample_repeat)
            # One complete effective batch; only now may the actor update.
            data_queue.put([combined])
            self._iter_steps += 1
        finally:
            data_queue.put(None)

    def _write_json(self, name, value):
        if self._ds_dir:
            temporary = self._ds_dir / (name + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(self._ds_dir / name)

    def _write_summary(self, *, status, generation_batches):
        self._write_json("ds_summary.json", {
            "method": "grpo_ds", "status": status,
            "prepared_updates": self._ds_updates,
            "actor_train_steps_at_preparation": int(self.rl_cluster.actor_trainer.train_steps),
            "counts_scope": "current non-resumed process; selected counts include a cap-failed partial preparation",
            "generation_batches": self._ds_rounds,
            "generated_groups": self._ds_generated, "eligible_groups": self._ds_nonconstant,
            "selected_groups": self._ds_selected,
            "zero_variance_groups": self._ds_generated - self._ds_nonconstant,
            "excess_eligible_groups": self._ds_nonconstant - self._ds_selected,
            "selected_fraction": self._ds_selected / max(self._ds_generated, 1),
            "last_update_generation_batches": generation_batches,
        })

    def _record_round(self, rewards, eligible, chosen, prompts, generation_batch):
        if not self._ds_dir:
            return
        noise = getattr(self, "noise_config", None)
        chosen = set(chosen.tolist())
        n = self.algo_config.num_generations
        with (self._ds_dir / "ds_selection.jsonl").open("a", encoding="utf-8") as stream:
            for i, row in enumerate(rewards):
                prompt = str(prompts[i*n])
                corrupted = False
                if noise is not None:
                    from .reward_rank_noise import selected_prompt
                    corrupted = selected_prompt(prompt, noise)
                stream.write(json.dumps({
                    "update_step": int(self.rl_cluster.actor_trainer.train_steps),
                    "candidate_batch": self._ds_candidate_batches,
                    "generation_batch_in_update": generation_batch,
                    "group_in_batch": i,
                    "prompt_id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                    "observed_rewards": row.tolist(), "observed_reward_std": float(row.std()),
                    "eligible": bool(eligible[i]), "selected": i in chosen,
                    "reason": "selected" if i in chosen else ("excess_eligible" if eligible[i] else "zero_variance"),
                    "mismatch_selected": corrupted,
                }, allow_nan=False) + "\n")


def dynamic_sampling_learner_class(base):
    """Preserve the existing clean or mismatch reward implementation via MRO."""
    class DynamicSamplingLearner(DynamicSamplingMixin, base):
        pass
    return DynamicSamplingLearner
