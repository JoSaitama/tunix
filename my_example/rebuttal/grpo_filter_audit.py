"""Collect and summarize early-training filtering decisions, not utility probes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


METHODS = {"dtv": "PolicySelfInfTrainer", "dtv-loo": "PolicySelfInfLooTrainer"}
CONDITIONS = {"clean": 0.0, "mismatch0.2": 0.2}
SEEDS = (5, 0, 13, 21, 42)
GROUP_SIZE = 4
BATCH_SIZE = 16
SCHEDULE_STEPS = 691
FORMAL_STEPS = 32
SCHEMA = "grpo-filter-audit-v1"
HASH = re.compile(r"[0-9a-f]{64}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class FilterAudit:
    def __init__(self, method, condition, seed, steps, output):
        require(method in METHODS and condition in CONDITIONS, "Invalid method or condition")
        require(seed in SEEDS and steps in {1, FORMAL_STEPS}, "Invalid seed or window")
        self.method, self.condition, self.seed, self.steps = method, condition, seed, steps
        self.fraction = CONDITIONS[condition]
        self.out = Path(output)
        self.prompt_batches = []
        self.batches = []

    @classmethod
    def from_env(cls):
        audit = cls(
            os.environ["TUNIX_FILTER_AUDIT_METHOD"], os.environ["TUNIX_FILTER_AUDIT_CONDITION"],
            int(os.environ["TUNIX_EXPERIMENT_SEED"]), int(os.environ["TUNIX_FILTER_AUDIT_STEPS"]),
            os.environ["TUNIX_FILTER_AUDIT_OUT"],
        )
        require(float(os.environ["TUNIX_REWARD_RANK_NOISE_FRACTION"]) == audit.fraction
                and int(os.environ["TUNIX_REWARD_RANK_NOISE_SEED"]) == audit.seed
                and os.environ["TUNIX_REWARD_MODE"] == "accuracy", "Invalid audit reward setup")
        return audit

    def check_config(self, cfg, max_steps):
        require((max_steps, cfg.data.train_micro_batch_size, cfg.grpo.num_generations,
                 cfg.grpo.num_iterations, cfg.data.num_epochs, cfg.data.max_train_examples,
                 cfg.data.train_fraction, cfg.data.source)
                == (SCHEDULE_STEPS, 4, 4, 1, 1, 3072, 0.9, "tfds"),
                "Expected production GSM8K data and 691-update budget")
        require((cfg.model.model_id, cfg.lora.rank, cfg.lora.alpha,
                 cfg.grpo.max_prompt_length, cfg.grpo.total_generation_steps, cfg.grpo.beta,
                 cfg.grpo.epsilon, cfg.grpo.temperature, cfg.grpo.top_p, cfg.grpo.top_k)
                == ("google/gemma-3-1b-it", 64, 64, 256, 768, 0.08, 0.2, 1.0, 0.95, 64),
                "Expected production model/LoRA/GRPO settings")
        require((cfg.training.learning_rate, cfg.training.warmup_fraction, cfg.training.b1,
                 cfg.training.b2, cfg.training.weight_decay, cfg.training.max_grad_norm)
                == (1e-6, 0.1, 0.9, 0.99, 0.1, 0.1), "Unexpected optimizer setup")
        require(not (cfg.runtime.use_wandb or cfg.runtime.eval_before_train
                     or cfg.runtime.eval_after_train), "Disable audit evaluation and WandB")

    def check_topology(self, jax, mesh):
        require((jax.process_count(), jax.process_index(), jax.local_device_count(),
                 jax.device_count(), tuple(mesh.devices.shape)) == (1, 0, 4, 4, (4, 1))
                and all(d.platform == "tpu" and d.process_index == 0 for d in mesh.devices.flat),
                "Expected four local TPU devices on one worker")

    def configure_loop(self, learner, full, mini, micro, service, accumulation):
        require((full, mini, micro, service, accumulation) == (4, 4, 4, 4, 1)
                and not learner.can_enable_async_rollout,
                "Expected one four-prompt rollout batch and actor update per iteration")

    def encode_rollouts(self, prompts, noise_flags):
        """Attach compact numeric row IDs/flags that follow TrainExample slicing."""
        import numpy as np

        require(len(prompts) == BATCH_SIZE, "Expected 16 generated candidates")
        groups = [prompts[i:i + GROUP_SIZE] for i in range(0, BATCH_SIZE, GROUP_SIZE)]
        require(all(all(x == g[0] for x in g) for g in groups), "Noncontiguous prompt group")
        if self.fraction:
            require(len(noise_flags) == 1, "Expected the production reversal's group labels")
            selected, effective = (np.asarray(x) for x in noise_flags[0])
        else:
            require(not noise_flags, "Clean audit unexpectedly captured reversal")
            selected = effective = np.zeros(4, dtype=bool)
        require(selected.shape == effective.shape == (4,)
                and selected.dtype == effective.dtype == np.dtype(bool)
                and not (effective & ~selected).any(), "Invalid group noise flags")
        self.prompt_batches.append(tuple(hashlib.sha256(str(g[0]).encode()).hexdigest() for g in groups))
        return np.column_stack((
            np.full(BATCH_SIZE, len(self.prompt_batches), dtype=np.int32),
            np.arange(BATCH_SIZE, dtype=np.int32),
            np.repeat(selected, GROUP_SIZE).astype(np.int32),
            np.repeat(effective, GROUP_SIZE).astype(np.int32),
        ))

    def capture(self, aux):
        """Read only the final mask, the actual loss advantages and their row IDs."""
        import numpy as np

        metadata, advantage, kept, skipped = map(np.asarray, self.jax.device_get(aux))
        step = len(self.batches) + 1
        require(step <= self.steps and len(self.prompt_batches) == step
                and self.actor.train_steps == step - 1, "Unexpected update/capture order")
        require(metadata.shape == (BATCH_SIZE, 4) and advantage.shape == kept.shape == (BATCH_SIZE,)
                and skipped.shape == () and kept.dtype == np.dtype(bool)
                and np.isfinite(advantage).all(), "Malformed actor audit output")
        rows = []
        for i, ((batch, index, selected, effective), adv, keep) in enumerate(zip(metadata, advantage, kept)):
            require((batch, index) == (step, i) and selected in {0, 1} and effective in {0, 1},
                    "Actor input and rollout row IDs disagree")
            rows.append({
                "prompt_id": self.prompt_batches[step - 1][i // GROUP_SIZE],
                "group_index": i // GROUP_SIZE, "completion_index": i % GROUP_SIZE,
                "advantage": float(adv), "keep": bool(keep),
                "noise_selected": bool(selected), "noise_effective": bool(effective),
            })
        entry = {"step": step, "skipped_samples": float(skipped), "rows": rows}
        validate_batch(entry, step, self.condition, self.method)
        self.batches.append(entry)

    def after_iteration(self):
        updates = self.cluster.global_steps
        require(updates == self.actor.train_steps == len(self.batches) <= self.steps,
                "Learner, actor and recorded update counts disagree")
        if updates % 8 == 0:
            print(f"Audit {self.condition}/{self.method}/seed{self.seed}: {updates}/{self.steps} updates",
                  flush=True)
        return updates == self.steps

    def optimizer_steps(self):
        optimizer = self.actor.optimizer

        def scalar(value):
            value = value.get_value() if hasattr(value, "get_value") else getattr(value, "value", value)
            return int(self.jax.device_get(value))

        steps = scalar(optimizer.opt_state.gradient_step)
        require(steps == scalar(optimizer.step), "NNX and Optax update counts disagree")
        return steps

    def run(self, jax, nnx, mesh, learner, dataset, model_path):
        import numpy as np

        self.jax, self.cluster = jax, learner.rl_cluster
        self.actor = self.cluster.actor_trainer
        expected_learner = "RewardRankNoiseGRPOLearner" if self.fraction else "GRPOLearner"
        require(type(learner).__name__ == expected_learner
                and type(self.actor).__name__ == METHODS[self.method]
                and (self.actor.scope, self.actor.num_generations) == ("group", GROUP_SIZE)
                and self.actor._score_loss_fn is not None, "Unexpected production learner/actor route")
        if self.fraction:
            require((learner.noise_config.fraction, learner.noise_config.seed) == (self.fraction, self.seed),
                    "Unexpected actual noise config")
        if self.method == "dtv-loo":
            require(self.actor.min_keep_fraction == 0.25 and not self.actor.decisions_path
                    and self.actor.mask_application == "full", "Unexpected LOO retention/update settings")
        else:
            require(self.actor.dot_threshold == 0, "Unexpected DTV threshold")
        require(self.actor.config.checkpoint_root_directory is None
                and self.optimizer_steps() == self.actor.train_steps == self.cluster.global_steps == 0,
                "Expected a fresh audit without checkpoint restore or save")
        leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.state(self.actor.model, nnx.LoRAParam))
        require(bool(leaves), "Expected trainable LoRA parameters")
        fingerprint = hashlib.sha256()
        for path, value in leaves:
            array = np.asarray(jax.device_get(value))
            fingerprint.update(repr((path, array.shape, str(array.dtype))).encode())
            fingerprint.update(array.tobytes())
        self.cluster._grpo_filter_audit = self
        self.actor._filter_audit_enabled = True
        # Reuse the existing quiet aux transport, not the benchmark runner.
        self.actor._benchmark_quiet = True
        self.actor._benchmark_aux_sink = self.capture
        with mesh:
            learner.train(dataset, None)
        require(self.optimizer_steps() == len(self.batches) == self.steps,
                "Run did not finish the requested optimizer updates")
        with (self.out / "decisions.jsonl").open("x", encoding="utf-8") as stream:
            for entry in self.batches:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        write_json(self.out / "run.json", {
            "schema": SCHEMA, "scope": "early_training", "method": self.method,
            "condition": self.condition, "seed": self.seed, "noise_fraction": self.fraction,
            "noise_seed": self.seed, "shuffle_seed": 42 + self.seed,
            "optimizer_updates": self.steps, "schedule_updates": SCHEDULE_STEPS,
            "model_revision": Path(model_path).name, "jax_version": jax.__version__,
            "initial_lora_sha256": fingerprint.hexdigest(),
            "devices": [d.device_kind for d in mesh.devices.flat],
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        })
        print(f"PASS {self.condition}/{self.method}/seed{self.seed}: {self.steps} optimizer updates", flush=True)


def validate_batch(entry, step, condition, method):
    rows = entry["rows"]
    require(type(entry["step"]) is int and entry["step"] == step and len(rows) == BATCH_SIZE,
            f"Step {step}: missing, repeated or malformed update")
    require(all(type(r[k]) is bool for r in rows for k in ("keep", "noise_selected", "noise_effective"))
            and all(isinstance(r["advantage"], (int, float)) and not isinstance(r["advantage"], bool)
                    and math.isfinite(r["advantage"]) and HASH.fullmatch(r["prompt_id"]) for r in rows),
            f"Step {step}: invalid completion fields")
    require(entry["skipped_samples"] == sum(not r["keep"] for r in rows),
            f"Step {step}: mask and skipped_samples disagree")
    require(all((r["group_index"], r["completion_index"]) == divmod(i, GROUP_SIZE)
                for i, r in enumerate(rows)), f"Step {step}: duplicate or reordered completion")
    for start in range(0, BATCH_SIZE, GROUP_SIZE):
        group = rows[start:start + GROUP_SIZE]
        require(len({(r["prompt_id"], r["noise_selected"], r["noise_effective"]) for r in group}) == 1
                and not (group[0]["noise_effective"] and not group[0]["noise_selected"])
                and not (condition == "clean" and group[0]["noise_selected"])
                and (method != "dtv-loo" or any(r["keep"] for r in group)),
                f"Step {step}: invalid group labels or LOO minimum retention")
    return rows


def read_suite(root, smoke=False):
    """Read exactly the requested matrix; do not pool incomplete or duplicate runs."""
    steps = 1 if smoke else FORMAL_STEPS
    seeds = SEEDS[:1] if smoke else SEEDS
    runs = {}
    for run in sorted(root.glob("*/run.json")):
        meta = load_json(run)
        method, condition, seed = (meta[k] for k in ("method", "condition", "seed"))
        key = condition, method, seed
        require(key not in runs and method in METHODS and condition in CONDITIONS and seed in seeds
                and meta["schema"] == SCHEMA and meta["scope"] == "early_training"
                and meta["optimizer_updates"] == steps and meta["schedule_updates"] == SCHEDULE_STEPS
                and meta["noise_fraction"] == CONDITIONS[condition] and meta["noise_seed"] == seed
                and meta["shuffle_seed"] == 42 + seed,
                f"{run}: incompatible or duplicate run")
        with run.with_name("decisions.jsonl").open(encoding="utf-8") as stream:
            batches = [validate_batch(json.loads(line), i, condition, method) for i, line in enumerate(stream, 1)]
        require(len(batches) == steps, f"{run}: incomplete decision log")
        runs[key] = meta, [row for batch in batches for row in batch]
    expected = {(c, m, s) for c in CONDITIONS for m in METHODS for s in seeds}
    require(runs.keys() == expected, f"Incomplete audit matrix; missing {sorted(expected - runs.keys())}")
    first_meta = next(iter(runs.values()))[0]
    for meta, _ in runs.values():
        require(all(meta[k] == first_meta[k] for k in ("commit", "model_revision", "jax_version", "devices")),
                "Mixed code/model/runtime across the audit matrix")
    for seed in seeds:
        reference_meta, reference = runs["clean", "dtv", seed]
        for condition in CONDITIONS:
            other = [runs[condition, method, seed] for method in METHODS]
            for meta, rows in other:
                require(meta["initial_lora_sha256"] == reference_meta["initial_lora_sha256"],
                        f"seed{seed}: initial LoRA parameters differ")
                require([r["prompt_id"] for r in rows] == [r["prompt_id"] for r in reference],
                        f"seed{seed}: prompt order differs across methods or conditions")
            require([r["noise_selected"] for r in other[0][1]] == [r["noise_selected"] for r in other[1][1]],
                    f"seed{seed}: reversal assignment differs across methods")
    return runs


def summarize(rows):
    signs = ("positive", "negative", "zero")
    groups = ("all", "unassigned", "assigned", "effective", "assigned_no_change")
    counts = defaultdict(Counter)
    for row in rows:
        a = row["advantage"]
        sign = "positive" if a > 0 else "negative" if a < 0 else "zero"
        labels = ["all", "assigned" if row["noise_selected"] else "unassigned"]
        if row["noise_selected"]:
            labels.append("effective" if row["noise_effective"] else "assigned_no_change")
        for label in labels:
            for subset in ("all", sign):
                counts[label, subset].update(total=1, dropped=int(not row["keep"]))
    return [
        {"group": g, "advantage": a, "dropped": counts[g, a]["dropped"],
         "total": (total := counts[g, a]["total"]),
         "drop_rate": counts[g, a]["dropped"] / total if total else None}
        for g in groups for a in ("all", *signs)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print pooled and per-seed counts")
    args = parser.parse_args()
    runs = read_suite(args.root, args.smoke)
    steps = 1 if args.smoke else FORMAL_STEPS
    seeds = SEEDS[:1] if args.smoke else SEEDS
    if args.json:
        output = {"scope": "early_training", "optimizer_updates_per_run": steps, "results": []}
    else:
        print(f"Early-training filtering audit: first {steps} update(s) per run; seeds {seeds}")
        print("| Condition | Method | Prompt groups | All | A>0 | A<0 | A=0 |")
        print("|---|---|---|---:|---:|---:|---:|")
    for c in CONDITIONS:
        for m in METHODS:
            by_seed = {s: runs[c, m, s][1] for s in seeds}
            pooled = summarize([r for rows in by_seed.values() for r in rows])
            if args.json:
                output["results"].append({
                    "condition": c, "method": m, "pooled": pooled,
                    "per_seed": {s: summarize(rows) for s, rows in by_seed.items()},
                })
                continue
            table = {(x["group"], x["advantage"]): x for x in pooled}
            for group in ("all",) if c == "clean" else ("unassigned", "assigned", "all"):
                cells = []
                for sign in ("all", "positive", "negative", "zero"):
                    row = table[group, sign]
                    cells.append(f'{row["dropped"]}/{row["total"]} ({row["drop_rate"]:.2%})'
                                 if row["total"] else "N/A")
                print("| " + " | ".join([c, m, group, *cells]) + " |")
    if args.json:
        print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
