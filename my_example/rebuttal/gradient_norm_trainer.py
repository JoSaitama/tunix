"""Reviewer baseline: per-prompt policy-gradient norm top50%, full update.

Same per-completion AD order and trainable space as PolicySelfInfTrainer.
No gradient projection, new rollout, reward rewrite, advantage recomputation,
precision promotion, dynamic batch shrinking, or optimizer/schedule change.
This is top50% with deterministic ties, NOT a strict >median threshold.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import trainer as rl_trainer


def configuration_from_environment(environ):
    if environ.get("TUNIX_DYNAMIC_SAMPLING", "") == "1" or environ.get("TUNIX_PODS_KEEP_COUNT", "").strip():
        raise ValueError("Gradient-norm top50 cannot be combined with DS or PODS")
    if environ.get("TUNIX_GRPO_NUM_GENERATIONS", "4") != "4":
        raise ValueError("This frozen norm baseline requires four generations per prompt")
    return {"num_generations": 4, "artifact_dir": environ.get("TUNIX_NORM_ARTIFACT_DIR")}


def _reject_nonfinite(values):
    def fail(_):
        def raise_error():
            raise ValueError("nonfinite gradient/loss in gradient-norm top50; no silent filtering")
        jax.debug.callback(raise_error)
        return jnp.int32(0)
    return jax.lax.cond(jnp.all(jnp.isfinite(values)), lambda _: jnp.int32(0), fail, None)


def squared_gradient_norms(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves or leaves[0].ndim < 1:
        raise ValueError("expected per-completion gradient arrays")
    n = leaves[0].shape[0]
    if not n or any(x.ndim < 1 or x.shape[0] != n for x in leaves):
        raise ValueError("inconsistent gradient batch dimensions")
    result = jnp.zeros(n, dtype=jnp.float32)
    for leaf in leaves:
        flat = leaf.astype(jnp.float32).reshape(n, -1)
        result = result + jnp.sum(flat * flat, axis=1)
    _reject_nonfinite(result)
    return result


def group_top50_mask(norm_squared, num_generations=4):
    if num_generations != 4 or norm_squared.ndim != 1 or norm_squared.size == 0 or norm_squared.size % 4:
        raise ValueError("norm top50 requires contiguous complete four-completion groups")
    _reject_nonfinite(norm_squared)
    values = norm_squared.reshape(-1, 4)
    # Stable descending ranking: ties, including four zeros, keep local0/1.
    order = jnp.argsort(-values, axis=1, stable=True)
    ranks = jnp.argsort(order, axis=1, stable=True)
    return (ranks < 2).reshape(-1)


def masked_mean(tree, mask):
    denominator = jnp.clip(jnp.sum(mask), 1.)
    return jax.tree.map(lambda x: jnp.sum(
        x * mask.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0) / denominator, tree)


class GradientNormTop50Trainer(rl_trainer.Trainer):
    def __init__(self, *args, num_generations=4, artifact_dir=None, **kwargs):
        if num_generations != 4:
            raise ValueError("frozen gradient-norm baseline requires four completions")
        self.num_generations = num_generations
        self._score_loss_fn, self._score_loss_has_aux = None, False
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._norm_seen = self._norm_kept = self._norm_logged_updates = 0
        super().__init__(*args, **kwargs)
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "method": "gradient_norm_top50", "scope": "group", "num_generations": 4,
                "keep_per_prompt": 2, "retained_fraction": .5,
                "score_objective": "KL-free policy loss, as DTV policy-score",
                "parameter_space": "LoRAParam" if self._lora_enabled else "Param",
                "update_gradient": "original full GRPO gradient including configured KL",
                "normalization": "mean of selected full gradients",
                "advantages": "unchanged original four-answer group advantages",
                "ties": "stable original completion index; four zeros keep indices0/1",
                "precision": "original model/AD; only squared-norm reduction in FP32",
                "batch_shape": "unchanged, full16-answer backwards; no gathered smaller batch",
                "max_steps": self.config.max_steps, "argv": sys.argv,
                "experiment_seed": os.environ.get("TUNIX_EXPERIMENT_SEED"),
                "mismatch_fraction": os.environ.get("TUNIX_REWARD_RANK_NOISE_FRACTION", "0"),
                "mismatch_seed": os.environ.get("TUNIX_REWARD_RANK_NOISE_SEED", "0"),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            (self.artifact_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Gradient-norm top50 enabled: policy score, full update, group4 keep2; original advantages")

    def with_policy_score_loss_fn(self, score_loss_fn, *, has_aux=False):
        # GRPOLearner already supplies beta=0 ONLY for attribution via this hook.
        self.clear_jit_cache()
        self._score_loss_fn, self._score_loss_has_aux = score_loss_fn, has_aux

    def _train_step(self, model, optimizer, inputs):
        if self._score_loss_fn is None:
            raise RuntimeError("gradient-norm top50 requires the GRPO policy-score loss hook")
        inputs = self.gen_model_input_fn(inputs)
        wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
        axes = jax.tree.map(lambda _: 0, inputs)

        def per_unit(loss, has_aux):
            def calculate(model, item):
                return loss(model, **item) if isinstance(item, dict) else loss(model, item)
            return jax.vmap(nnx.value_and_grad(calculate, argnums=nnx.DiffState(0, wrt),
                                              has_aux=has_aux), in_axes=(None, axes))(model, inputs)

        output, full = per_unit(self.loss_fn, self._has_aux)
        losses, aux = output if self._has_aux else (output, None)
        score_output, policy = per_unit(self._score_loss_fn, self._score_loss_has_aux)
        score_losses = score_output[0] if self._score_loss_has_aux else score_output
        norm_squared = squared_gradient_norms(policy)
        finite = jnp.all(jnp.isfinite(losses)) & jnp.all(jnp.isfinite(score_losses))
        for leaf in jax.tree.leaves(full):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        _reject_nonfinite(jnp.where(finite, 0., jnp.nan))
        mask = group_top50_mask(norm_squared, self.num_generations)
        final_grads = masked_mean(full, mask)
        final_loss = jnp.sum(losses * mask) / jnp.clip(jnp.sum(mask), 1.)
        optimizer.update(model, final_grads)
        if not self._has_aux or getattr(self, "_benchmark_quiet", False):
            return final_loss, None
        final_aux = masked_mean(aux, mask)
        if not isinstance(final_aux, dict):
            raise TypeError("GRPO norm baseline requires dictionary loss aux")
        norms = jnp.sqrt(norm_squared)
        final_aux.update({"skipped_samples": jnp.sum(~mask),
                          "grad_norm_mean": jnp.mean(norms), "grad_norm_std": jnp.std(norms),
                          "norm_top50_kept_fraction": jnp.mean(mask.astype(jnp.float32)),
                          "norm_top50_zero_fraction": jnp.mean((norm_squared == 0).astype(jnp.float32)),
                          "norm_top50_kept_norm_mean": jnp.sum(norms * mask) / jnp.sum(mask),
                          "norm_top50_dropped_norm_mean": jnp.sum(norms * ~mask) / jnp.sum(~mask)})
        if self.artifact_dir:
            final_aux["_norm_values"], final_aux["_norm_mask"] = norms, mask
            final_aux["_norm_squared_values"] = norm_squared
            example = inputs.get("train_example") if isinstance(inputs, dict) else inputs
            if hasattr(example, "advantages"):
                final_aux["_norm_advantages"] = example.advantages
            if hasattr(example, "prompt_ids"):
                final_aux["_norm_prompt_tokens"] = example.prompt_ids.reshape(
                    (-1, 4) + example.prompt_ids.shape[1:])[:, 0]
        return final_loss, final_aux

    def _post_process_train_step(self, aux):
        if isinstance(aux, dict) and self.artifact_dir and "_norm_values" in aux:
            norms = np.asarray(jax.device_get(aux.pop("_norm_values"))).reshape(-1, 4)
            squared = np.asarray(jax.device_get(aux.pop("_norm_squared_values"))).reshape(-1, 4)
            mask = np.asarray(jax.device_get(aux.pop("_norm_mask")), dtype=bool).reshape(-1, 4)
            advantages = aux.pop("_norm_advantages", None)
            tokens = aux.pop("_norm_prompt_tokens", None)
            advantages = None if advantages is None else np.asarray(jax.device_get(advantages)).reshape(-1, 4)
            tokens = None if tokens is None else np.asarray(jax.device_get(tokens), dtype=np.int32)
            if not np.isfinite(norms).all() or not np.all(mask.sum(1) == 2):
                raise RuntimeError("invalid norm top50 selection report")
            self._norm_seen += mask.size
            self._norm_kept += int(mask.sum())
            self._norm_logged_updates += 1
            groups = []
            for index in range(len(mask)):
                groups.append({"group_index": index,
                    "prompt_token_hash": (hashlib.sha256(tokens[index].tobytes()).hexdigest()[:16]
                                          if tokens is not None else None),
                    "policy_gradient_l2_norms": norms[index].tolist(),
                    "policy_gradient_squared_l2_norms": squared[index].tolist(),
                    "selected_local_indices": np.flatnonzero(mask[index]).tolist(),
                    "advantages": advantages[index].tolist() if advantages is not None else None})
            record = {"train_step": int(self._train_steps) + 1, "method": "gradient_norm_top50",
                      "scope": "group", "score_objective": "policy", "mask_application": "full",
                      "advantages_recomputed": False, "groups": groups}
            with (self.artifact_dir / "norm_selection.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            summary = {"method": "gradient_norm_top50", "updates_logged": self._norm_logged_updates,
                       "observed_completions": self._norm_seen, "retained_completions": self._norm_kept,
                       "retained_fraction": self._norm_kept / self._norm_seen,
                       "counts_scope": "current process session; JSONL appends if resumed"}
            temporary = self.artifact_dir / "norm_summary.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.artifact_dir / "norm_summary.json")
        # The original learner's metric map stays untouched. Add only our keys.
        for key in ("norm_top50_kept_fraction", "norm_top50_zero_fraction",
                    "norm_top50_kept_norm_mean", "norm_top50_dropped_norm_mean"):
            self.rl_metrics_to_log.setdefault(key, np.mean)
        super()._post_process_train_step(aux)
