"""TPU-only execution for the rebuttal utility audit; no trainer or optimizer.

JAX/Flax imports are lazy. Forward/AD use the repository GRPO loss. Per-unit
gradients follow its vmap(value_and_grad) path at the fixed 4x4 batch shape.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import time
import numpy as np

from .parameter_snapshot import ParameterSnapshot
from .utility_math import numerical_pair_check


def production_per_unit_gradient(loss):
    """Same differentiation order as the production per-sample trainer."""
    import jax
    from flax import nnx
    grad = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, nnx.LoRAParam))

    def calculate(model, example):
        axes = jax.tree.map(lambda value: None if value is None else 0, example,
                            is_leaf=lambda value: value is None)
        return jax.vmap(grad, in_axes=(None, axes))(model, example)
    return nnx.jit(calculate)


def production_mean_loss(loss):
    """Match h's per-unit forward shape when evaluating the clean objective."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    def calculate(model, example):
        axes = jax.tree.map(lambda value: None if value is None else 0, example,
                            is_leaf=lambda value: value is None)
        return jnp.mean(jax.vmap(loss, in_axes=(None, axes))(model, example))
    return nnx.jit(calculate)


def production_per_unit_logps(compute, pad_id, eos_id):
    """Measure the forward-shape risk without changing frozen old logps."""
    import jax
    from flax import nnx

    def single(model, example):
        return compute(model, example.prompt_ids[None, :], example.completion_ids[None, :],
                       pad_id, eos_id, stop_gradient=True).reshape(-1)

    def calculate(model, example):
        axes = jax.tree.map(lambda value: None if value is None else 0, example,
                            is_leaf=lambda value: value is None)
        return jax.vmap(single, in_axes=(None, axes))(model, example)
    return nnx.jit(calculate)


class UtilityRuntime:
    def __init__(self, args, checkpoint, output):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from orbax import checkpoint as ocp
        from tunix.generate import sampler, tokenizer_adapter
        from tunix.rl import common
        from tunix.rl.grpo.grpo_learner import GRPOConfig, TrainExample, grpo_loss_fn
        from tunix.rl.self_inf_loo_trainer import _group_loo_statistics, _batch_loo_statistics
        from my_example.audit_grpo_checkpoint_restore import (
            parameter_summary, suppress_monitoring_scalars,
        )
        from my_example.config import ModelConfig, LoraConfig
        from my_example.model import (
            download_model, resolve_model_config, load_model, apply_lora, load_eos_tokens,
        )
        from my_example.sharding import make_mesh

        self.args, self.output = args, output
        expected_loss_file = Path(__file__).resolve().parents[2] / "tunix/rl/grpo/grpo_learner.py"
        if Path(inspect.getfile(grpo_loss_fn)).resolve() != expected_loss_file:
            raise RuntimeError("tunix imports from a different checkout; fix the server editable install before auditing")
        self.jax, self.jnp, self.nnx, self.common = jax, jnp, nnx, common
        self.TrainExample = TrainExample
        self.parameter_summary = parameter_summary
        self.phases = []
        self.topology = {"process_count": jax.process_count(), "process_index": jax.process_index(),
                         "device_count": jax.device_count(), "local_device_count": jax.local_device_count(),
                         "devices": [str(device) for device in jax.local_devices()]}
        if self.topology["process_count"] != 1 or self.topology["device_count"] != 4 or \
                self.topology["local_device_count"] != 4 or any(
                    device.platform != "tpu" for device in jax.local_devices()):
            raise ValueError("requires one process and four LOCAL TPU devices, not multiworker/CPU")
        self.mesh, _ = make_mesh((4, 1))
        self.monitor = suppress_monitoring_scalars
        self.model_path = (str(Path(args.backbone_path).expanduser().resolve(strict=True))
                           if args.backbone_path else download_model(ModelConfig(model_id=args.model_id)))
        config = resolve_model_config(args.model_id)
        # Match production main: Qwix's dummy forward must be outside fsdp=4.
        with self.monitor(jax):
            self.base = load_model(self.model_path, config, self.mesh)
            self.actor = apply_lora(self.base, LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha), self.mesh)
        if jax.tree.leaves(nnx.state(self.base, nnx.LoRAParam)):
            raise ValueError("reference backbone unexpectedly contains LoRA parameters")
        with self.mesh, self.monitor(jax):
            target = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
                                  nnx.state(self.actor, nnx.LoRAParam))
            checkpointer = ocp.StandardCheckpointer()
            try:
                restored = checkpointer.restore(str(checkpoint / "model_params"), target=target)
            finally:
                checkpointer.close()
            nnx.update(self.actor, restored)
            self._snapshot = ParameterSnapshot(nnx.state(self.actor, nnx.LoRAParam))
            self.initial_summary = parameter_summary(jax, self.snapshot)
            if parameter_summary(jax, restored)["sha256"] != self.initial_summary["sha256"]:
                raise ValueError("actor parameters differ from restored checkpoint")
        self.tokenizer = tokenizer_adapter.Tokenizer(tokenizer_path=args.tokenizer_path)
        self.pad_id, self.eos_id = self.tokenizer.pad_id(), self.tokenizer.eos_id()
        self.eos_tokens = load_eos_tokens(self.model_path)
        if self.eos_id not in self.eos_tokens:
            self.eos_tokens.append(self.eos_id)
        self.sampler = sampler.Sampler(self.actor, self.tokenizer, sampler.CacheConfig(
            cache_size=args.prompt_length + args.completion_length + 256,
            num_layers=config.num_layers, num_kv_heads=config.num_kv_heads, head_dim=config.head_dim))
        self.algo = GRPOConfig(num_generations=4, num_iterations=1, beta=args.beta, epsilon=args.epsilon)
        policy_config = replace(self.algo, beta=0.)

        def policy_loss(model, example):
            return grpo_loss_fn(model, example, policy_config, self.pad_id, self.eos_id)[0]

        def total_loss(model, example):
            return grpo_loss_fn(model, example, self.algo, self.pad_id, self.eos_id)[0]

        self.policy_grad = production_per_unit_gradient(policy_loss)
        self.total_grad = production_per_unit_gradient(total_loss)
        # Do not evaluate F with a different full-batch forward graph than h:
        # per-unit/vmap versus batched AD differed materially in earlier audits.
        self.clean_loss = production_mean_loss(policy_loss)
        self.unit_logps = production_per_unit_logps(common.compute_per_token_logps, self.pad_id, self.eos_id)
        self._h_halves = None

        @jax.jit
        def average(tree, mask):
            denominator = jnp.maximum(mask.sum(), 1.)
            return jax.tree.map(lambda x: jnp.sum(
                x.astype(jnp.float32) * mask.reshape((-1,) + (1,) * (x.ndim - 1)), axis=0) / denominator, tree)
        self.average = average

        @jax.jit
        def add(tree, other):
            return jax.tree.map(lambda x, y: x.astype(jnp.float32) + y.astype(jnp.float32), tree, other)
        self.add = add

        @jax.jit
        def scale(tree, value):
            return jax.tree.map(lambda x: x.astype(jnp.float32) * value, tree)
        self.scale = scale

        @jax.jit
        def dot(a, b):
            return sum(jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32))
                       for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
        self.dot = dot

        @jax.jit
        def statistics(pg, full, h, h_a, h_b):
            pgl, ful, hl = jax.tree.leaves(pg), jax.tree.leaves(full), jax.tree.leaves(h)
            n = pgl[0].shape[0]
            gram = jnp.zeros((n, n), dtype=jnp.float32)
            u_pg = jnp.zeros(n, dtype=jnp.float32)
            u_full = jnp.zeros(n, dtype=jnp.float32)
            u_a, u_b = jnp.zeros(n, dtype=jnp.float32), jnp.zeros(n, dtype=jnp.float32)
            dtv = jnp.zeros(n, dtype=jnp.float32)
            width = 4 if args.scope == "group" else n
            reference_norm_squared = jnp.zeros(n // width, dtype=jnp.float32)
            finite = jnp.array(True)
            for p, f, target_gradient, a, b in zip(
                    pgl, ful, hl, jax.tree.leaves(h_a), jax.tree.leaves(h_b)):
                finite = finite & jnp.all(jnp.isfinite(p)) & jnp.all(jnp.isfinite(f))
                flat = p.astype(jnp.float32).reshape(n, -1)
                target_flat = target_gradient.astype(jnp.float32).reshape(-1)
                gram = gram + flat @ flat.T
                u_pg = u_pg + flat @ target_flat
                u_full = u_full + f.astype(jnp.float32).reshape(n, -1) @ target_flat
                u_a = u_a + flat @ a.astype(jnp.float32).reshape(-1)
                u_b = u_b + flat @ b.astype(jnp.float32).reshape(-1)
                reference = flat.reshape((n // width, width, -1)).mean(axis=1)
                reference_norm_squared = reference_norm_squared + jnp.sum(reference * reference, axis=1)
                # Preserve ordinary DTV's per-leaf reduction, not a hash or
                # cosine replacement. LOO below reuses the production helper.
                if args.scope == "group":
                    grouped = p.reshape((n // 4, 4) + p.shape[1:])
                    local = jnp.sum(grouped * jnp.mean(grouped, axis=1)[:, None, ...],
                                    axis=tuple(range(2, grouped.ndim))).reshape(n)
                else:
                    local = jnp.sum(p * p.mean(axis=0), axis=tuple(range(1, p.ndim)))
                dtv = dtv + local
            loo = (_group_loo_statistics(pg, 4) if args.scope == "group" else _batch_loo_statistics(pg))
            return {"gram": gram, "utility_policy": u_pg, "utility_full": u_full,
                    "utility_half_a": u_a, "utility_half_b": u_b,
                    "score_dtv": dtv, "score_loo": loo["loo_score"],
                    "self_term": loo["standard_self_term"], "cross_term": loo["standard_cross_term"],
                    "reference_norm_squared": reference_norm_squared, "gradients_finite": finite}
        self.statistics = statistics

        @jax.jit
        def perturb(initial, direction, eta, maximum_norm):
            norm = jnp.sqrt(dot(direction, direction))
            clip = jnp.where(maximum_norm > 0, jnp.minimum(1., maximum_norm / jnp.maximum(norm, 1e-30)), 1.)
            changed = jax.tree.map(lambda p, g: (p.astype(jnp.float32) - eta * clip * g).astype(p.dtype),
                                   initial, direction)
            delta = jax.tree.map(lambda p, q: q.astype(jnp.float32) - p.astype(jnp.float32), initial, changed)
            count = sum(jnp.count_nonzero(x) for x in jax.tree.leaves(delta))
            return changed, delta, norm, clip, count
        self.perturb = perturb
        self.snapshot_memory("restored_model")

    @property
    def snapshot(self):
        # nnx.update may mutate/alias State Variable wrappers. Return a NEW
        # wrapper tree each time; the authoritative copied array tuple is private.
        return self._snapshot.materialize()

    def assert_original_parameters(self, stage):
        with self.mesh, self.monitor(self.jax):
            current = self.nnx.state(self.actor, self.nnx.LoRAParam)
            equal = self._snapshot.matches(current)
        if not equal:
            from my_example.audit_grpo_checkpoint_restore import write_report
            details = self._snapshot.differences(current)
            details.update(stage=stage, criterion="same parameter paths, shapes, dtypes and exact bits")
            write_report(self.output / "parameter_restore_failure.json", details)
            raise RuntimeError(f"parameters differ from immutable theta0 at {stage}; see parameter_restore_failure.json")
        return {"stage": stage, "bitwise_equal": True,
                "snapshot_storage": "detached_array_values_no_live_NNX_variables"}

    def roundtrip_check(self):
        """Fast actual-checkpoint mutation/restore test, no rollout or backward."""
        results = []
        self.assert_original_parameters("roundtrip_start")
        for shift in (1e-4, -2e-4):
            try:
                with self.mesh, self.monitor(self.jax):
                    temporary = self.jax.tree.map(
                        lambda leaf: (leaf.astype(self.jnp.float32) + shift).astype(leaf.dtype), self.snapshot)
                    self.jax.block_until_ready(temporary)
                    self.nnx.update(self.actor, temporary)
                    if self._snapshot.matches(self.nnx.state(self.actor, self.nnx.LoRAParam)):
                        raise RuntimeError("roundtrip test did not change any parameter bits")
                    # The backup must remain equal to its original host hash,
                    # even WHILE the model holds different parameters.
                    backup = self.parameter_summary(self.jax, self.snapshot)
                    if backup["sha256"] != self.initial_summary["sha256"]:
                        raise RuntimeError("immutable backup changed during model mutation")
            finally:
                with self.mesh, self.monitor(self.jax):
                    self.nnx.update(self.actor, self.snapshot)
                results.append(self.assert_original_parameters("roundtrip_restore"))
        self.snapshot_memory("parameter_roundtrip_only_no_gradients")
        return {"passed": True, "roundtrips": len(results), "backup_bytes": int(self._snapshot.bytes),
                "bitwise_restoration": True, "utility_audit_completed": False,
                "scope": "parameter mutation/restoration only, NOT utility or gradient HBM"}

    def snapshot_memory(self, name):
        self.jax.effects_barrier()
        records, headroom = [], []
        for device in self.jax.local_devices():
            try:
                stats = device.memory_stats()
            except (NotImplementedError, RuntimeError):
                stats = None
            stats = None if stats is None else {key: int(value) for key, value in stats.items()}
            records.append({"device": str(device), "stats": stats})
            if stats and stats.get("bytes_limit") and "peak_bytes_in_use" in stats:
                headroom.append(1 - stats["peak_bytes_in_use"] / stats["bytes_limit"])
        self.phases.append({"name": name, "devices": records, "peak_headroom_fractions": headroom})
        from my_example.audit_grpo_checkpoint_restore import write_report
        write_report(self.output / "hbm_report.json", {
            "phases": self.phases, "topology": self.topology,
            "headroom_verified": all(len(p["peak_headroom_fractions"]) == 4 for p in self.phases),
            "scope": "observed audit execution peaks; not a guarantee for arbitrary inputs",
        })
        if headroom and min(headroom) < self.args.min_headroom_fraction:
            raise RuntimeError("audit peak HBM headroom below requested limit; no silent fallback")

    def prepare(self, examples, *, role, batch_id):
        from my_example import rewards as reward_lib
        from my_example.reward_rank_noise import RewardRankNoiseConfig, apply_reward_rank_noise
        from tunix.rl.grpo.grpo_learner import compute_advantages
        if len(examples) != 4:
            raise ValueError("every forward/backward batch must retain four prompts x four rollouts")
        prompts = [str(item["prompts"]) for item in examples for _ in range(4)]
        answers = [str(item["answer"]) for item in examples for _ in range(4)]
        seed = int(np.random.SeedSequence([self.args.rollout_seed, 0 if role == "candidate" else 1,
                                          batch_id]).generate_state(1)[0]) & 0x7fffffff
        with self.mesh, self.monitor(self.jax):
            output = self.sampler(input_strings=prompts, max_generation_steps=self.args.completion_length,
                                  max_prompt_length=self.args.prompt_length, temperature=self.args.temperature,
                                  top_p=self.args.top_p, top_k=self.args.top_k, seed=int(seed),
                                  echo=False, return_logits=False, pad_output=True, eos_tokens=self.eos_tokens)
        if self.args.reward_mode == "accuracy":
            functions = [reward_lib.format_integrity_reward, reward_lib.numeric_relative_reward,
                         reward_lib.check_numbers]
        else:
            functions = [reward_lib.match_format_exactly, reward_lib.match_format_approximately,
                         reward_lib.check_answer, reward_lib.check_numbers]
        clean = sum(np.asarray(fn(prompts=prompts, completions=output.text, answer=answers), dtype=np.float64)
                    for fn in functions)
        noise = apply_reward_rank_noise(clean, prompts, 4, RewardRankNoiseConfig(
            self.args.candidate_noise if role == "candidate" else 0., self.args.noise_seed))
        advantages = compute_advantages(noise.corrupted_rewards, 4)
        correctness = []
        for text, answer in zip(output.text, answers):
            match = reward_lib.MATCH_NUMBERS.search(text)
            try:
                correctness.append(bool(match is not None and float(match.group(1).strip()) == float(answer.strip())))
            except (TypeError, ValueError):
                correctness.append(False)
        if role == "heldout" and self.args.heldout_reward_mode == "binary":
            clean = np.asarray(correctness, dtype=np.float64)
            noise = apply_reward_rank_noise(clean, prompts, 4, RewardRankNoiseConfig(0., self.args.noise_seed))
            advantages = compute_advantages(noise.corrupted_rewards, 4)
        ids = np.asarray(output.tokens, dtype=np.int32)
        mask = self.common.np_make_completion_mask(ids, eos_tok=self.eos_id) * (ids != self.pad_id)
        host = {"prompt_ids": np.asarray(output.padded_prompt_tokens, dtype=np.int32),
                "completion_ids": ids, "completion_mask": mask,
                "advantages": np.asarray(advantages, dtype=np.float32),
                "clean_rewards": clean, "observed_rewards": noise.corrupted_rewards,
                "noise_selected": noise.selected_groups, "noise_effective": noise.effective_groups,
                "correct": np.asarray(correctness), "text": np.asarray(output.text),
                "answers": np.asarray(answers), "prompt_ids_hash": np.asarray([
                    hashlib.sha256(text.encode()).hexdigest()[:16] for text in prompts]),
                "prompt_strings": np.asarray(prompts),
                "source_indices": np.repeat([item["source_index"] for item in examples], 4),
                "rollout_seed": np.asarray(seed, dtype=np.uint32)}
        example = self.as_example(host)
        with self.mesh, self.monitor(self.jax):
            ref = self.common.compute_per_token_logps(self.base, example.prompt_ids, example.completion_ids,
                                                     self.pad_id, self.eos_id, stop_gradient=True)
            host["ref_per_token_logps"] = np.asarray(self.jax.device_get(ref))
            del ref
            old = self.common.compute_per_token_logps(self.actor, example.prompt_ids, example.completion_ids,
                                                     self.pad_id, self.eos_id, stop_gradient=True)
            host["old_per_token_logps"] = np.asarray(self.jax.device_get(old))
        if not all(np.isfinite(host[key]).all() for key in (
            "ref_per_token_logps", "old_per_token_logps", "advantages", "observed_rewards")):
            raise FloatingPointError("nonfinite rollout/reward/log probability")
        self.save_rollout(host, role=role, batch_id=batch_id)
        self.snapshot_memory(f"{role}_rollout_{batch_id}")
        return host

    def save_rollout(self, host, *, role, batch_id):
        serializable = dict(host)
        for name in ("ref_per_token_logps", "old_per_token_logps"):
            # NumPy archives cannot portably reconstruct ml_dtypes.bfloat16.
            # Save its exact representable values as float32 plus original dtype;
            # keep the ORIGINAL dtype in RAM for this audit's production loss.
            serializable[name + "_original_dtype"] = np.asarray(str(host[name].dtype))
            serializable[name] = host[name].astype(np.float32)
        np.savez_compressed(self.output / "fixed_rollouts" / f"{role}_{batch_id:03d}.npz", **serializable)

    def replay_prepare(self, path, indices, *, role, batch_id, original_args):
        from .utility_data import load_rollout
        from my_example.reward_rank_noise import RewardRankNoiseConfig, apply_reward_rank_noise
        from tunix.rl.grpo.grpo_learner import compute_advantages

        host = load_rollout(path, indices)
        for name in ("old_per_token_logps", "ref_per_token_logps"):
            dtype = str(host.get(name + "_original_dtype", "float32"))
            if dtype not in {"float32", "float16", "bfloat16"}:
                raise ValueError(f"unsupported replay log-probability dtype {dtype}")
            host[name] = np.asarray(self.jax.device_get(self.jnp.asarray(host[name], dtype=getattr(self.jnp, dtype))))
        if "prompt_strings" in host:
            prompts = host["prompt_strings"].tolist()
            if len(prompts) != 16 or any(hashlib.sha256(p.encode()).hexdigest()[:16] != identity
                                         for p, identity in zip(prompts, host["prompt_ids_hash"])):
                raise ValueError("replay prompt text/hash mismatch")
            noise = apply_reward_rank_noise(host["clean_rewards"], prompts, 4, RewardRankNoiseConfig(
                self.args.candidate_noise if role == "candidate" else 0., self.args.noise_seed))
            host.update(observed_rewards=noise.corrupted_rewards, noise_selected=noise.selected_groups,
                        noise_effective=noise.effective_groups)
            host["advantages"] = compute_advantages(noise.corrupted_rewards, 4).astype(np.float32)
        else:
            # v1 archives did not save exact prompt strings. They can be used
            # for SAME-condition numerical debugging, not new corruption hashes.
            if role == "candidate" and any(getattr(self.args, name) != original_args.get(name)
                                            for name in ("candidate_noise", "noise_seed")):
                raise ValueError("legacy replay lacks prompt strings: cannot change candidate noise/seed")
            expected = compute_advantages(host["observed_rewards"], 4).astype(np.float32)
            if not np.array_equal(expected, host["advantages"]):
                raise ValueError("legacy replay advantages do not match recorded observed rewards")
        if role == "heldout" and (host["noise_selected"].any() or host["noise_effective"].any() or
                                   not np.array_equal(host["clean_rewards"], host["observed_rewards"])):
            raise ValueError("heldout replay contains corrupted rewards")
        self.save_rollout(host, role=role, batch_id=batch_id)
        self.snapshot_memory(f"{role}_replay_{batch_id}")
        print(f"Replayed {role} batch {batch_id}: no generation", flush=True)
        return host

    def as_example(self, host):
        jp = self.jnp
        return self.TrainExample(
            prompt_ids=jp.asarray(host["prompt_ids"]),
            prompt_mask=jp.asarray(host["prompt_ids"] != self.pad_id),
            completion_ids=jp.asarray(host["completion_ids"]),
            completion_mask=jp.asarray(host["completion_mask"]),
            advantages=jp.asarray(host["advantages"]),
            ref_per_token_logps=(jp.asarray(host["ref_per_token_logps"])
                                 if "ref_per_token_logps" in host else None),
            old_per_token_logps=(jp.asarray(host["old_per_token_logps"])
                                 if "old_per_token_logps" in host else None),
            filter_random_values=None)

    def gradients(self, host, *, full=False):
        with self.mesh, self.monitor(self.jax):
            values, tree = (self.total_grad if full else self.policy_grad)(self.actor, self.as_example(host))
            self.jax.block_until_ready((values, tree))
        return tree

    def clean_value(self, heldout):
        with self.mesh, self.monitor(self.jax):
            values = [float(self.jax.device_get(self.clean_loss(self.actor, self.as_example(host))))
                      for host in heldout]
        result = float(np.mean(values))
        if not np.isfinite(result):
            raise FloatingPointError("independent clean objective is nonfinite")
        return result

    def heldout_gradient(self, heldout):
        sums, counts = [None, None], [0, 0]
        zeros = 0
        for index, host in enumerate(heldout):
            tree = self.gradients(host)
            with self.mesh:
                mean = self.average(tree, self.jnp.ones(16, dtype=self.jnp.float32))
                half = index % 2  # alternate whole prompt batches, never split responses
                sums[half] = mean if sums[half] is None else self.add(sums[half], mean)
                counts[half] += 1
                self.jax.block_until_ready(sums[half])
            zeros += int(np.count_nonzero(host["advantages"] == 0))
            del tree
            self.snapshot_memory(f"heldout_backward_{index}")
            print(f"Heldout gradients: {(index + 1) * 4}/{len(heldout) * 4} prompts", flush=True)
        with self.mesh:
            h = self.scale(sums[0] if sums[1] is None else self.add(sums[0], sums[1]), 1. / len(heldout))
            norm = float(self.jax.device_get(self.dot(h, h))) ** .5
            halves = [self.scale(s, 1. / count) if count else self.scale(h, 0.)
                      for s, count in zip(sums, counts)]
            norms = [float(self.jax.device_get(self.dot(s, s))) ** .5 for s in halves]
            cosine = (float(self.jax.device_get(self.dot(*halves))) / np.prod(norms)
                      if min(counts) > 0 and min(norms) > self.args.zero_epsilon else None)
            self._h_halves = tuple(halves)
        if not np.isfinite(norm):
            raise FloatingPointError("heldout gradient is nonfinite")
        self.snapshot_memory("heldout_split_means")
        return h, {"l2_norm": norm, "zero_advantage_units": zeros,
                   "units": len(heldout) * 16, "informative": norm > self.args.zero_epsilon,
                   "informative_definition": "nonzero norm ONLY, not a reliability guarantee",
                   "split_half": {"available": min(counts) > 0, "prompts": [n * 4 for n in counts],
                                  "l2_norms": norms, "cosine": float(np.clip(cosine, -1., 1.)) if cosine is not None else None,
                                  "positive_alignment": bool(cosine is not None and cosine > 0),
                                  "definition": "alternating whole four-prompt batches; sensitivity, NOT proof"}}

    def batch_statistics(self, pg, full, h):
        with self.mesh:
            if self._h_halves is None:
                raise RuntimeError("compute heldout gradients before candidate statistics")
            result = self.jax.device_get(self.statistics(pg, full, h, *self._h_halves))
        if not bool(result["gradients_finite"]) or not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("nonfinite exact-gradient statistics")
        return result

    def direction(self, pg, full, mask):
        with self.mesh:
            weights = self.jnp.asarray(mask, dtype=self.jnp.float32)
            direction = self.average(full, weights)
            if self.args.mask_application == "policy-only":
                direction = self.add(self.average(pg, weights), self.add(
                    self.average(full, self.jnp.ones(16)), self.scale(self.average(pg, self.jnp.ones(16)), -1.)))
            self.jax.block_until_ready(direction)
        return direction

    def probe(self, direction, h, heldout, baseline, eta):
        jp, jax = self.jnp, self.jax
        started = time.monotonic()
        self.assert_original_parameters("before_probe")
        try:
            with self.mesh, self.monitor(jax):
                params, delta, raw_norm, clip, changed = self.perturb(
                    self.snapshot, direction, jp.asarray(eta), jp.asarray(self.args.probe_max_grad_norm))
                jax.block_until_ready(params)
                self.nnx.update(self.actor, params)
                changed = int(jax.device_get(changed))
                delta_norm = float(jax.device_get(self.dot(delta, delta))) ** .5
                predicted = -float(jax.device_get(self.dot(h, delta)))
            after = self.clean_value(heldout)
            self.snapshot_memory("finite_step_evaluation")
            return {"clean_loss_before": baseline, "clean_loss_after": after,
                    "utility": baseline - after, "eta": eta,
                    "unclipped_direction_norm": float(jax.device_get(raw_norm)),
                    "gradient_clip_factor": float(jax.device_get(clip)),
                    "parameter_delta_norm": delta_norm, "changed_parameter_elements": changed,
                    "parameter_delta_informative": changed > 0,
                    "predicted_first_order_utility": predicted,
                    "elapsed_seconds": time.monotonic() - started}
        finally:
            # Restore from detached ARRAY VALUES, not a State sharing model
            # Variables. Check exact bits; numerical allclose is insufficient.
            with self.mesh, self.monitor(jax):
                self.nnx.update(self.actor, self.snapshot)
            self.assert_original_parameters("after_probe_restore")

    def numerical_preflight(self, heldout, h, baseline, baseline_repeat):
        """Bounded same-rollout +/- self-gradient test, independent of selectors.

        No precision/batch/old-policy rewrite or outcome-dependent step search.
        A failed gate is a valid diagnostic outcome, not evidence for DTV.
        """
        from my_example.audit_grpo_checkpoint_restore import write_report

        args = self.args
        floor = max(abs(baseline - baseline_repeat),
                    float(np.finfo(np.float32).eps) * max(1., abs(baseline)))
        report = {"passed": False, "validated_step_sizes": [], "baseline": baseline,
                  "same_parameter_repeat_difference": abs(baseline - baseline_repeat),
                  "minimum_resolution_floor": floor, "pairs": [],
                  "rtol": args.numerical_rtol, "atol": args.numerical_atol,
                  "purpose": "measurement validation, NOT method outcome selection",
                  "forward_contract": []}
        # Only the first heldout batch: diagnose full16 vs per-unit forward at
        # identical theta0. Never replace old logps to force this test to pass.
        host = heldout[0]
        with self.mesh, self.monitor(self.jax):
            raw_values = self.jax.device_get(self.unit_logps(self.actor, self.as_example(host)))
            values = np.asarray(raw_values, dtype=np.float32)
        old = host["old_per_token_logps"].astype(np.float32)
        valid = host["completion_mask"].astype(bool)
        difference = (values - old)[valid]
        if not difference.size or not np.isfinite(difference).all():
            raise ValueError("numerical check needs finite valid heldout completion logps")
        report["forward_contract"].append({
            "batch_id": 0, "valid_tokens": int(difference.size), "old_dtype": str(host["old_per_token_logps"].dtype),
            "unit_forward_dtype": str(raw_values.dtype),
            "full16_vs_unit_logps_max_abs": float(np.max(np.abs(difference))),
            "full16_vs_unit_logps_mean_abs": float(np.mean(np.abs(difference))),
            "nonidentical_logps_fraction": float(np.mean(difference != 0)),
            "importance_ratio_clipped_fraction_at_theta0": float(np.mean(
                (difference < np.log(1. - args.epsilon)) | (difference > np.log(1. + args.epsilon)))),
            "note": "reported, not silently recalibrated; can expose mixed-precision/shape differences"})
        zero = self.probe(h, h, heldout, baseline, 0.)
        restored = self.clean_value(heldout)
        report["zero_step"] = zero
        report["zero_step_restored_loss_difference"] = abs(restored - baseline)
        noop_ok = (zero["changed_parameter_elements"] == 0 and abs(zero["utility"]) <= args.numerical_atol
                   and abs(restored - baseline) <= args.numerical_atol)
        report["zero_step_passed"] = noop_ok
        write_report(self.output / "numerical_report.json", report)
        for eta in args.probe_step_sizes:
            print(f"Numerical control +/- eta={eta}: heldout SELF gradient, no filtering", flush=True)
            plus = self.probe(h, h, heldout, baseline, eta)
            restore_a = abs(self.clean_value(heldout) - baseline)
            minus = self.probe(h, h, heldout, baseline, -eta)
            restore_b = abs(self.clean_value(heldout) - baseline)
            check = numerical_pair_check(plus, minus, floor=floor, rtol=args.numerical_rtol, atol=args.numerical_atol)
            check["restored_loss_max_difference"] = max(restore_a, restore_b)
            check["passed"] = bool(check["passed"] and noop_ok and
                                   max(restore_a, restore_b) <= args.numerical_atol)
            report["pairs"].append({"eta": eta, "positive": plus, "negative": minus, "check": check})
            if check["passed"]:
                report["validated_step_sizes"].append(eta)
            report["passed"] = bool(report["validated_step_sizes"])
            write_report(self.output / "numerical_report.json", report)
        self.assert_original_parameters("numerical_preflight_end")
        self.snapshot_memory("numerical_preflight")
        return report

    def stress(self, heldout, h, baseline):
        """Full configured length, all-valid synthetic masks; never scientific data."""
        n, p, c = 16, self.args.prompt_length, self.args.completion_length
        valid = np.array([i for i in range(10, 100) if i not in self.eos_tokens and i != self.pad_id])
        host = {"prompt_ids": np.tile(valid[:1], (n, p)).astype(np.int32),
                "completion_ids": valid[(np.arange(c)[None, :] + np.arange(n)[:, None]) % len(valid)].astype(np.int32),
                "completion_mask": np.ones((n, c), dtype=np.int32),
                "advantages": np.tile([-1., -.3, .3, 1.], 4).astype(np.float32)}
        host["completion_ids"][:, -1] = self.eos_id
        example = self.as_example(host)
        with self.mesh, self.monitor(self.jax):
            host["ref_per_token_logps"] = np.asarray(self.jax.device_get(self.common.compute_per_token_logps(
                self.base, example.prompt_ids, example.completion_ids, self.pad_id, self.eos_id)))
            host["old_per_token_logps"] = np.asarray(self.jax.device_get(self.common.compute_per_token_logps(
                self.actor, example.prompt_ids, example.completion_ids, self.pad_id, self.eos_id)))
        full = self.gradients(host, full=True)
        self.snapshot_memory("full_length_total_gradients")
        pg = self.gradients(host)
        stress_stats = self.batch_statistics(pg, full, h)
        if not np.any(np.diag(stress_stats["gram"]) > self.args.zero_epsilon):
            raise RuntimeError("synthetic HBM stress did not exercise nonzero policy gradients")
        self.snapshot_memory("full_length_policy_and_exact_gram")
        direction = self.direction(pg, full, np.ones(16))
        self.probe(direction, h, heldout, baseline, self.args.probe_step_sizes[0])
        self.snapshot_memory("full_length_probe_and_restore")
