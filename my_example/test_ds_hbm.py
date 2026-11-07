"""Idle-worker DS preflight: real generation plus controlled-reward HBM stress.

Controlled rewards exist ONLY in this disposable test, including its real
generation phase. They guarantee a non-degenerate selector input independently
of sampled answer correctness/format. Production reward behavior is tested by
the separate clean/mismatch end-to-end smokes, NOT certified by this HBM test.
Two later updates additionally use synthetic full-length tokens. Every update
forces two refill rounds at fixed 4x4 shape, max prompt256/response768.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


def make_hbm_batch():
    """Use the real prompt template but do not depend on reward diversity."""
    from .prompts import SYSTEM_PROMPT, TEMPLATE

    return {
        "prompts": np.array([TEMPLATE.format(system_prompt=SYSTEM_PROMPT,
                            question=f"What is {i}+1?") for i in range(4)]),
        "answer": np.array([str(i+1) for i in range(4)]),
    }


def controlled_hbm_rewards(prompts, completions, **unused):
    """Exactly two eligible groups per fixed-size candidate batch, test-only.

    Even identical generated answers must not stall a memory stress fixture.
    Production rank reversal still runs AFTER this reward function and keeps
    the variance of each group invariant.
    """
    if len(prompts) != 16 or len(completions) != 16:
        raise ValueError("DS HBM fixture expects four prompts x four completions")
    return np.array([[0., .5, 3., 4.], [0., 0., 0., 0.],
                     [0., .5, 3., 4.], [1., 1., 1., 1.]]).reshape(-1)


def validate_hbm_selection(records, phases):
    """Certify that all three updates exercised selected AND rejected groups."""
    expected_names = ["real_generation_controlled_reward_update",
                      "full_length_forced_refill_update_1",
                      "full_length_forced_refill_update_2"]
    if [phase["name"] for phase in phases] != expected_names:
        raise RuntimeError("DS HBM phase labels do not match the executed test")
    if [phase["cumulative_generation_batches"] for phase in phases] != [2, 4, 6]:
        raise RuntimeError("DS HBM expected exactly two candidate batches per actor update")
    if len(records) != 24 or sum(row["selected"] for row in records) != 12:
        raise RuntimeError("DS HBM expected 24 generated / 12 selected prompt groups")
    by_update = {}
    for row in records:
        by_update.setdefault(row["update_step"], []).append(row)
    if len(by_update) != 3:
        raise RuntimeError("DS HBM expected records from three separate updates")
    for rows in by_update.values():
        if len(rows) != 8 or sum(row["selected"] for row in rows) != 4:
            raise RuntimeError("DS HBM expected eight generated / four selected groups per update")
        if sum(row["reason"] == "zero_variance" for row in rows) != 4:
            raise RuntimeError("DS HBM did not exercise rejection of constant-reward groups")
        if any(not row["eligible"] or row["observed_reward_std"] <= 0
               for row in rows if row["selected"]):
            raise RuntimeError("DS HBM selected a degenerate group")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--min-headroom-fraction", type=float, default=.05)
    args = parser.parse_args()
    if not 0 <= args.min_headroom_fraction < 1:
        parser.error("min headroom must be in [0,1)")
    os.environ.update({"TPU_PROCESS_BOUNDS":"1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS":"2,2,1",
                       "TPU_VISIBLE_CHIPS":"0,1,2,3", "CLOUD_TPU_TASK_ID":"0"})
    output = Path(args.report_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"passed":False, "phases":[], "headroom_verified":False,
              "scope":"real generation with controlled rewards; two full-length synthetic forced-refill updates",
              "reward_source":"controlled_hbm_rewards in ALL phases; not production correctness rewards",
              "caveat":"observed peak memory does not guarantee all future execution peaks"}

    def save():
        (output/"hbm_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")

    try:
        import jax
        import jax.numpy as jnp
        from tunix.generate import tokenizer_adapter
        from tunix.rl import rl_cluster as cluster_lib
        from .config import config_from_args
        from .deps import assert_dependencies
        from .model import apply_lora, download_model, load_eos_tokens, load_model, resolve_model_config
        from .sharding import make_mesh
        from .train import build_cluster_config, build_optimizer, build_trainer

        assert_dependencies()
        if (jax.process_count()!=1 or jax.device_count()!=4 or jax.local_device_count()!=4 or
            any(device.platform!="tpu" for device in jax.local_devices())):
            raise ValueError("DS HBM preflight requires one process / four local TPU devices")
        report["topology"]={"process_count":jax.process_count(),"process_index":jax.process_index(),
            "device_count":jax.device_count(),"local_device_count":jax.local_device_count(),
            "devices":[str(device) for device in jax.local_devices()]}
        cfg=config_from_args(["--num-generations","4","--train-micro-batch-size","4",
            "--learning-rate","1e-6","--max-prompt-length","256","--total-generation-steps","768",
            "--mesh-counts","4,1","--no-wandb","--metrics-log-dir",str(output/"tensorboard"),
            "--checkpoint-root",str(output/"checkpoints"),"--save-interval-steps","10000"])
        model_path=download_model(cfg.model)
        mesh,_=make_mesh(cfg.runtime.mesh_counts)
        base=load_model(model_path,resolve_model_config(cfg.model.model_id),mesh)
        actor=apply_lora(base,cfg.lora,mesh=mesh)
        tokenizer=tokenizer_adapter.Tokenizer(tokenizer_path=cfg.model.tokenizer_path)
        eos=load_eos_tokens(model_path)
        if tokenizer.eos_id() not in eos:
            eos.append(tokenizer.eos_id())
        cluster=cluster_lib.RLCluster(actor=actor,reference=base,tokenizer=tokenizer,
            cluster_config=build_cluster_config(mesh,cfg.grpo,cfg.eval,cfg.training,
                build_optimizer(cfg.training,691),691,4,eos,False))
        with mock.patch.dict(os.environ,{"TUNIX_DYNAMIC_SAMPLING":"1","TUNIX_PODS_KEEP_COUNT":"",
            "TUNIX_DS_MAX_GENERATION_BATCHES":"20","TUNIX_DS_ARTIFACT_DIR":str(output/"selection")}):
            trainer=build_trainer(cluster,cfg.grpo)
        # Stop after three real actor updates, but keep the frozen 691-step LR
        # schedule. This changes only this disposable test's stop condition.
        cluster.cluster_config.training_config.max_steps=3
        batch=make_hbm_batch()
        class Batches:
            def __iter__(self):
                yield batch
        original_generate, original_update = cluster.generate, cluster.update_actor
        stress={"active":False,"updates":0}
        valid=np.array([i for i in range(10,100) if i not in eos and i!=cluster.rollout.pad_id()])

        def generate(*args,**kwargs):
            if not stress["active"]:
                return original_generate(*args,**kwargs)
            size=len(kwargs["prompts"])
            tokens=valid[(np.arange(768)[None,:]+np.arange(size)[:,None])%len(valid)].astype(np.int32)
            tokens[:,-1]=tokenizer.eos_id()
            return SimpleNamespace(tokens=jnp.asarray(tokens),text=["stress"]*size,
                left_padded_prompt_tokens=jnp.full((size,256),int(valid[0]),dtype=jnp.int32))

        def update(*a,**k):
            result=original_update(*a,**k)
            jax.effects_barrier()
            stats,headrooms=[],[]
            for device in jax.local_devices():
                values=device.memory_stats()
                values=None if values is None else {key:int(value) for key,value in values.items()}
                stats.append({"device":str(device),"stats":values})
                if values and values.get("bytes_limit") and "peak_bytes_in_use" in values:
                    headrooms.append(1-values["peak_bytes_in_use"]/values["bytes_limit"])
            report["phases"].append({"name":"real_generation_controlled_reward_update" if not stress["active"]
                else f"full_length_forced_refill_update_{stress['updates']}",
                "devices":stats,"headroom":headrooms,
                "cumulative_generation_batches":trainer._ds_rounds})
            save()
            if len(headrooms)!=4:
                raise RuntimeError("DS HBM headroom unavailable on some local devices")
            if min(headrooms)<args.min_headroom_fraction:
                raise RuntimeError("DS HBM headroom below the requested safety margin")
            stress["updates"]+=1
            stress["active"]=True
            return result

        report["configuration"]={"prompts":4,"num_generations":4,"selected_groups_per_update":4,
            "prompt_length":256,"stress_completion_length":768,"beta":cfg.grpo.beta,
            "controlled_rewards_in_all_phases":True,"candidate_batches_per_update":2}
        # Install deterministic rewards BEFORE the first rollout. Installing
        # them only after its update creates a bootstrap failure on constant rewards:
        # DS cannot prepare that first update in the first place.
        with mock.patch.object(trainer,"reward_fns",[controlled_hbm_rewards]), \
             mock.patch.object(cluster,"generate",side_effect=generate), \
             mock.patch.object(cluster,"update_actor",side_effect=update), mesh:
            trainer.train(Batches())
        if cluster.actor_trainer.train_steps!=3 or len(report["phases"])!=3:
            raise RuntimeError("expected three actor updates")
        records=[json.loads(line) for line in (output/"selection/ds_selection.jsonl").read_text().splitlines()]
        validate_hbm_selection(records,report["phases"])
        report["headroom_verified"]=True
        report["passed"]=True
        save()
        print(f"PASS: {output/'hbm_report.json'}")
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__=="__main__":
    main()
