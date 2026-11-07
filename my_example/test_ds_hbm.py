"""Idle-worker DS preflight: real generation/update plus forced refill stress.

Synthetic generation/rewards exist ONLY in this disposable test, never in the
production DS selector. All stress candidate batches retain the frozen 4x4
shape, max prompt256/response768. Two stress updates force at least two refill
rounds, testing rejected and selected groups at full token length.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


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
              "scope":"one real rollout/update; two full-length synthetic forced-refill updates",
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
        batch={"prompts":np.array([f"What is {i}+1?" for i in range(4)]),
               "answer":np.array([str(i+1) for i in range(4)])}
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

        def rewards(prompts,completions,**unused):
            # Only two of four groups qualify: each update MUST refill twice.
            return np.array([[0,.5,3,4],[0,0,0,0],[0,.5,3,4],[1,1,1,1]]).reshape(-1)

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
            report["phases"].append({"name":"real_generation_and_actor_update" if not stress["active"]
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
            trainer.reward_fns=[rewards]
            return result

        report["configuration"]={"prompts":4,"num_generations":4,"selected_groups_per_update":4,
            "prompt_length":256,"stress_completion_length":768,"beta":cfg.grpo.beta}
        with mock.patch.object(cluster,"generate",side_effect=generate), \
             mock.patch.object(cluster,"update_actor",side_effect=update), mesh:
            trainer.train(Batches())
        if cluster.actor_trainer.train_steps!=3 or len(report["phases"])!=3:
            raise RuntimeError("expected three actor updates")
        records=[json.loads(line) for line in (output/"selection/ds_selection.jsonl").read_text().splitlines()]
        if sum(row["selected"] for row in records)!=12:
            raise RuntimeError("expected four complete selected groups per update")
        for phase in report["phases"][1:]:
            if not phase["name"].startswith("full_length_forced_refill"):
                raise RuntimeError("missing forced refill stress")
        # Last two prepared updates each exercised exactly two candidate batches.
        if report["phases"][2]["cumulative_generation_batches"]-report["phases"][0]["cumulative_generation_batches"]!=4:
            raise RuntimeError("stress did not exercise two refill rounds per update")
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
