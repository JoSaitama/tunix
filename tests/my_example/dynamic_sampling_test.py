"""Host DS behavior tests plus real JAX gather/padding, without model download."""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from my_example.dynamic_sampling import (
    CyclingBatches, concatenate_examples, dynamic_sampling_learner_class, eligible_groups,
)


def production_noise_module():
    """Run the actual host noise definitions without importing the TPU trainer."""
    source=Path(__file__).resolve().parents[2]/"my_example/reward_rank_noise.py"
    tree=ast.parse(source.read_text())
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=="RewardRankNoiseGRPOLearner":
            break
        if isinstance(node,ast.ImportFrom) and (node.module or "").startswith("tunix"):
            continue
        nodes.append(node)
    module=ModuleType("_ds_actual_host_noise")
    with mock.patch.dict(sys.modules,{module.__name__:module}):
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),module.__dict__)
    return module


@dataclasses.dataclass(frozen=True)
class Example:
    prompt_ids: object
    prompt_mask: object
    completion_ids: object
    completion_mask: object
    ref_per_token_logps: object
    old_per_token_logps: object
    advantages: object
    filter_random_values: object

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def tree_map(fn, example):
    return Example(**{f.name: None if getattr(example, f.name) is None
                     else fn(getattr(example, f.name)) for f in dataclasses.fields(example)})


class Training(SimpleNamespace):
    def get_with_default(self, key, default):
        return getattr(self, key, default)


def cluster():
    return SimpleNamespace(
        cluster_config=SimpleNamespace(training_config=Training(
            use_dynamic_batch_curation=False, max_steps=691, train_micro_batch_size=4)),
        actor_trainer=SimpleNamespace(train_steps=0), global_steps=0,
        rollout=SimpleNamespace(pad_id=lambda: 99),
        buffer_metrics=lambda *a, **k: None,
    )


def batch(index, rewards=None):
    return {"prompts": np.array([f"p{index+i}" for i in range(4)]),
            "answer": np.array([str(i) for i in range(4)]),
            "ids": np.arange(index, index+4),
            "test_rewards": np.asarray(rewards if rewards is not None else [[0,1,2,3]]*4)}


class Base:
    def __init__(self, rl_cluster, algo_config, **kwargs):
        self.rl_cluster, self.algo_config = rl_cluster, algo_config
        self._training_config = rl_cluster.cluster_config.training_config
        self.can_enable_async_rollout = False
        self._last_iter_step = self._iter_steps = 0
        self.calls = []

    def _compute_rewards(self, prompts, completions, mode, **kwargs):
        values = np.asarray(completions, dtype=float)
        if getattr(self,"noise_config",None) is not None:
            noise=production_noise_module()
            values=noise.apply_reward_rank_noise(values,prompts,4,self.noise_config).corrupted_rewards
        return -values if getattr(self, "reverse", False) else values

    def _generate_and_compute_advantage(self, data, mode):
        self.calls.append(data)
        # Each prompt-leading test_rewards vector is repeated four times;
        # retain one vector per prompt as the four generated reward outcomes.
        rewards = self._compute_rewards(data["prompts"], data["test_rewards"][::4].reshape(-1), mode)
        grouped = rewards.reshape(-1,4)
        advantages = ((grouped-grouped.mean(1,keepdims=True)) /
                      (grouped.std(1,ddof=1,keepdims=True)+1e-4)).reshape(-1)
        ids = np.asarray(data["ids"])[:,None]
        return Example(ids, ids+100, ids+200, ids+300, ids+400, None, advantages,
                       np.repeat(ids, 2, axis=1))

    def _prepare_data(self, **kwargs):
        return "unchanged-evaluation"

    def train(self, train_ds, eval_ds=None, skip_jit=False):
        return train_ds, eval_ds, skip_jit


class AlgorithmTest(unittest.TestCase):
    def test_binary_and_dense_rule(self):
        np.testing.assert_array_equal(eligible_groups([[0]*4,[1]*4,[0,1,0,1],[.1,.2,.1,.2]]),
                                      [False,False,True,True])

    def test_rank_permutation_invariance(self):
        values = np.array([[0,0,0,0],[.2,.5,3,4],[1,1,0,0]])
        np.testing.assert_array_equal(eligible_groups(values),eligible_groups(values[:,::-1]))

    def test_invalid(self):
        for value in ([], [0,1], [[0]], [[0,np.nan]], [[0,np.inf]]):
            with self.assertRaises(ValueError): eligible_groups(value)

    def test_cycle_only_supplied_partition(self):
        iterator = iter(CyclingBatches(["train0","train1"]))
        self.assertEqual([next(iterator) for _ in range(5)], ["train0","train1","train0","train1","train0"])
        with self.assertRaises(ValueError): next(iter(CyclingBatches([])))
        with self.assertRaises(ValueError): CyclingBatches(iter([1,2]))


class PipelineTest(unittest.TestCase):
    def setUp(self):
        jax = ModuleType("jax")
        jax.tree = SimpleNamespace(map=tree_map)
        jnp = ModuleType("jax.numpy")
        jnp.pad, jnp.concatenate = np.pad, np.concatenate
        tunix, rl, cl = ModuleType("tunix"), ModuleType("tunix.rl"), ModuleType("tunix.rl.rl_cluster")
        cl.Mode = SimpleNamespace(TRAIN="train", EVAL="eval")
        rl.rl_cluster, tunix.rl = cl, rl
        self.patch = mock.patch.dict(sys.modules, {"jax":jax,"jax.numpy":jnp,
            "tunix":tunix,"tunix.rl":rl,"tunix.rl.rl_cluster":cl})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.env = mock.patch.dict(os.environ, {"TUNIX_PODS_KEEP_COUNT":""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def learner(self, cap=20):
        return dynamic_sampling_learner_class(Base)(rl_cluster=cluster(),
            algo_config=SimpleNamespace(num_generations=4,num_iterations=1,beta=.08),
            ds_max_generation_batches=cap, ds_artifact_dir=self.tmp.name)

    def test_identity_all_eligible(self):
        learner=self.learner()
        result=learner._fill_training_batch(iter([batch(0)]),4)
        np.testing.assert_array_equal(result.prompt_ids[:,0],np.repeat(np.arange(4),4))
        np.testing.assert_array_equal(result.ref_per_token_logps,result.prompt_ids+400)
        self.assertIsNone(result.old_per_token_logps)
        self.assertEqual(len(learner.calls),1)

    def test_refill_preserves_answers_advantages_and_order(self):
        learner=self.learner()
        first=batch(0,[[1]*4,[0,1,2,3],[0]*4,[1,2,3,4]])
        result=learner._fill_training_batch(iter([first,batch(4)]),4)
        np.testing.assert_array_equal(result.prompt_ids[:,0],np.repeat([1,3,4,5],4))
        self.assertTrue(np.isfinite(result.advantages).all())
        self.assertEqual(result.filter_random_values.shape,(16,2))
        # All four original advantages retained; no PODS-style subset normalization.
        np.testing.assert_allclose(result.advantages[:4],np.array([-1.5,-.5,.5,1.5])/(np.std([0,1,2,3],ddof=1)+1e-4))
        records=[json.loads(x) for x in (Path(self.tmp.name)/"ds_selection.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["selected"] for r in records),4)
        self.assertEqual(sum(r["reason"]=="excess_eligible" for r in records),2)
        self.assertEqual(sum(r["reason"]=="zero_variance" for r in records),2)
        ids=[r["trajectory_ids"] for r in learner.calls]
        self.assertFalse(set(ids[0]) & set(ids[1]))
        summary=json.loads((Path(self.tmp.name)/"ds_summary.json").read_text())
        self.assertEqual(summary["generated_groups"],8)
        self.assertEqual(summary["selected_groups"],4)

    def test_refill_cap_never_queues_partial_update(self):
        learner=self.learner(cap=2)
        q=queue.Queue()
        with self.assertRaisesRegex(RuntimeError,"no partial actor update"):
            learner._prepare_data(iterator=iter([batch(0,[[0]*4]*4)]*2),
                proceed_num_steps=1,sample_repeat=4,batch_repeat=1,
                service_target_batch_size=4,data_queue=q,mode="train")
        self.assertIsNone(q.get_nowait())
        self.assertTrue(q.empty())
        self.assertEqual(learner.rl_cluster.actor_trainer.train_steps,0)
        self.assertEqual(json.loads((Path(self.tmp.name)/"ds_summary.json").read_text())["status"],"refill_cap_exceeded")

    def test_queue_and_eval_delegation(self):
        learner=self.learner()
        q=queue.Queue()
        learner._prepare_data(iterator=iter([batch(0)]),proceed_num_steps=1,
            sample_repeat=4,batch_repeat=1,service_target_batch_size=4,data_queue=q,mode="train")
        self.assertEqual(len(q.get_nowait()[0].advantages),16)
        self.assertIsNone(q.get_nowait())
        self.assertEqual(learner._iter_steps,1)
        self.assertEqual(learner._prepare_data(iter([]),-1,4,1,4,q,mode="eval"),"unchanged-evaluation")

    def test_rank_noise_chain_is_used_for_selection_and_training(self):
        learner=self.learner()
        noise=production_noise_module()
        learner.noise_config=noise.RewardRankNoiseConfig(fraction=1.,seed=5)
        with mock.patch.dict(sys.modules,{"my_example.reward_rank_noise":noise}):
            result=learner._fill_training_batch(iter([batch(0)]),4)
        self.assertGreater(result.advantages[0],0)
        records=[json.loads(x) for x in (Path(self.tmp.name)/"ds_selection.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["observed_rewards"],[3,2,1,0])
        self.assertTrue(all(row["mismatch_selected"] for row in records))

    def test_mismatch20_mixed_pool_survives_into_actor_inputs(self):
        learner=self.learner()
        noise=production_noise_module()
        learner.noise_config=noise.RewardRankNoiseConfig(fraction=.2,seed=5)
        noisy=next(f"q{i}" for i in range(1000) if noise.selected_prompt(f"q{i}",learner.noise_config))
        clean=next(f"q{i}" for i in range(1000) if not noise.selected_prompt(f"q{i}",learner.noise_config))
        data=batch(0)
        data["prompts"]=np.array([noisy,clean,noisy,clean])
        # Pure host membership function is loaded as the production module for
        # the record writer too; no synthetic noise implementation is used.
        with mock.patch.dict(sys.modules,{"my_example.reward_rank_noise":noise}):
            result=learner._fill_training_batch(iter([data]),4)
        self.assertGreater(result.advantages[0],0)
        self.assertLess(result.advantages[4],0)
        records=[json.loads(x) for x in (Path(self.tmp.name)/"ds_selection.jsonl").read_text().splitlines()]
        self.assertEqual([row["mismatch_selected"] for row in records],[True,False,True,False])
        self.assertTrue(all(row["selected"] for row in records))

    def test_reject_unsupported_modes_and_resume(self):
        learner=self.learner()
        learner.rl_cluster.actor_trainer.train_steps=1
        # A policy change during generation must be caught.
        original=learner._compute_rewards
        def changed(*args,**kwargs):
            learner.rl_cluster.actor_trainer.train_steps += 1
            return original(*args,**kwargs)
        learner._compute_rewards=changed
        with self.assertRaisesRegex(RuntimeError,"Actor changed"):
            learner._fill_training_batch(iter([batch(0)]),4)

    def test_short_batch_rejected(self):
        learner=self.learner()
        short={k:v[:2] for k,v in batch(0).items()}
        with self.assertRaisesRegex(ValueError,"full fixed-size"):
            learner._fill_training_batch(iter([short]),4)

    def test_constructor_rejects_restored_actor_step(self):
        current=cluster()
        current.actor_trainer.train_steps=1
        with self.assertRaisesRegex(ValueError,"resume is not implemented"):
            dynamic_sampling_learner_class(Base)(rl_cluster=current,
                algo_config=SimpleNamespace(num_generations=4,num_iterations=1,beta=.08))

    def test_mutually_exclusive_and_unsupported_configs(self):
        for dbc,iterations,accum in [(True,1,1),(False,2,1),(False,1,2)]:
            current=cluster()
            current.cluster_config.training_config.use_dynamic_batch_curation=dbc
            current.cluster_config.training_config.gradient_accumulation_steps=accum
            with self.assertRaises(ValueError):
                dynamic_sampling_learner_class(Base)(rl_cluster=current,
                    algo_config=SimpleNamespace(num_generations=4,num_iterations=iterations,beta=.08))

    def test_partial_cap_does_not_become_a_training_batch(self):
        learner=self.learner(cap=1)
        q=queue.Queue()
        with self.assertRaisesRegex(RuntimeError,"1/4 eligible"):
            learner._prepare_data(iterator=iter([batch(0,[[0,1,2,3],[0]*4,[0]*4,[0]*4])]),
                proceed_num_steps=1,sample_repeat=4,batch_repeat=1,
                service_target_batch_size=4,data_queue=q,mode="train")
        self.assertIsNone(q.get_nowait())
        self.assertTrue(q.empty())
        summary=json.loads((Path(self.tmp.name)/"ds_summary.json").read_text())
        self.assertEqual(summary["prepared_updates"],0)
        self.assertEqual(summary["selected_groups"],1)

    def test_train_cycles_only_training_and_preserves_eval(self):
        learner=self.learner()
        dataset,ev,skip=learner.train(["train"], ["eval"], True)
        it=iter(dataset)
        self.assertEqual([next(it),next(it)],["train","train"])
        self.assertEqual(ev,["eval"])
        self.assertTrue(skip)


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires JAX; run server CPU tests")
class RealJaxTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("flax"),"requires Flax")
    def test_real_flax_pytree_gather_preserves_optional_fields(self):
        import flax.struct
        import jax
        import jax.numpy as jnp
        @flax.struct.dataclass
        class RealExample:
            prompt_ids: object
            advantages: object
            old_per_token_logps: object
        example=RealExample(jnp.arange(16)[:,None],jnp.arange(16.),None)
        selected=jax.tree.map(lambda leaf:leaf[np.array([4,5,6,7,12,13,14,15])],example)
        result=concatenate_examples([selected,selected],pad_id=99)
        np.testing.assert_array_equal(result.prompt_ids[:,0],[4,5,6,7,12,13,14,15]*2)
        self.assertIsNone(result.old_per_token_logps)

    def test_variable_lengths_and_none_fields(self):
        import jax.numpy as jnp
        def example(prompt_width, completion_width):
            return Example(jnp.full((4,prompt_width),7),jnp.ones((4,prompt_width)),
                jnp.full((4,completion_width),8),jnp.ones((4,completion_width)),
                jnp.ones((4,completion_width)),None,jnp.arange(4.),jnp.ones((4,2)))
        result=concatenate_examples([example(2,3),example(3,2)],pad_id=99)
        np.testing.assert_array_equal(result.prompt_ids[0],[99,7,7])
        np.testing.assert_array_equal(result.prompt_mask[0],[0,1,1])
        np.testing.assert_array_equal(result.completion_ids[-1],[8,8,99])
        np.testing.assert_array_equal(result.completion_mask[-1],[1,1,0])
        np.testing.assert_array_equal(result.ref_per_token_logps[-1],[1,1,0])
        self.assertIsNone(result.old_per_token_logps)
        self.assertEqual(result.advantages.shape,(8,))


class LauncherTest(unittest.TestCase):
    def test_dry_run_frozen_parameters_and_naming(self):
        root=Path(__file__).resolve().parents[2]
        env=dict(os.environ,TUNIX_PODS_KEEP_COUNT="",TUNIX_EXPERIMENT_SEED="5",
                 TUNIX_REWARD_RANK_NOISE_FRACTION="0.2")
        completed=subprocess.run(["bash","my_example/run_ds_baseline.sh","--dry-run"],
            cwd=root,env=env,text=True,capture_output=True,check=True)
        command=completed.stdout
        for text in ["TUNIX_DYNAMIC_SAMPLING=1","TPU_PROCESS_BOUNDS=1\\,1\\,1",
                     "--num-generations 4","--train-micro-batch-size 4","--learning-rate 1e-6",
                     "--max-eval-examples 1319","rebuttal_grpo_gsm8k_ds_seed5_mismatch0.2_"]:
            self.assertIn(text,command)
        self.assertNotIn("--skip-eval",command)

    def test_factory_optin_clean_and_noise_and_original_paths(self):
        source=Path(__file__).resolve().parents[2]/"my_example/train.py"
        tree=ast.parse(source.read_text())
        fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="build_trainer")
        namespace={"GRPOLearner":Base,"GRPOLearner__annotation":Base,
            "GrpoGenerationConfig":object,"os":os,"build_grpo_config":lambda value:value}
        for name in ["format_integrity_reward","numeric_relative_reward","match_format_exactly",
                     "match_format_approximately","check_answer","check_numbers"]:
            namespace[name]=lambda **kwargs:None
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),"exec"),namespace)
        noise=production_noise_module()
        noise.config_from_env=lambda:SimpleNamespace(enabled=False)
        noise.RewardRankNoiseGRPOLearner=Base
        cfg=SimpleNamespace(num_generations=4,num_iterations=1,beta=.08)
        with mock.patch.dict(sys.modules,{"my_example.reward_rank_noise":noise}), \
             mock.patch.dict(namespace,{"__package__":"my_example"}), \
             mock.patch.dict(os.environ,{"TUNIX_DYNAMIC_SAMPLING":"","TUNIX_PODS_KEEP_COUNT":""}):
            original=namespace["build_trainer"](cluster(),cfg)
            self.assertIs(type(original),Base)
            with mock.patch.dict(os.environ,{"TUNIX_DYNAMIC_SAMPLING":"1"}):
                learner=namespace["build_trainer"](cluster(),cfg)
                from my_example.dynamic_sampling import DynamicSamplingMixin
                self.assertIsInstance(learner,DynamicSamplingMixin)
                noise.config_from_env=lambda:SimpleNamespace(enabled=True,fraction=.2,seed=5)
                noisy=namespace["build_trainer"](cluster(),cfg)
                self.assertIsInstance(noisy,DynamicSamplingMixin)
                with mock.patch.dict(os.environ,{"TUNIX_PODS_KEEP_COUNT":"2"}):
                    with self.assertRaises(ValueError): namespace["build_trainer"](cluster(),cfg)


if __name__ == "__main__":
    unittest.main()
