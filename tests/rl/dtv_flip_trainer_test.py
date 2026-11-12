"""DTV-flip numerical/update tests; optional isolated framework import for CPU QA.

TUNIX_FLIP_TEST_ISOLATED=1 replaces only the unavailable Trainer base import.
JAX, Flax autodiff, JIT, model, and optimizer remain real. Default mode imports
all production dependencies and should be run in the training environment.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[2]


def load_trainer(name):
  if os.environ.get("TUNIX_FLIP_TEST_ISOLATED") != "1":
    import importlib
    return importlib.import_module("tunix.rl." + name)
  path = ROOT / "tunix/rl" / (name + ".py")
  tree = ast.parse(path.read_text())
  tree.body = [n for n in tree.body if not (
      isinstance(n, ast.ImportFrom) and n.module == "tunix.rl"
      and any(a.name == "trainer" for a in n.names)
  )]
  base = type("IsolatedTrainer", (), {"_post_process_train_step": lambda self, aux: None})
  module = SimpleNamespace(rl_trainer=SimpleNamespace(Trainer=base))
  exec(compile(tree, str(path), "exec"), module.__dict__)
  return module


flip = load_trainer("dtv_flip_trainer")
ordinary = load_trainer("self_inf_trainer")
loo_reference = load_trainer("self_inf_loo_trainer")


class Model(nnx.Module):
  def __init__(self):
    self.weight = nnx.Param(jnp.asarray(2.0))


def make_trainer(loo=False, scope="batch", generations=4, aux=True, ratio="1",
                 batch_ratio=None, frequency=1, grad_acc_steps=1, random_seed=None):

  trainer = object.__new__(flip.DTVFlipTrainer)
  trainer.flip_numerator, trainer.flip_denominator = flip.ratio_terms(
      ratio if batch_ratio is None else batch_ratio
  )
  trainer.quota_policy = ("cumulative_floor_no_catchup" if batch_ratio is None
                          else "batch_nearest_even")
  trainer.frequency = frequency
  trainer.random_flip_seed = random_seed
  trainer._grad_acc_steps = grad_acc_steps
  trainer.scope, trainer.loo = scope, loo
  trainer.num_generations, trainer.decisions_path = generations, None
  trainer._lora_enabled, trainer._has_aux = False, aux
  trainer.gen_model_input_fn = lambda inputs: inputs
  def loss(model, policy, kl):
    value = model.weight[...] * jnp.mean(policy + kl)
    return (value, {"kl": model.weight[...] * jnp.mean(kl)}) if aux else value
  trainer.loss_fn = loss
  return trainer


def load_grpo_loss():
  if os.environ.get("TUNIX_FLIP_TEST_ISOLATED") != "1":
    from tunix.rl.grpo import grpo_learner
    return grpo_learner.grpo_loss_fn, grpo_learner.common
  # Keep the production loss, KL and reduction while substituting a tiny model
  # forward for the server-only import graph.
  namespace = dict(jax=jax, jnp=jnp, Any=Any)
  for filename, names in (
      ("common.py", {"compute_kl_divergence", "aggregate_loss"}),
      ("grpo/grpo_learner.py", {"grpo_loss_fn"}),
  ):
    source = ROOT / "tunix/rl" / filename
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
      node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
  common = SimpleNamespace(compute_per_token_logps=None,
                           compute_kl_divergence=namespace["compute_kl_divergence"],
                           aggregate_loss=namespace["aggregate_loss"])
  namespace["common"] = common
  return namespace["grpo_loss_fn"], common


class DTVFlipTest(unittest.TestCase):
  def test_loo_matches_existing_full_gradient_statistics(self):
    grads = {"a": jnp.array([[1., -2.], [3., 1.], [-4., 2.], [0., 1.]]),
             "b": jnp.array([2., -1., 0., 4.])}
    for scope in ("batch", "group"):
      expected = (loo_reference._batch_loo_statistics(grads) if scope == "batch"
                  else loo_reference._group_loo_statistics(grads, 2))
      for is_loo, key in ((True, "loo_score"), (False, "standard_score")):
        with self.subTest(scope=scope, loo=is_loo):
          actual = flip.alignment_scores(grads, scope=scope,
                                         num_generations=2, loo=is_loo)
          np.testing.assert_allclose(actual, expected[key], atol=1e-6)

  def test_ordinary_matches_existing_mask_decision(self):
    # Compare actual old trainer decisions, without changing the old trainer.
    class Recorder:
      def update(self, model, grads):
        self.grads = grads
    for scope in ("batch", "group"):
      trainer = object.__new__(ordinary.SelfInfTrainer)
      trainer.__dict__.update(make_trainer(scope=scope, generations=2).__dict__)
      trainer.dot_threshold = 0.0
      trainer._score_loss_fn = None
      recorder = Recorder()
      policy = jnp.array([2., -1., 1., -3.])
      _, aux = trainer._train_step(Model(), recorder,
                                  dict(policy=policy, kl=jnp.zeros(4)))
      scores = flip.alignment_scores({"w": policy}, scope=scope, num_generations=2)
      self.assertEqual(float(aux["skipped_samples"]), int(jnp.sum(scores < 0)))
      np.testing.assert_allclose(aux["self_inf_dot_mean"], jnp.mean(scores))

  def test_full_gradients_control_scores_and_flip_kl_too(self):
    # Policy gradients are all +1; total gradients [2,-1,0,0] give a negative
    # second DTV score. Policy-only scoring/flipping would give other updates.
    for is_loo, expected in ((False, .75), (True, -.25)):
      for compiled in (False, True):
        with self.subTest(loo=is_loo, compiled=compiled):
          trainer = make_trainer(loo=is_loo)
          model = Model()
          optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
          trainer.initialize_budget_state(optimizer, {}, 0)
          step = nnx.jit(trainer._train_step) if compiled else trainer._train_step
          loss, aux = step(model, optimizer, dict(
              policy=jnp.ones(4), kl=jnp.array([1., -2., -1., -1.])))
          np.testing.assert_allclose(model.weight[...], 2. - .1 * expected, rtol=1e-6)
          np.testing.assert_allclose(loss, 2. * expected)
          np.testing.assert_allclose(aux["kl"], -1.5)  # diagnostic is not signed
          self.assertEqual(float(aux["skipped_samples"]), 0.)
          self.assertEqual(int(optimizer.step[...]), 1)
          self.assertEqual(int(aux["dtv_flip_count"]), 2 if is_loo else 1)

  def test_no_aux_path(self):
    trainer = make_trainer(aux=False)
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    loss, aux = nnx.jit(trainer._train_step)(
        model, optimizer, dict(policy=jnp.array([2., -1.]), kl=jnp.zeros(2)))
    self.assertIsNone(aux)
    np.testing.assert_allclose(loss, 3.)
    np.testing.assert_allclose(model.weight[...], 1.85)

  def test_group_scope_has_independent_references(self):
    grads = {"w": jnp.array([2., -1., -3., 1.])}
    np.testing.assert_allclose(flip.alignment_scores(grads, scope="group", num_generations=2),
                               [1., -.5, 3., -1.])
    np.testing.assert_allclose(flip.alignment_scores(grads, scope="group", num_generations=2, loo=True),
                               [-2., -2., -3., -3.])

  def test_all_negative_loo_has_no_retention_cap(self):
    trainer = make_trainer(loo=True, generations=2)
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    _, aux = trainer._train_step(model, optimizer,
                                dict(policy=jnp.array([2., -1.]), kl=jnp.zeros(2)))
    self.assertEqual(int(aux["dtv_flip_count"]), 2)
    np.testing.assert_allclose(model.weight[...], 2.05)

  def test_zero_score_preserves_nonzero_gradient(self):
    # Orthogonal gradients have zero LOO scores and must both keep +1 weight.
    grads = {"w": jnp.eye(2)}
    scores = flip.alignment_scores(grads, loo=True)
    np.testing.assert_array_equal(scores, [0., 0.])
    np.testing.assert_allclose(flip.signed_mean(grads, jnp.where(scores < 0, -1., 1.))["w"], [.5, .5])

  def test_signed_denominator_is_original_count(self):
    grads = {"a": jnp.array([3., -1.]), "b": jnp.array([[2., 4.], [-6., -8.]])}
    result = flip.signed_mean(grads, jnp.array([1., -1.]))
    np.testing.assert_allclose(result["a"], 2.)
    np.testing.assert_allclose(result["b"], [4., 6.])

  def test_positive_scores_match_unmodified_update(self):
    trainer = make_trainer()
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    _, aux = trainer._train_step(model, optimizer,
                                dict(policy=jnp.array([1., 3.]), kl=jnp.zeros(2)))
    np.testing.assert_allclose(model.weight[...], 1.8)
    self.assertEqual(int(aux["dtv_flip_count"]), 0)

  def test_invalid_scoring_populations(self):
    for kwargs in (dict(scope="bad"), dict(scope="group", num_generations=3),
                   dict(scope="group", num_generations=0),
                   dict(scope="group", num_generations=1, loo=True)):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        flip.alignment_scores({"w": jnp.ones(4)}, **kwargs)
    with self.assertRaises(ValueError):
      flip.alignment_scores({"w": jnp.ones(1)}, loo=True)

  def test_nonfinite_score_aborts_eager_and_jit(self):
    for compiled in (False, True):
      for value in (float("nan"), float("inf")):
        fn = jax.jit(flip._reject_nonfinite) if compiled else flip._reject_nonfinite
        with self.subTest(compiled=compiled, value=value), self.assertRaisesRegex(Exception, "nonfinite alignment score"):
          fn(jnp.array([value]))
          jax.effects_barrier()

  def test_environment_rejects_policy_variants(self):
    for key in ("TUNIX_DBC_SELF_INF_POLICY", "TUNIX_DBC_SELF_INF_LOO_POLICY",
                "TUNIX_DBC_SELF_INF_LOO_POLICY_ONLY"):
      with self.assertRaisesRegex(ValueError, "full gradients"):
        flip.configuration_from_environment({key: "1"})
    config = flip.configuration_from_environment({"TUNIX_DBC_SELF_INF_LOO": "1",
        "TUNIX_DBC_SELF_INF_MIN_KEEP_FRACTION": "invalid-but-unused"})
    self.assertTrue(config["loo"])
    self.assertNotIn("min_keep_fraction", config)

  def test_cluster_branch_selects_flip_only_when_opted_in(self):
    # Execute the real selection block with lightweight class sentinels. This
    # avoids requiring TPU rollout packages in a CPU routing test.
    source = (ROOT / "tunix/rl/rl_cluster.py").read_text()
    start = source.index("    actor_trainer_kwargs = {}")
    end = source.index("    if actor_config.use_dynamic_batch_curation and actor_trainer_cls is robust_trainer", start)
    import textwrap
    block = ast.parse(textwrap.dedent(source[start:end]))
    for node in ast.walk(block):
      if isinstance(node, ast.If):
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    from unittest import mock
    names = {
        "dtv_flip_trainer": flip,
        "self_inf_trainer": SimpleNamespace(SelfInfTrainer=type("dtv", (), {})),
        "self_inf_loo_trainer": SimpleNamespace(SelfInfLooTrainer=type("loo", (), {})),
        "robust_trainer": SimpleNamespace(RobustTrainer="robust"),
        "rl_trainer": SimpleNamespace(Trainer="baseline"),
        "logging": SimpleNamespace(info=lambda *a: None), "os": os,
    }
    for enabled, flag, loo, random, expected in (
        (False, "1", "0", "0", "baseline"), (True, "", "0", "0", "dtv"),
        (True, "", "1", "0", "loo"), (True, "1", "0", "0", flip.DTVFlipTrainer),
        (True, "1", "1", "0", flip.DTVFlipTrainer),
        (True, "1", "0", "1", flip.DTVFlipTrainer)):
      env = {"TUNIX_DBC_VARIANT": "self_inf", "TUNIX_DBC_SELF_INF_FLIP": flag,
             "TUNIX_DBC_SELF_INF_LOO": loo, "TUNIX_DTV_FLIP_RANDOM": random}
      namespace = dict(names, actor_config=SimpleNamespace(use_dynamic_batch_curation=enabled))
      with mock.patch.dict(os.environ, env, clear=True):
        exec(compile(block, "cluster_selection", "exec"), namespace)
      actual = namespace["actor_trainer_cls"]
      if expected in ("dtv", "loo"):
        actual = actual.__name__
      self.assertEqual(actual, expected)
      if expected == flip.DTVFlipTrainer:
        self.assertEqual(namespace["actor_trainer_kwargs"]["loo"], loo == "1")
        self.assertEqual(namespace["actor_trainer_kwargs"]["random_flip_seed"],
                         0 if random == "1" else None)
        self.assertNotIn("min_keep_fraction", namespace["actor_trainer_kwargs"])


  @unittest.skipUnless(jax.local_device_count() >= 2, "requires multiple CPU devices")
  def test_sharded_jit_update(self):
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    trainer = make_trainer(loo=True, scope="group", generations=2, ratio="0.1")
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    inputs = dict(policy=jax.device_put(jnp.tile(jnp.array([2., -1.]), 8), sharding),
                  kl=jax.device_put(jnp.zeros(16), sharding))
    step = nnx.jit(trainer._train_step)
    quotas = []
    for _ in range(5):
      _, aux = step(model, optimizer, inputs)
      quotas.append(int(aux["dtv_flip_count"]))
    self.assertEqual(quotas, [1, 2, 1, 2, 2])
    np.testing.assert_allclose(model.weight[...], 1.8375, rtol=1e-6)
    self.assertEqual(int(optimizer.dtv_flip_budget.seen[...]), 80)

  def test_lora_updates_only_trainable_lora_parameters(self):
    class LoRAModel(nnx.Module):
      def __init__(self):
        self.adapter = nnx.LoRAParam(jnp.asarray(2.0))
        self.frozen = nnx.Param(jnp.asarray(7.0))
    trainer = make_trainer()
    trainer._lora_enabled = True
    trainer.loss_fn = lambda model, x: (model.adapter[...] * x + model.frozen[...], {})
    model = LoRAModel()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.LoRAParam)
    trainer.initialize_budget_state(optimizer, {}, 0)
    nnx.jit(trainer._train_step)(model, optimizer, {"x": jnp.array([2., -1.])})
    np.testing.assert_allclose(model.adapter[...], 1.85)
    np.testing.assert_allclose(model.frozen[...], 7.)

  def test_decision_record_preserves_raw_scores_and_signed_weights(self):
    trainer = make_trainer(loo=True)
    trainer._train_steps = 0
    trainer.rl_metrics_to_log = {}
    trainer._buffered_train_metrics = SimpleNamespace(additional_metrics={})
    with tempfile.TemporaryDirectory() as directory:
      trainer.decisions_path = str(Path(directory) / "decisions.jsonl")
      model = Model()
      optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
      trainer.initialize_budget_state(optimizer, {}, 0)
      _, aux = trainer._train_step(model, optimizer,
                                  dict(policy=jnp.array([2., -1.]), kl=jnp.zeros(2)))
      trainer._post_process_train_step(aux)
      record = json.loads(Path(trainer.decisions_path).read_text())
      self.assertEqual(record["method"], "dtv-flip")
      self.assertEqual(record["score"], "dtv-loo")
      self.assertEqual(record["score_objective"], "total")
      self.assertEqual(record["flip_application"], "full")
      self.assertEqual(record["weights"], [-1., -1.])
      self.assertEqual(record["scores"], [-2., -2.])
      self.assertNotIn("dtv_flip_scores", aux)
      self.assertIn("dtv_flip_fraction", trainer.rl_metrics_to_log)

  def test_launcher_forwards_modes_paths_and_stops_on_failure(self):
    # Execute the real launcher with a fake downstream command; no TPU/network.
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      scripts = root / "my_example"
      scripts.mkdir()
      launcher = scripts / "run_dtv_flip.sh"
      launcher.write_text((ROOT / "my_example/run_dtv_flip.sh").read_text())
      downstream = scripts / "run_grpo_gemma.sh"
      downstream.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$TUNIX_DBC_SELF_INF_FLIP" "$TUNIX_DBC_SELF_INF_LOO" "$TUNIX_DTV_FLIP_DECISIONS_PATH" "$TUNIX_DTV_FLIP_MAX_FRACTION" "$TUNIX_DTV_FLIP_BATCH_FRACTION" "$TUNIX_DTV_FLIP_FREQUENCY" "$TUNIX_DTV_FLIP_RANDOM" "$@" > "$CAPTURE"\nexit "${FAKE_EXIT:-0}"\n')
      downstream.chmod(0o755)
      (scripts / "save_results_to_my_result.py").write_text("pass\n")
      results = scripts / "my result"
      results.mkdir()
      (results / "plot_global_eval_rewards_sum.py").write_text("pass\n")
      capture = root / "args.txt"
      env = dict(os.environ, CAPTURE=str(capture), TUNIX_MY_RESULT_DIR=str(results),
                 TUNIX_DTV_FLIP_MAX_FRACTION="0.1", TUNIX_DTV_FLIP_FREQUENCY="1",
                 TUNIX_DTV_FLIP_BATCH_FRACTION="")
      # Export subprocess's interpreter location so the legacy export command
      # named 'python' resolves in both server and isolated CPU environments.
      import sys
      env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
      for score in ("dtv", "dtv-loo"):
        for scope in ("batch", "group"):
          command = ["bash", str(launcher), score, scope,
                     "--metrics-log-dir", str(root / "custom metrics"),
                     "--checkpoint-root=" + str(root / "custom ckpt")]
          result = subprocess.run(command, env=env, capture_output=True, text=True)
          self.assertEqual(result.returncode, 0, result.stderr)
          lines = capture.read_text().splitlines()
          self.assertEqual(lines[:2], ["1", "1" if score == "dtv-loo" else "0"])
          self.assertIn(f"dtv-flip_{score}_{scope}_cap0p1_", lines[2])
          self.assertEqual(lines[3], "0.1")
          self.assertEqual(lines[4:7], ["", "1", "0"])
          self.assertIn("--use-dbc-self-inf-" + scope, lines)
          self.assertIn(str(root / "custom metrics"), lines)
          self.assertIn(str(root / "custom ckpt"), lines)
      for cap in ("0", "0.2", "1"):
        result = subprocess.run(["bash", str(launcher), "dtv", "batch"],
            env=dict(env, TUNIX_DTV_FLIP_MAX_FRACTION=cap), capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(capture.read_text().splitlines()[3], cap)
      for cap in ("nan", "-0.1", "1.1", "../../bad"):
        result = subprocess.run(["bash", str(launcher), "dtv", "batch"],
            env=dict(env, TUNIX_DTV_FLIP_MAX_FRACTION=cap), capture_output=True)
        self.assertEqual(result.returncode, 2)
      command = ["bash", str(launcher), "dtv", "batch"]
      for batch_ratio, slug in (("", "cap0p1_freq2"), ("0.2", "batchcap0p2_freq2")):
        result = subprocess.run(command, env=dict(env, TUNIX_DTV_FLIP_FREQUENCY="2",
            TUNIX_DTV_FLIP_BATCH_FRACTION=batch_ratio), capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = capture.read_text().splitlines()
        self.assertIn(slug, lines[2])
        self.assertEqual(lines[4:6], [batch_ratio, "2"])
      result = subprocess.run(["bash", str(launcher), "random", "batch"],
          env=dict(env, TUNIX_DTV_FLIP_FREQUENCY="2"), capture_output=True)
      self.assertEqual(result.returncode, 0, result.stderr)
      lines = capture.read_text().splitlines()
      self.assertEqual(lines[:2], ["1", "0"])
      self.assertIn("dtv-flip_random_batch_batchcap0p1_freq2_", lines[2])
      self.assertEqual(lines[4:7], ["0.1", "2", "1"])
      for variable, value in (("TUNIX_DTV_FLIP_BATCH_FRACTION", "-0.1"),
                              ("TUNIX_DTV_FLIP_FREQUENCY", "0"),
                              ("TUNIX_DTV_FLIP_FREQUENCY", "2147483648")):
        result = subprocess.run(command, env=dict(env, **{variable: value}), capture_output=True)
        self.assertEqual(result.returncode, 2)
      result = subprocess.run(["bash", str(launcher), "dtv", "batch"],
                              env=dict(env, FAKE_EXIT="7"), capture_output=True)
      self.assertEqual(result.returncode, 7)
      for args in (["bad"], ["dtv", "invalid"], ["random", "group"],
                   ["dtv", "batch", "--use-dbc-self-inf-group"],
                   ["dtv", "batch", "--metrics-log-dir"]):
        result = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True)
        self.assertEqual(result.returncode, 2)


  def test_quota_sequence_and_stable_lowest_negative_selection(self):
    remainder = jnp.asarray(0, jnp.int32)
    scores = -jnp.arange(1, 17, dtype=jnp.float32)
    quotas = []
    for _ in range(5):
      weights, quota, remainder = jax.jit(
          lambda x, r: flip.select_flips(x, r, 1, 10))(scores, remainder)
      quotas.append(int(quota))
      self.assertEqual(int(jnp.sum(weights < 0)), int(quota))
      np.testing.assert_array_equal(np.flatnonzero(weights < 0),
                                    np.arange(16 - int(quota), 16))
    self.assertEqual(quotas, [1, 2, 1, 2, 2])
    self.assertEqual(int(remainder), 0)
    weights, _, _ = flip.select_flips(jnp.array([-2., -2., -1., 0.]), 0, 1, 2)
    np.testing.assert_array_equal(weights, [-1., -1., 1., 1.])

  def test_unused_negative_quota_expires_and_never_flips_nonnegative(self):
    remainder = jnp.asarray(0, jnp.int32)
    for scores, expected_quota, expected_count in (
        (jnp.ones(16), 1, 0),
        (jnp.array([-1.] + [1.] * 15), 2, 1),
        (-jnp.ones(16), 1, 1)):
      weights, quota, remainder = flip.select_flips(scores, remainder, 1, 10)
      self.assertEqual(int(quota), expected_quota)
      self.assertEqual(int(jnp.sum(weights < 0)), expected_count)
      self.assertFalse(bool(jnp.any((weights < 0) & (scores >= 0))))

  def test_variable_batch_sizes_preserve_every_prefix_cap(self):
    for ratio in ("0", "0.1", "0.25", "0.333333", "1"):
      p, q = flip.ratio_terms(ratio)
      seen = total = 0
      remainder = jnp.asarray(0, jnp.int32)
      for size in (4, 16, 4, 8, 3, 16, 1, 40):
        weights, _, remainder = flip.select_flips(-jnp.ones(size), remainder, p, q)
        seen += size
        total += int(jnp.sum(weights < 0))
        self.assertEqual(total, seen * p // q)
        self.assertEqual(int(remainder), seen * p % q)

  def test_quota_state_advances_inside_repeated_cached_jit(self):
    trainer = make_trainer(loo=True, scope="group", generations=2, ratio="0.1")
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    # Every 2-completion group is anti-aligned. Group scores stay local while
    # quota ranks all 16 completions; stable ties prioritize original indices.
    inputs = dict(policy=jnp.tile(jnp.array([2., -1.]), 8), kl=jnp.zeros(16))
    step = nnx.cached_partial(nnx.jit(trainer._train_step), model, optimizer)
    actual = []
    for _ in range(5):
      _, aux = step(inputs)
      actual.append(int(aux["dtv_flip_count"]))
      self.assertLessEqual(int(aux["dtv_flip_total"]), int(aux["dtv_flip_seen"]) // 10)
    self.assertEqual(actual, [1, 2, 1, 2, 2])
    np.testing.assert_allclose(model.weight[...], 1.8375, rtol=1e-6)
    self.assertEqual(int(optimizer.dtv_flip_budget.seen[...]), 80)
    self.assertEqual(int(optimizer.dtv_flip_budget.flipped[...]), 8)

  def test_nearest_even_quota_and_candidate_shortage(self):
    for size, quota in ((16, 2), (32, 4), (64, 6), (512, 52), (10, 2)):
      with self.subTest(size=size):
        self.assertEqual(flip.nearest_even_quota(size, 1, 10), quota)
    self.assertEqual(flip.nearest_even_quota(3, 1, 1), 2)
    trainer = make_trainer(batch_ratio="0.1")
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    step = nnx.jit(trainer._train_step)
    for policy, expected in ((jnp.ones(16), 0), (jnp.array([-1.] + [1.] * 15), 1)):
      _, aux = step(model, optimizer, dict(policy=policy, kl=jnp.zeros(16)))
      self.assertEqual(int(aux["dtv_flip_quota"]), 2)
      self.assertEqual(int(aux["dtv_flip_count"]), expected)

  def test_frequency_and_ordinary_adamw_share_one_optimizer(self):
    for batch_ratio, quotas in ((None, [1, 0, 2, 0, 1]), ("0.1", [2, 0, 2, 0, 2])):
      with self.subTest(batch_ratio=batch_ratio):
        trainer = make_trainer(loo=True, scope="group", generations=2, ratio="0.1",
                               batch_ratio=batch_ratio, frequency=2)
        trainer._train_steps = 0
        trainer.rl_metrics_to_log = {}
        trainer._buffered_train_metrics = SimpleNamespace(additional_metrics={})
        model, reference = Model(), Model()
        tx = optax.chain(optax.clip_by_global_norm(.3), optax.adamw(.1))
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        reference_optimizer = nnx.Optimizer(reference, tx, wrt=nnx.Param)
        trainer.initialize_budget_state(optimizer, {}, 0)
        policy = jnp.tile(jnp.array([2., -1.]), 8)
        inputs = dict(policy=policy, kl=jnp.zeros(16))
        step = nnx.cached_partial(nnx.jit(trainer._train_step), model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
          trainer.decisions_path = str(Path(directory) / "decisions.jsonl")
          for index, quota in enumerate(quotas):
            loss, aux = step(inputs)
            enabled = index % 2 == 0
            self.assertEqual(bool(aux["dtv_flip_enabled"]), enabled)
            self.assertEqual(int(aux["dtv_flip_quota"]), quota)
            self.assertEqual(int(aux["dtv_flip_count"]), quota)
            self.assertEqual(int(aux["dtv_flip_total_seen"]), 16 * (index + 1))
            self.assertEqual(int(aux["dtv_flip_seen"]), 16 * (index // 2 + 1))
            weights = jnp.ones(16).at[:quota].set(-1.)
            expected_loss, grads = nnx.value_and_grad(
                lambda model: trainer.loss_fn(model, policy * weights, inputs["kl"])[0]
            )(reference)
            reference_optimizer.update(reference, grads)
            np.testing.assert_allclose(loss, expected_loss, atol=1e-6)
            np.testing.assert_allclose(model.weight[...], reference.weight[...], atol=1e-6)
            trainer._post_process_train_step(aux)
            trainer._train_steps += 1
          rows = [json.loads(row) for row in Path(trainer.decisions_path).read_text().splitlines()]
        self.assertEqual([row["dtv_flip_quota"] for row in rows], quotas)
        self.assertIsNone(rows[1]["scores"])
        self.assertIsNone(rows[1]["dtv_flip_negative_count"])
        self.assertEqual(rows[1]["dtv_flip_fraction"], 0)
        self.assertEqual(rows[1]["batch_completion_count"], 16)
        self.assertEqual(rows[-1]["frequency"], 2)
        self.assertAlmostEqual(rows[-1]["dtv_flip_cumulative_fraction"], sum(quotas) / 80)
        self.assertAlmostEqual(rows[-1]["dtv_flip_active_fraction"], sum(quotas) / 48)

  @unittest.skipUnless(jax.local_device_count() >= 2, "requires multiple CPU devices")
  def test_frequency_uses_optimizer_updates_under_sharded_accumulation(self):
    trainer = make_trainer(loo=True, scope="group", generations=2, batch_ratio="0.1",
                           frequency=2, grad_acc_steps=2)
    model = Model()
    optimizer = nnx.Optimizer(model, optax.MultiSteps(optax.sgd(.1), 2), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    inputs = dict(policy=jax.device_put(jnp.tile(jnp.array([2., -1.]), 8), sharding),
                  kl=jax.device_put(jnp.zeros(16), sharding))
    step = nnx.jit(trainer._train_step)
    enabled = []
    for _ in range(6):
      _, aux = step(model, optimizer, inputs)
      enabled.append(bool(aux["dtv_flip_enabled"]))
    self.assertEqual(enabled, [True, True, False, False, True, True])

  def test_skipped_step_bypasses_alignment_scoring_at_runtime(self):
    trainer = make_trainer(frequency=2)
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(optimizer, {}, 0)
    inputs = dict(policy=jnp.array([2., -1.]), kl=jnp.zeros(2))
    calls = []
    alignment_scores = flip.alignment_scores

    def tracked_scores(*args, **kwargs):
      scores = alignment_scores(*args, **kwargs)
      jax.debug.callback(lambda _: calls.append(True), scores)
      return scores

    with mock.patch.object(flip, "alignment_scores", tracked_scores):
      step = nnx.jit(trainer._train_step)
      for expected in (1, 1, 2):
        step(model, optimizer, inputs)
        jax.effects_barrier()
        self.assertEqual(len(calls), expected)

  def test_random_flip_matches_signed_grpo_gradient_and_resumes(self):
    grpo_loss, common = load_grpo_loss()
    config = SimpleNamespace(beta=.08, epsilon=.2, loss_algo="grpo",
                             loss_agg_mode="sequence-mean-token-mean")

    def loss(model, train_example, **kwargs):
      return grpo_loss(model, SimpleNamespace(**train_example), config, 0, 1, **kwargs)

    def build(seed):
      trainer = make_trainer(batch_ratio="0.1", frequency=2, random_seed=seed)
      trainer.loss_fn = loss
      trainer._random_flip_loss = loss
      trainer.gen_model_input_fn = lambda inputs: {"train_example": inputs}
      trainer.custom_checkpoint_metadata_fn = lambda: {}
      trainer._train_steps = 0
      trainer.rl_metrics_to_log = {}
      trainer._buffered_train_metrics = SimpleNamespace(additional_metrics={})
      model = Model()
      trainer.optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
      trainer.initialize_budget_state(trainer.optimizer, {}, 0)
      return trainer, model

    tokens = jnp.tile(jnp.array([[2, 3, 0], [3, 2, 4]]), (8, 1))
    inputs = dict(completion_ids=tokens, completion_mask=tokens != 0,
                  prompt_ids=jnp.ones((16, 1), dtype=jnp.int32),
                  advantages=jnp.linspace(-1., 1., 16),
                  old_per_token_logps=jnp.full((16, 3), -.07),
                  ref_per_token_logps=jnp.full((16, 3), -.3))
    if jax.local_device_count() > 1:
      mesh = jax.sharding.Mesh(np.array(jax.local_devices()), ("data",))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
      inputs = jax.device_put(inputs, sharding)
    trainer, model = build(5)
    reference = Model()
    reference_optimizer = nnx.Optimizer(reference, optax.sgd(.1), wrt=nnx.Param)
    with (mock.patch.object(common, "compute_per_token_logps", side_effect=
                           lambda model, completion_tokens, **_: -.1 * model.weight[...] * completion_tokens),
          mock.patch.object(flip, "alignment_scores", side_effect=AssertionError("unexpected DTV scoring")),
          tempfile.TemporaryDirectory() as directory):
      trainer.decisions_path = str(Path(directory) / "random.jsonl")
      step = nnx.jit(trainer._train_step)
      for index, quota in enumerate((2, 0, 2)):
        if index == 2:
          resumed, resumed_model = build(5)
          resumed_model.weight[...] = model.weight[...]
          metadata = trainer.custom_checkpoint_metadata()
          resumed.initialize_budget_state(resumed.optimizer, metadata, 2)
          resumed.decisions_path = trainer.decisions_path
          next_step = nnx.jit(resumed._train_step)
          _, restored_aux = next_step(resumed_model, resumed.optimizer, inputs)
          different_seed, _ = build(13)
          with self.assertRaisesRegex(ValueError, "configuration differs"):
            different_seed.initialize_budget_state(different_seed.optimizer, metadata, 2)
        actual_loss, aux = step(model, trainer.optimizer, inputs)
        weights = aux["dtv_flip_weights"]
        self.assertIsNone(aux["dtv_flip_scores"])
        self.assertEqual(int(aux["dtv_flip_quota"]), quota)
        self.assertEqual(int(aux["dtv_flip_count"]), quota)

        def reference_loss(model):
          per_sample = jax.vmap(lambda sample: loss(model, sample)[0])(inputs)
          return jnp.mean(per_sample * weights)

        expected_loss, grads = nnx.value_and_grad(reference_loss)(reference)
        reference_optimizer.update(reference, grads)
        np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-6)
        np.testing.assert_allclose(model.weight[...], reference.weight[...], atol=1e-6)
        if index == 2:
          np.testing.assert_array_equal(restored_aux["dtv_flip_weights"], weights)
          np.testing.assert_allclose(resumed_model.weight[...], model.weight[...], atol=1e-6)
        trainer._post_process_train_step(aux)
        trainer._train_steps += 1
      rows = [json.loads(row) for row in Path(trainer.decisions_path).read_text().splitlines()]
    self.assertEqual([row["flip_enabled"] for row in rows], [True, False, True])
    self.assertEqual([len(row["flipped_indices"]) for row in rows], [2, 0, 2])
    self.assertTrue(all(row["method"] == "random-flip" and row["scores"] is None for row in rows))
    self.assertAlmostEqual(rows[-1]["dtv_flip_cumulative_fraction"], 4 / 48)

  def test_metadata_roundtrip_matches_uninterrupted_sequence(self):
    def build():
      trainer = make_trainer(loo=True, scope="group", generations=2, ratio="0.1")
      trainer.custom_checkpoint_metadata_fn = lambda: {"global_step": 9}
      model = Model()
      optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
      trainer.initialize_budget_state(optimizer, {}, 0)
      trainer.optimizer = optimizer
      return trainer, model, optimizer
    trainer, model, optimizer = build()
    inputs = dict(policy=jnp.tile(jnp.array([2., -1.]), 8), kl=jnp.zeros(16))
    step = nnx.jit(trainer._train_step)
    for _ in range(2):
      step(model, optimizer, inputs)
    metadata = json.loads(json.dumps(trainer.custom_checkpoint_metadata()))
    self.assertEqual(metadata["global_step"], 9)
    resumed, model2, optimizer2 = build()
    model2.weight[...] = model.weight[...]
    resumed.initialize_budget_state(optimizer2, metadata, 2)
    step2 = nnx.jit(resumed._train_step)
    for _ in range(3):
      loss1, aux1 = step(model, optimizer, inputs)
      loss2, aux2 = step2(model2, optimizer2, inputs)
      np.testing.assert_allclose(loss1, loss2)
      self.assertEqual(int(aux1["dtv_flip_quota"]), int(aux2["dtv_flip_quota"]))
      np.testing.assert_allclose(model.weight[...], model2.weight[...])
    self.assertEqual(trainer.custom_checkpoint_metadata(), resumed.custom_checkpoint_metadata())
    old_budget = dict(metadata["dtv_flip_budget"], version=1)
    for key in ("quota_policy", "frequency", "grad_acc_steps", "total_seen"):
      del old_budget[key]
    resumed.initialize_budget_state(optimizer2, {"dtv_flip_budget": old_budget}, 2)
    self.assertEqual(int(optimizer2.dtv_flip_budget.total_seen[...]), old_budget["seen"])
    resumed.frequency = 2
    with self.assertRaisesRegex(ValueError, "configuration differs"):
      resumed.initialize_budget_state(optimizer2, {"dtv_flip_budget": old_budget}, 2)
    resumed.frequency = 1
    for changed in (dict(metadata, dtv_flip_budget={}), {}):
      with self.assertRaises(ValueError):
        resumed.initialize_budget_state(optimizer2, changed, 2)
    resumed.flip_numerator, resumed.flip_denominator = 1, 5
    with self.assertRaisesRegex(ValueError, "configuration differs"):
      resumed.initialize_budget_state(optimizer2, metadata, 2)

  def test_final_checkpoint_keeps_quota_metadata(self):
    trainer = make_trainer(ratio="0.1")
    trainer.model, trainer._train_steps = Model(), 3
    trainer.optimizer = nnx.Optimizer(trainer.model, optax.sgd(.1), wrt=nnx.Param)
    trainer.initialize_budget_state(trainer.optimizer, {}, 0)
    trainer.custom_checkpoint_metadata_fn = lambda: {"global_step": 3}
    trainer.checkpoint_manager = SimpleNamespace(latest_step=lambda: 2, save=mock.Mock())
    trainer._save_last_checkpoint()
    args = trainer.checkpoint_manager.save.call_args.kwargs
    self.assertTrue(args["force"])
    self.assertIn("dtv_flip_budget", args["custom_metadata"])


  def test_real_checkpoint_save_restore_preserves_quota(self):
    # Load the production checkpoint manager directly to avoid unrelated
    # top-level model/metrics imports in explicit isolated mode only.
    import importlib.util
    path = ROOT / "tunix/sft/checkpoint_manager.py"
    spec = importlib.util.spec_from_file_location("_flip_checkpoint_manager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def build():
      trainer = make_trainer(loo=True, scope="group", generations=2, ratio="0.1", frequency=2)
      trainer.model = Model()
      trainer.optimizer = nnx.Optimizer(trainer.model, optax.sgd(.1), wrt=nnx.Param)
      trainer.initialize_budget_state(trainer.optimizer, {}, 0)
      trainer.custom_checkpoint_metadata_fn = lambda: {"global_step": 1}
      return trainer
    inputs = dict(policy=jnp.tile(jnp.array([2., -1.]), 8), kl=jnp.zeros(16))
    trainer = build()
    step = nnx.jit(trainer._train_step)
    step(trainer.model, trainer.optimizer, inputs)
    with tempfile.TemporaryDirectory() as directory:
      trainer.checkpoint_manager = module.CheckpointManager(directory)
      trainer._train_steps = 1
      try:
        trainer._save_last_checkpoint()
        trainer.checkpoint_manager.wait_until_finished()
        resumed = build()
        restored_step, metadata = trainer.checkpoint_manager.maybe_restore(resumed.model)
        resumed.initialize_budget_state(resumed.optimizer, metadata, restored_step)
        self.assertEqual(restored_step, 1)
        np.testing.assert_allclose(trainer.model.weight[...], resumed.model.weight[...])
        next_step = nnx.jit(resumed._train_step)
        for expected in (0, 2, 0):
          _, aux1 = step(trainer.model, trainer.optimizer, inputs)
          _, aux2 = next_step(resumed.model, resumed.optimizer, inputs)
          self.assertEqual(int(aux1["dtv_flip_quota"]), expected)
          self.assertEqual(int(aux2["dtv_flip_quota"]), expected)
          np.testing.assert_allclose(trainer.model.weight[...], resumed.model.weight[...])
        self.assertEqual(trainer.custom_checkpoint_metadata(), resumed.custom_checkpoint_metadata())
      finally:
        trainer.checkpoint_manager.close()

  def test_ratio_validation_and_default_environment(self):
    defaults = flip.configuration_from_environment({})
    self.assertEqual(defaults["max_flip_fraction"], "0.1")
    self.assertIsNone(defaults["batch_flip_fraction"])
    self.assertEqual(defaults["frequency"], 1)
    config = flip.configuration_from_environment({
        "TUNIX_DTV_FLIP_BATCH_FRACTION": "0.2", "TUNIX_DTV_FLIP_FREQUENCY": "2",
    })
    self.assertEqual(config["batch_flip_fraction"], "0.2")
    self.assertEqual(config["frequency"], 2)
    self.assertEqual(flip.configuration_from_environment({
        "TUNIX_DTV_FLIP_RANDOM": "1", "TUNIX_EXPERIMENT_SEED": "0",
    })["random_flip_seed"], 0)
    self.assertEqual(flip.ratio_terms("0.1"), (1, 10))
    for value in ("nan", "inf", "bad", "-0.1", "1.01", "0.0000001"):
      with self.subTest(value=value), self.assertRaises(ValueError):
        flip.ratio_terms(value)


if __name__ == "__main__":
  unittest.main()
