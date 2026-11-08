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
import unittest

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


def make_trainer(loo=False, scope="batch", generations=4, aux=True):
  trainer = object.__new__(flip.DTVFlipTrainer)
  trainer.scope, trainer.loo = scope, loo
  trainer.num_generations, trainer.decisions_path = generations, None
  trainer._lora_enabled, trainer._has_aux = False, aux
  trainer.gen_model_input_fn = lambda inputs: inputs
  def loss(model, policy, kl):
    value = model.weight[...] * (policy + kl)
    return (value, {"kl": model.weight[...] * kl}) if aux else value
  trainer.loss_fn = loss
  return trainer


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
    for enabled, flag, loo, expected in (
        (False, "1", "0", "baseline"), (True, "", "0", "dtv"),
        (True, "", "1", "loo"), (True, "1", "0", flip.DTVFlipTrainer),
        (True, "1", "1", flip.DTVFlipTrainer)):
      env = {"TUNIX_DBC_VARIANT": "self_inf", "TUNIX_DBC_SELF_INF_FLIP": flag,
             "TUNIX_DBC_SELF_INF_LOO": loo}
      namespace = dict(names, actor_config=SimpleNamespace(use_dynamic_batch_curation=enabled))
      with mock.patch.dict(os.environ, env, clear=True):
        exec(compile(block, "cluster_selection", "exec"), namespace)
      actual = namespace["actor_trainer_cls"]
      if expected in ("dtv", "loo"):
        actual = actual.__name__
      self.assertEqual(actual, expected)
      if expected == flip.DTVFlipTrainer:
        self.assertEqual(namespace["actor_trainer_kwargs"]["loo"], loo == "1")
        self.assertNotIn("min_keep_fraction", namespace["actor_trainer_kwargs"])


  @unittest.skipUnless(jax.local_device_count() >= 2, "requires multiple CPU devices")
  def test_sharded_jit_update(self):
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    trainer = make_trainer()
    model = Model()
    optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.Param)
    inputs = dict(policy=jax.device_put(jnp.array([2., -1., 0., 0.]), sharding),
                  kl=jax.device_put(jnp.zeros(4), sharding))
    _, aux = nnx.jit(trainer._train_step)(model, optimizer, inputs)
    np.testing.assert_allclose(model.weight[...], 1.925)
    self.assertEqual(int(aux["dtv_flip_count"]), 1)

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
      downstream.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$TUNIX_DBC_SELF_INF_FLIP" "$TUNIX_DBC_SELF_INF_LOO" "$TUNIX_DTV_FLIP_DECISIONS_PATH" "$@" > "$CAPTURE"\nexit "${FAKE_EXIT:-0}"\n')
      downstream.chmod(0o755)
      (scripts / "save_results_to_my_result.py").write_text("pass\n")
      results = scripts / "my result"
      results.mkdir()
      (results / "plot_global_eval_rewards_sum.py").write_text("pass\n")
      capture = root / "args.txt"
      env = dict(os.environ, CAPTURE=str(capture), TUNIX_MY_RESULT_DIR=str(results))
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
          self.assertIn(f"dtv-flip_{score}_{scope}_", lines[2])
          self.assertIn("--use-dbc-self-inf-" + scope, lines)
          self.assertIn(str(root / "custom metrics"), lines)
          self.assertIn(str(root / "custom ckpt"), lines)
      result = subprocess.run(["bash", str(launcher), "dtv", "batch"],
                              env=dict(env, FAKE_EXIT="7"), capture_output=True)
      self.assertEqual(result.returncode, 7)
      for args in (["bad"], ["dtv", "invalid"],
                   ["dtv", "batch", "--use-dbc-self-inf-group"],
                   ["dtv", "batch", "--metrics-log-dir"]):
        result = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
  unittest.main()
