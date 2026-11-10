"""Numerical oracle and actor update tests for the groupwise baselines."""

import ast
import contextlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import optimize

ROOT = Path(__file__).resolve().parents[2]


def load_modules():
  names = ("pcgrad", "cagrad", "gradient_conflict_trainer")
  if os.environ.get("TUNIX_CONFLICT_TEST_ISOLATED") != "1":
    return tuple(importlib.import_module(f"tunix.rl.{name}") for name in names)
  # Explicit local CPU mode skips only the base framework import. Autodiff,
  # sharding, NNX optimizer, and the production algorithm code remain real.
  package = types.ModuleType("tunix")
  package.__path__ = [str(ROOT / "tunix")]
  rl = types.ModuleType("tunix.rl")
  rl.__path__ = [str(ROOT / "tunix" / "rl")]
  rl.trainer = types.SimpleNamespace(Trainer=object)
  with mock.patch.dict(sys.modules, {"tunix": package, "tunix.rl": rl}):
    return tuple(importlib.import_module(f"tunix.rl.{name}") for name in names)


pcgrad, cagrad, trainer_module = load_modules()


def reference_cagrad(gradients, c):
  """Independent parameter-space constrained optimization oracle in FP64."""
  center = gradients.mean(axis=0)
  radius_squared = (c * np.linalg.norm(center)) ** 2
  result = optimize.minimize(
      lambda z: -z[-1], np.r_[center, np.min(gradients @ center)],
      jac=lambda z: np.r_[np.zeros_like(center), -1.],
      constraints=[
          {"type": "ineq",
           "fun": lambda z: radius_squared - np.sum((z[:-1] - center)**2),
           "jac": lambda z: np.r_[-2 * (z[:-1] - center), 0.]},
          {"type": "ineq", "fun": lambda z: gradients @ z[:-1] - z[-1],
           "jac": lambda z: np.column_stack((gradients, -np.ones(len(gradients))))},
      ], method="SLSQP", options={"ftol": 1e-9, "maxiter": 300},
  )
  if not result.success:
    raise AssertionError(result.message)
  return result.x[-1]


class AlgorithmTest(unittest.TestCase):
  def test_pcgrad_matches_direct_vector_projection_with_fixed_original_references(self):
    original = np.array([[1., 1., 0.], [-1., 0., 2.], [0., -2., -1.], [0., 0., 0.]])
    solve = jax.jit(pcgrad.coefficients)
    for seed in (0, 5, 21):
      key = jax.random.key(seed)
      expected, count = [], 0
      for i, row_key in enumerate(jax.random.split(key, 4)):
        current = original[i].copy()
        others = (i + np.arange(1, 4)) % 4
        for j in np.asarray(jax.random.permutation(row_key, others)):
          dot = current @ original[j]
          if dot < 0:
            current -= dot / (original[j] @ original[j]) * original[j]
            count += 1
        expected.append(current)
      weights, actual_count = solve(jnp.asarray(original @ original.T), key)
      np.testing.assert_allclose(np.asarray(weights) @ original,
                                 np.mean(expected, axis=0), atol=2e-6)
      self.assertEqual(int(actual_count), count)

  def test_pcgrad_no_conflict_reproduces_mean(self):
    weights, count = pcgrad.coefficients(jnp.eye(4), jax.random.key(0))
    np.testing.assert_array_equal(weights, np.full(4, .25))
    self.assertEqual(int(count), 0)

  def test_cagrad_matches_primal_oracle_on_random_and_degenerate_groups(self):
    rng = np.random.default_rng(5)
    cases = [rng.normal(size=(4, 5)) for _ in range(16)]
    cases += [
        np.array([[1., 0.], [1., 0.], [-.5, 1.], [0., 0.]]),
        np.array([[1., 0.], [-1., 0.], [2., 0.], [2., 0.]]),
        np.array([[0., 0.], [-1., 0.], [2., 1.], [2., 1.]]),
    ]
    for c in (.1, .5, .9):
      solve = jax.jit(lambda gram: cagrad.coefficients(gram, c))
      for i, gradients in enumerate(cases):
        with self.subTest(c=c, case=i):
          weights = np.asarray(solve(jnp.asarray(gradients @ gradients.T)))
          direction = weights @ gradients
          center = gradients.mean(axis=0)
          self.assertLessEqual(np.linalg.norm(direction - center),
                               c * np.linalg.norm(center) + 3e-6)
          expected = reference_cagrad(gradients, c)
          np.testing.assert_allclose(np.min(gradients @ direction), expected, atol=3e-5)

  def test_cagrad_zero_radius_and_zero_mean(self):
    for gram, c in ((jnp.eye(4), 0.), (jnp.zeros((4, 4)), .5)):
      np.testing.assert_array_equal(cagrad.coefficients(gram, c), np.full(4, .25))
    gradients = jnp.array([[1., 2.], [-1., -2.], [3., 4.], [-3., -4.]])
    weights = cagrad.coefficients(gradients @ gradients.T, .5)
    np.testing.assert_allclose(weights @ gradients, [0., 0.], atol=1e-7)

  def test_cagrad_resolves_weak_active_gradient(self):
    # The smallest row is active at the optimum. Projecting from the large
    # row on that same face loses its improving component in FP32.
    gradients = np.random.default_rng(44).normal(size=(9, 4, 5))[-1]
    gradients[3] *= 1e-4
    gradients /= np.linalg.norm(gradients, axis=1).max()
    weights = np.asarray(cagrad.coefficients(jnp.asarray(gradients @ gradients.T), .5))
    achieved = np.min(gradients @ (weights @ gradients))
    np.testing.assert_allclose(
        achieved, reference_cagrad(gradients, .5), atol=1e-7
    )


class Model(nnx.Module):
  def __init__(self):
    self.adapter = nnx.LoRAParam(jnp.zeros(2))
    self.frozen = nnx.Param(jnp.array(1.0))


def make_trainer(method, include_kl, c=.5):
  trainer = object.__new__(trainer_module.GradientConflictTrainer)
  trainer.method, trainer.include_kl, trainer.c = method, include_kl, c
  trainer.seed, trainer._initial_iter_step = 5, 0
  trainer._has_aux, trainer._policy_has_aux, trainer._lora_enabled = True, True, True
  trainer.gen_model_input_fn = lambda x: x

  def policy_loss(model, policy, kl):
    del kl
    return jnp.mean(policy @ model.adapter[...]) + model.frozen[...], {"kl": 0.}

  def full_loss(model, policy, kl):
    k = jnp.mean(kl @ model.adapter[...])
    return policy_loss(model, policy, kl)[0] + k, {"kl": k}

  trainer.loss_fn, trainer._policy_loss = full_loss, policy_loss
  return trainer


class IntegrationTest(unittest.TestCase):
  def test_production_route_is_opt_in_and_validates_actual_group_size(self):
    from my_example.config import config_from_args

    source = (ROOT / "tunix/rl/rl_cluster.py").read_text()
    start = source.index("    actor_trainer_kwargs = {}")
    end = source.index(
        "    if actor_config.use_dynamic_batch_curation and actor_trainer_cls is robust_trainer",
        start,
    )
    block = ast.parse(textwrap.dedent(source[start:end]))
    for node in ast.walk(block):
      if isinstance(node, ast.If):
        node.body = [line for line in node.body if not isinstance(line, ast.ImportFrom)]
    route = compile(block, "actor_trainer_selection", "exec")
    common = {
        "gradient_conflict_trainer": trainer_module,
        "os": os,
        "logging": types.SimpleNamespace(info=lambda *args: None),
        "rl_trainer": types.SimpleNamespace(Trainer="vanilla"),
        "robust_trainer": types.SimpleNamespace(RobustTrainer="robust"),
    }
    for environment, expected in (
        ({}, "robust"),
        ({"TUNIX_DBC_VARIANT": "gradient_conflict",
          "TUNIX_GRADIENT_CONFLICT_METHOD": "pcgrad"},
         trainer_module.GradientConflictTrainer),
    ):
      with mock.patch.dict(os.environ, environment, clear=True):
        args = ["--num-generations", "4"]
        baseline = config_from_args(args)
        state = dict(common, actor_config=baseline.training)
        exec(route, state)
        self.assertEqual(state["actor_trainer_cls"], "vanilla")
        opt_in = config_from_args([*args, "--use-dynamic-batch-curation"])
        state["actor_config"] = opt_in.training
        exec(route, state)
        self.assertEqual(state["actor_trainer_cls"], expected)
        if expected == trainer_module.GradientConflictTrainer:
          self.assertFalse(state["actor_trainer_kwargs"]["include_kl"])
          with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            config_from_args(["--num-generations", "8", "--use-dynamic-batch-curation"])

  def test_pcgrad_kl_modes_change_optimizer_input_without_double_counting(self):
    inputs = {"policy": jnp.array([[1., 1.], [-1., 0.], [0., 0.], [0., 0.]]),
              "kl": jnp.array([[-1., .2], [1., .2], [0., .2], [0., .2]])}
    for include_kl, expected, projections in (
        (False, [-.125, .575], 2), (True, [0., .45], 0)):
      with self.subTest(include_kl=include_kl):
        trainer = make_trainer("pcgrad", include_kl)
        model = Model()
        optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.LoRAParam)
        _, aux = nnx.jit(trainer._train_step)(model, optimizer, inputs)
        np.testing.assert_allclose(model.adapter[...], -.1 * np.asarray(expected), atol=2e-7)
        self.assertEqual(float(model.frozen[...]), 1.)
        self.assertEqual(int(aux["pcgrad_projections"]), projections)
        self.assertEqual(int(aux["skipped_samples"]), 0)

  @unittest.skipUnless(jax.local_device_count() == 4, "requires four CPU devices")
  def test_cagrad_sharded_groups_and_original_batch_scale_at_c_zero(self):
    mesh = jax.sharding.Mesh(np.array(jax.local_devices()), ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    policy = np.array([[1., 1.], [-2., 0.], [.4, -1.], [.3, .1]] * 4, dtype=np.float32)
    policy[4:8] *= 2.
    kl = np.broadcast_to(np.array([.1, .2], dtype=np.float32), (16, 2))
    inputs = {"policy": jax.device_put(policy, sharding), "kl": jax.device_put(kl, sharding)}
    for c, include_kl in ((0., False), (0., True), (.5, False), (.5, True)):
      with self.subTest(c=c, include_kl=include_kl):
        trainer = make_trainer("cagrad", include_kl, c)
        model = Model()
        optimizer = nnx.Optimizer(model, optax.sgd(.1), wrt=nnx.LoRAParam)
        _, aux = nnx.jit(trainer._train_step)(model, optimizer, inputs)
        if c == 0:
          expected = (policy + kl).mean(axis=0)
          self.assertEqual(float(aux["gradient_conflict_shift_norm"]), 0.)
        else:
          vectors = policy + kl if include_kl else policy
          directions = [np.asarray(cagrad.coefficients(jnp.asarray(g @ g.T), c)) @ g
                        for g in vectors.reshape(-1, 4, 2)]
          expected = np.mean(directions, axis=0) + (0 if include_kl else kl.mean(axis=0))
        np.testing.assert_allclose(model.adapter[...], -.1 * expected, atol=2e-7)

  def test_launcher_routes_modes_and_preserves_baseline_defaults(self):
    launcher = ROOT / "my_example/rebuttal/run_gradient_conflict_baseline.sh"
    for method, flags, mode in (("pcgrad", [], "0"),
                                ("cagrad", ["--include-kl", "--cagrad-c", "0.4"], "1")):
      result = subprocess.run(["bash", str(launcher), method, *flags, "--dry-run"],
                              cwd=ROOT, capture_output=True, text=True, check=True)
      self.assertIn(f"TUNIX_GRADIENT_CONFLICT_METHOD={method}", result.stdout)
      self.assertIn(f"TUNIX_GRADIENT_CONFLICT_INCLUDE_KL={mode}", result.stdout)
      self.assertIn("--num-generations 4", result.stdout)
      self.assertIn("--max-eval-examples 1319", result.stdout)
      self.assertIn("--learning-rate 1e-6", result.stdout)
    for args in (["cagrad", "--cagrad-c", "1"],
                 ["pcgrad", "--num-generations=8"],
                 ["pcgrad", "--use-dbc-self-inf-group"]):
      result = subprocess.run(["bash", str(launcher), *args, "--dry-run"],
                              cwd=ROOT, capture_output=True, check=False)
      self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
  unittest.main()
