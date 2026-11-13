"""CPU checks for the filtering audit artifacts and actor capture contract."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from my_example.rebuttal import grpo_filter_audit as audit


class FilterAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sample(self, condition="mismatch0.2", method="dtv-loo"):
        collector = audit.FilterAudit(method, condition, 5, 1, self.root)
        collector.jax = SimpleNamespace(device_get=lambda x: x)
        collector.actor = SimpleNamespace(train_steps=0)
        flags = [(np.array([True, False, True, False]), np.array([True, False, False, False]))]
        metadata = collector.encode_rollouts(np.repeat(["p", "q", "r", "s"], 4),
                                             flags if condition != "clean" else [])
        advantages = np.tile([1., -1., 0., 0.], 4)
        keep = np.tile([True, False, True, True], 4)
        return collector, (metadata, advantages, keep, 4)

    def test_actor_capture_and_optimizer_boundary(self):
        collector, aux = self.sample()
        collector.capture(aux)
        self.assertEqual(len(collector.batches[0]["rows"]), 16)
        collector.actor.train_steps = 1
        collector.cluster = SimpleNamespace(global_steps=1)
        collector.actor.optimizer = SimpleNamespace(opt_state=SimpleNamespace(gradient_step=1), step=1)
        self.assertTrue(collector.after_iteration())
        self.assertEqual(collector.optimizer_steps(), 1)
        collector.actor.optimizer.step = 0
        with self.assertRaisesRegex(ValueError, "NNX and Optax"):
            collector.optimizer_steps()

    def test_reject_row_permutation_and_mask_count(self):
        collector, (metadata, adv, keep, skipped) = self.sample()
        with self.assertRaisesRegex(ValueError, "row IDs"):
            collector.capture((metadata[::-1], adv, keep, skipped))
        with self.assertRaisesRegex(ValueError, "skipped_samples"):
            collector.capture((metadata, adv, keep, 0))

    def test_reject_wrong_noise_capture_and_retention_floor(self):
        collector, (metadata, adv, keep, _) = self.sample()
        with self.assertRaisesRegex(ValueError, "group noise flags"):
            collector.encode_rollouts(np.repeat(["p", "q", "r", "s"], 4), [
                (np.zeros(4, dtype=bool), np.ones(4, dtype=bool)),
            ])
        keep[:4] = False
        with self.assertRaisesRegex(ValueError, "LOO minimum retention"):
            collector.capture((metadata, adv, keep, int((~keep).sum())))

    def test_conditional_denominators_and_empty_strata(self):
        collector, aux = self.sample()
        collector.capture(aux)
        cells = {(x["group"], x["advantage"]): (x["dropped"], x["total"], x["drop_rate"])
                 for x in audit.summarize(collector.batches[0]["rows"])}
        self.assertEqual(cells["all", "all"], (4, 16, .25))
        self.assertEqual(cells["assigned", "negative"], (2, 2, 1.))
        self.assertEqual(cells["unassigned", "zero"], (0, 4, 0.))
        self.assertEqual(cells["effective", "all"], (1, 4, .25))
        collector, aux = self.sample("clean")
        collector.capture(aux)
        self.assertTrue(all(x["drop_rate"] is None for x in audit.summarize(collector.batches[0]["rows"])
                            if x["group"] == "assigned"))

    def test_suite_and_prompt_matching(self):
        for condition in audit.CONDITIONS:
            for method in audit.METHODS:
                collector, aux = self.sample(condition, method)
                collector.capture(aux)
                path = self.root / f"{condition}-{method}"
                path.mkdir()
                (path / "decisions.jsonl").write_text(json.dumps(collector.batches[0]) + "\n")
                audit.write_json(path / "run.json", {
                    "schema": audit.SCHEMA, "scope": "early_training", "method": method,
                    "condition": condition, "seed": 5, "noise_fraction": collector.fraction,
                    "noise_seed": 5, "shuffle_seed": 47, "optimizer_updates": 1,
                    "schedule_updates": 691, "commit": "same", "model_revision": "same",
                    "jax_version": "same", "devices": ["same"] * 4, "initial_lora_sha256": "same",
                })
        self.assertEqual(len(audit.read_suite(self.root, smoke=True)), 4)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            audit.read_suite(self.root)
        path = self.root / "clean-dtv" / "decisions.jsonl"
        data = json.loads(path.read_text())
        for row in data["rows"][:4]:
            row["prompt_id"] = "0" * 64
        path.write_text(json.dumps(data) + "\n")
        with self.assertRaisesRegex(ValueError, "prompt order"):
            audit.read_suite(self.root, smoke=True)

    def test_launcher_dry_run_matrix(self):
        script = Path(audit.__file__).with_name("run_grpo_filter_audit.sh")
        destination = self.root / "must-not-exist"
        for mode, expected in (("smoke", 4), ("full", 20)):
            plan = subprocess.check_output(["bash", script, mode, destination, "--dry-run"], text=True)
            self.assertEqual(plan.count("--skip-eval-before --skip-eval-after"), expected)
            self.assertEqual(plan.count("fraction=0.2"), expected // 2)
            self.assertEqual(plan.count(f"updates={1 if mode == 'smoke' else 32}"), expected)
        self.assertFalse(destination.exists())

    def test_actual_actor_jit_masks_and_update_match_unaudited_paths(self):
        from flax import nnx, struct
        import jax.numpy as jnp
        import optax

        class Model(nnx.Module):
            def __init__(self):
                self.weight = nnx.Param(jnp.asarray(2.))

        @struct.dataclass
        class Example:
            advantages: object
            filter_audit_metadata: object

        # Load the real trainer kernels without the unrelated metrax stack.
        package, rl = ModuleType("tunix"), ModuleType("tunix.rl")
        package.rl = rl
        rl.trainer = SimpleNamespace(Trainer=object)
        repo = Path(__file__).resolve().parents[2]
        for method, module_name, cls in (
            ("dtv", "self_inf_trainer", "SelfInfTrainer"),
            ("dtv-loo", "self_inf_loo_trainer", "SelfInfLooTrainer"),
        ):
            with self.subTest(method=method):
                spec = importlib.util.spec_from_file_location(module_name, repo / "tunix/rl" / f"{module_name}.py")
                module = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, {"tunix": package, "tunix.rl": rl}):
                    spec.loader.exec_module(module)
                trainer = object.__new__(getattr(module, cls))
                trainer.gen_model_input_fn = lambda x: {"train_example": x}
                trainer._lora_enabled = False
                trainer._has_aux = trainer._score_loss_has_aux = True
                trainer.scope, trainer.num_generations = "group", 4
                trainer.dot_threshold, trainer.min_keep_fraction = 0., .25
                trainer._score_loss_fn = lambda model, train_example: (
                    model.weight.get_value() * train_example.advantages, {}
                )
                trainer.loss_fn = lambda model, train_example: (
                    model.weight.get_value() * (3 + train_example.advantages), {}
                )
                inputs = Example(jnp.array([3., -1., -1., -1.]), jnp.arange(4))
                # Zero mean in DTV keeps all scores; every LOO score is
                # negative, so the final LOO mask rescues the first tie among
                # the three lower-magnitude gradients (index 1).
                expected = [True] * 4 if method == "dtv" else [False, True, False, False]
                plain_model, audited_model = Model(), Model()
                plain_opt = nnx.Optimizer(plain_model, optax.sgd(.01), wrt=nnx.Param)
                audited_opt = nnx.Optimizer(audited_model, optax.sgd(.01), wrt=nnx.Param)
                plain_loss, plain = nnx.jit(trainer._train_step)(plain_model, plain_opt, inputs)
                trainer._filter_audit_enabled = True
                loss, (metadata, adv, keep, skipped) = nnx.jit(trainer._train_step)(
                    audited_model, audited_opt, inputs
                )
                np.testing.assert_array_equal(metadata, inputs.filter_audit_metadata)
                np.testing.assert_array_equal(adv, inputs.advantages)
                np.testing.assert_array_equal(keep, expected)
                self.assertEqual(int(skipped), int(plain["skipped_samples"]))
                np.testing.assert_allclose(loss, plain_loss)
                np.testing.assert_allclose(audited_model.weight.get_value(), plain_model.weight.get_value())


if __name__ == "__main__":
    unittest.main()
