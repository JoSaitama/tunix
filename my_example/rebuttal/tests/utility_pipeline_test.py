"""Server CPU orchestration tests with an explicit NumPy-only fake runtime.

These exercise report plumbing, NOT model restoration, gradients or TPU memory.
The actual runtime is covered separately by tiny JAX tests and server TPU gates.
"""

from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from my_example.rebuttal.audit_grpo_utility import main, parser, run, validate
from my_example.rebuttal.utility_math import scores_from_gram


class FakeRuntime:
    def __init__(self, args, checkpoint, output):
        self.args, self.output = args, output
        self.mesh = nullcontext()
        self.jax = SimpleNamespace(device_get=lambda x: x, block_until_ready=lambda x: x)
        self.jnp = np
        self.actor = np.zeros(2)
        self.nnx = SimpleNamespace(state=lambda actor, _: actor, LoRAParam=object)
        self._snapshot = SimpleNamespace(bytes=8)
        self.initial_summary = {"sha256": "fake-runtime-only"}
        self.parameter_summary = lambda jax, state: self.initial_summary
        self.model_path = "/fake/no-model"
        self.topology = {"explicit_fake_runtime": True}
        self.gradient_calls = 0
        self.snapshot_memory("fixture_only")

    def snapshot_memory(self, name):
        (self.output / "hbm_report.json").write_text(json.dumps(
            {"headroom_verified": True, "scope": "fake fixture, NOT TPU memory"}))

    def assert_original_parameters(self, stage):
        return {"stage": stage, "bitwise_equal": True, "explicit_fake_runtime": True}

    def prepare(self, examples, *, role, batch_id):
        return {"correct": np.tile([True, False, False, False], 4),
                "advantages": np.tile([1., .5, -.5, -1.], 4),
                "clean_rewards": np.tile([3., 2., 1., 0.], 4),
                "observed_rewards": np.tile([3., 2., 1., 0.], 4),
                "noise_selected": np.zeros(4, dtype=bool), "noise_effective": np.zeros(4, dtype=bool),
                "source_indices": np.repeat([x["source_index"] for x in examples], 4),
                "prompt_ids_hash": np.repeat([x["prompts"] for x in examples], 4),
                "completion_mask": np.ones((16, 2), dtype=np.int32)}

    def heldout_gradient(self, heldout):
        return np.array([1., 0.]), {"l2_norm": 1., "informative": True,
            "split_half": {"available": True, "l2_norms": [1., 1.], "cosine": 1., "positive_alignment": True}}

    def clean_value(self, heldout):
        return 0.

    def gradients(self, host, full=False):
        self.gradient_calls += 1
        value = np.tile([[2., 1.], [2., -1.], [2., 0.], [-1., 3.]], (4, 1))
        return value + .01 if full else value

    def batch_statistics(self, pg, full, h):
        gram = pg @ pg.T
        score = scores_from_gram(gram)
        return {"gram": gram, "score_dtv": score["dtv"], "score_loo": score["loo"],
                "utility_policy": pg @ h, "utility_full": full @ h,
                "utility_half_a": pg @ h, "utility_half_b": pg @ h,
                "self_term": score["self"], "cross_term": score["cross"],
                "reference_norm_squared": np.square(pg.reshape(4, 4, 2).mean(1)).sum(1)}

    def average(self, value, mask):
        return (value * mask[:, None]).sum(0) / max(mask.sum(), 1)

    def direction(self, pg, full, mask):
        return self.average(full, mask)

    def dot(self, a, b):
        return a @ b

    def add(self, a, b):
        return a + b

    def scale(self, a, b):
        return a * b

    def numerical_preflight(self, *unused):
        report = {"passed": False, "validated_step_sizes": []}
        (self.output / "numerical_report.json").write_text(json.dumps(report))
        return report


class UtilityPipelineTest(unittest.TestCase):
    def setUp(self):
        # run() enforces TPU environment for real model modes. The fake must
        # not leak that environment into subsequent CPU JAX test cases.
        patch = mock.patch.dict(os.environ, {}, clear=False)
        patch.start()
        self.addCleanup(patch.stop)
        capture = redirect_stdout(io.StringIO())
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def source(self, root):
        source = root / "source"
        params = source / "checkpoints/actor/500/model_params"
        (params / "d").mkdir(parents=True)
        (params / "d/payload").write_bytes(b"metadata fixture only")
        (params / "_METADATA").write_text("{}")
        (params.parent / "_CHECKPOINT_METADATA").write_text("{}")
        (source / "git-commit.txt").write_text("a" * 40)
        return source

    def argv(self, source, output, mode):
        return ["--input-dir", str(source), "--output-dir", str(output), "--expected-commit", "a" * 40,
                "--reward-mode", "accuracy", "--mode", mode, "--pool-prompts", "16", "--train-fraction", ".5",
                "--candidate-prompts", "4", "--heldout-prompts", "8", "--bootstrap-repeats", "0"]

    def partition(self, args):
        candidates = [dict(source_index=i, prompts=f"p{i}") for i in range(4)]
        heldout = [dict(source_index=i, prompts=f"p{i}") for i in range(8, 16)]
        return candidates, heldout, {"candidate_indices": list(range(4)), "heldout_indices": list(range(8, 16))}

    def test_score_outputs_evidence_without_any_finite_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            with mock.patch("my_example.rebuttal.utility_runtime.UtilityRuntime", FakeRuntime), \
                 mock.patch("my_example.rebuttal.audit_grpo_utility.select_examples", self.partition):
                output = run(validate(parser().parse_args(self.argv(source, root / "out", "score"))))
            summary = json.loads((output / "audit_summary.json").read_text())
            self.assertTrue(summary["execution_passed"])
            self.assertTrue(summary["utility_audit_completed"])
            self.assertFalse(summary["finite_step_completed"])
            self.assertFalse(summary["finite_step_validated"])
            self.assertFalse(summary["assumption_evidence"]["assumption_proven"])
            self.assertTrue((output / "filter_strata.csv").is_file())
            self.assertTrue((output / "assumption_report.json").is_file())
            self.assertFalse((output / "batch_utility.csv").exists())

    def test_smoke_numerical_failure_skips_method_probes_but_preserves_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            with mock.patch("my_example.rebuttal.utility_runtime.UtilityRuntime", FakeRuntime), \
                 mock.patch("my_example.rebuttal.audit_grpo_utility.select_examples", self.partition):
                output = run(validate(parser().parse_args(self.argv(source, root / "out", "smoke"))))
            summary = json.loads((output / "audit_summary.json").read_text())
            self.assertTrue(summary["execution_passed"])
            self.assertFalse(summary["numerical_check"]["passed"])
            self.assertFalse(summary["finite_step_completed"])
            self.assertTrue(summary["units"]["dtv"]["units"] == 16)

    def test_numerics_exit_two_is_distinct_from_execution_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.source(root)
            with mock.patch("my_example.rebuttal.utility_runtime.UtilityRuntime", FakeRuntime), \
                 mock.patch("my_example.rebuttal.audit_grpo_utility.select_examples", self.partition):
                with self.assertRaises(SystemExit) as caught:
                    main(self.argv(source, root / "out", "numerics"))
            self.assertEqual(caught.exception.code, 2)
            summary = json.loads((root / "out/audit_summary.json").read_text())
            self.assertTrue(summary["execution_passed"])
            self.assertFalse(summary["numerical_check"]["passed"])
            self.assertFalse(summary["utility_audit_completed"])


if __name__ == "__main__":
    unittest.main()
