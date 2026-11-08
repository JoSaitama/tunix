"""Standard-library preflight tests. No JAX, TPU, or model initialization."""

import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from my_example import audit_grpo_checkpoint_restore as audit


class CheckpointPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "inputs"
        self.step = self.source / "checkpoints/actor/500"
        params = self.step / "model_params"
        (params / "d").mkdir(parents=True)
        (params / "d/shard").write_bytes(b"fake payload; not a real checkpoint")
        (self.step / "_CHECKPOINT_METADATA").write_text("{}")
        (params / "_METADATA").write_text(json.dumps({"tree_metadata": {"weight": {}}}))
        (self.source / "git-commit.txt").write_text("a" * 40 + "\n")
        (self.source / "pip-freeze.txt").write_text("jax==0.8.1\nflax==0.12.6\n")

    def test_complete_preflight_and_read_only_hash(self):
        before = audit.fingerprint_directory(self.step)
        report = audit.metadata_preflight(self.source, self.step, "a" * 40)
        self.assertEqual(report["input_fingerprint"], before)
        self.assertEqual(report["parameter_metadata_entry_count"], 1)
        self.assertEqual(report["recorded_package_versions"]["jax"], "0.8.1")
        self.assertEqual(audit.fingerprint_directory(self.step), before)

    def test_missing_payload_rejected(self):
        (self.step / "model_params/d/shard").unlink()
        with self.assertRaisesRegex(ValueError, "payload"):
            audit.metadata_preflight(self.source, self.step)

    def test_missing_metadata_rejected(self):
        (self.step / "model_params/_METADATA").unlink()
        with self.assertRaisesRegex(ValueError, "required"):
            audit.metadata_preflight(self.source, self.step)

    def test_bad_json_rejected(self):
        (self.step / "model_params/_METADATA").write_text("not json")
        with self.assertRaises(json.JSONDecodeError):
            audit.metadata_preflight(self.source, self.step)

    def test_commit_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "commit mismatch"):
            audit.metadata_preflight(self.source, self.step, "b" * 40)

    def test_malformed_commit_rejected(self):
        (self.source / "git-commit.txt").write_text("not a commit")
        with self.assertRaisesRegex(ValueError, "full commit"):
            audit.metadata_preflight(self.source, self.step)

    def test_output_overlap_and_existing_output_rejected(self):
        for output in (self.source, self.source / "reports", self.root):
            with self.assertRaisesRegex(ValueError, "separate"):
                audit.validate_paths(self.source, output, 500)
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            audit.validate_paths(self.source, output, 500)

    def test_missing_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            audit.validate_paths(self.source, self.root / "out", 123)

    def test_source_digest_detects_mutation(self):
        before = audit.fingerprint_directory(self.step)
        (self.step / "model_params/d/shard").write_bytes(b"changed")
        self.assertNotEqual(before, audit.fingerprint_directory(self.step))

    def test_symlink_rejected(self):
        (self.step / "linked").symlink_to(self.source / "pip-freeze.txt")
        with self.assertRaisesRegex(ValueError, "symlink"):
            audit.fingerprint_directory(self.step)

    def test_metadata_cli_has_no_jax_import_or_utility_claim(self):
        output = self.root / "report"
        command = (
            "import sys; from my_example.audit_grpo_checkpoint_restore import main; "
            "main(sys.argv[1:]); "
            "assert 'jax' not in sys.modules; assert 'flax' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", command, "--input-dir", str(self.source),
             "--report-dir", str(output), "--expected-commit", "a" * 40],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = audit.read_json(output / "restore_report.json")
        self.assertTrue(report["passed"])
        self.assertEqual(report["mode"], "metadata")
        self.assertTrue(report["source_unchanged"])
        self.assertFalse(report["optimizer_restored"])
        self.assertFalse(report["utility_audit_completed"])

    def test_failure_writes_failed_report(self):
        output = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "commit mismatch"):
            audit.main(["--input-dir", str(self.source), "--report-dir", str(output),
                        "--expected-commit", "b" * 40])
        report = audit.read_json(output / "restore_report.json")
        self.assertFalse(report["passed"])
        self.assertEqual(report["error"]["type"], "ValueError")

    def test_lora_dummy_forward_not_wrapped_in_fsdp_mesh(self):
        # Static regression for the TPU-only 2-vs-4 activation failure. A CPU
        # Gemma run would bypass shard() and therefore not reproduce that bug.
        tree = ast.parse(inspect.getsource(audit.restore_checkpoint))
        lora_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "apply_lora"]
        self.assertEqual(len(lora_calls), 1)
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            has_mesh = any(isinstance(item.context_expr, ast.Name)
                           and item.context_expr.id == "mesh" for item in node.items)
            if has_mesh:
                self.assertNotIn(lora_calls[0], list(ast.walk(node)))
        mesh_keywords = [item for item in lora_calls[0].keywords if item.arg == "mesh"]
        self.assertEqual(len(mesh_keywords), 1)
        self.assertIsInstance(mesh_keywords[0].value, ast.Name)
        self.assertEqual(mesh_keywords[0].value.id, "mesh")


if __name__ == "__main__":
    unittest.main()
