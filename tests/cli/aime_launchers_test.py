"""Exercise AIME shell configuration boundaries without TPU or JAX."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "runs_xuesong/scripts"


class AimeLaunchersTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    # Capture the actual seeded launcher's boundary to the TPU transport.
    scripts = self.root / "runs_xuesong/scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_official_like_dual_worker.sh").write_text(
        '#!/usr/bin/env bash\nexec "$CAPTURE_PYTHON" -c '
        + shlex.quote(
            "import json, os, sys; from pathlib import Path; "
            "Path(os.environ['CAPTURE_FILE']).write_text(json.dumps({"
            "'args': sys.argv[1:], 'run_name': os.environ['RUN_NAME'], "
            "'model_id': os.getenv('MODEL_ID'), "
            "'model_path': os.getenv('MODEL_PATH'), "
            "'num_batches': os.environ['NUM_BATCHES']}))"
        )
        + ' "$@"\n'
    )
    self.env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("TUNIX_", "AIME_", "T_"))
        and key not in ("NUM_BATCHES", "MAX_STEPS_OVERRIDE", "CHECKPOINT_INTERVAL",
                        "MODEL_ID", "MODEL_PATH", "TOKENIZER_PATH")
    }
    self.env.update(
        REPO=str(self.root),
        CAPTURE_PYTHON=sys.executable,
        CAPTURE_FILE=str(self.root / "capture.json"),
        TUNIX_RUN_TIMESTAMP="contract_test",
    )

  def train(self, script, method, *overrides, extra_env=None):
    env = dict(self.env, **(extra_env or {}))
    result = subprocess.run(
        ["bash", str(SCRIPTS / script), method, "0", *overrides],
        env=env, text=True, capture_output=True,
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    captured = json.loads((self.root / "capture.json").read_text())
    config = dict(arg.split("=", 1) for arg in captured["args"])
    return captured, config

  def test_1p5b_default_launch_and_run_name_are_preserved(self):
    captured, cfg = self.train("run_aime_seeded_full.sh", "baseline")
    self.assertEqual(captured["run_name"], "grpo_aime_baseline_seed0_clean_contract_test")
    self.assertEqual(captured["num_batches"], "64")
    self.assertEqual(cfg["agentic_grpo_config.num_generations"], "8")
    self.assertEqual(cfg["agentic_grpo_config.max_response_length"], "8192")
    self.assertNotIn("model_config.model_name", cfg)
    self.assertNotIn("rl_training_config.train_micro_batch_size", cfg)

  def test_7b_baseline_and_loo_share_full_training_preset(self):
    results = []
    for method in ("baseline", "group_loo_policy"):
      captured, cfg = self.train(
          "run_aime_7b_full.sh", method, extra_env={"MODEL_PATH": "/models/7b"}
      )
      self.assertTrue(captured["run_name"].startswith("grpo_aime_ds7b_full_"))
      self.assertEqual(captured["model_id"], "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
      self.assertEqual(captured["model_path"], "/models/7b")
      self.assertEqual(captured["num_batches"], "2")
      self.assertEqual(cfg["model_config.model_name"], "deepseek_r1_distill_qwen_7b")
      self.assertEqual(cfg["actor_model_config.lora_config"], "{}")
      self.assertEqual(cfg["rl_training_config.train_micro_batch_size"], "1")
      self.assertEqual(cfg["batch_size"], cfg["rl_training_config.mini_batch_size"])
      self.assertEqual(cfg["rl_training_config.actor_optimizer_config.decay_steps"], "314")
      results.append(cfg)
    variant = results[1].pop("rl_training_config.dynamic_batch_curation_variant")
    self.assertEqual(variant, "self_inf_group_loo_policy")
    results[0].pop("rl_training_config.checkpointing_options.save_interval_steps")
    results[1].pop("rl_training_config.checkpointing_options.save_interval_steps")
    self.assertEqual(results[0], results[1])

  def test_7b_formal_budget_overrides_preset(self):
    captured, cfg = self.train(
        "run_aime_7b_full.sh", "baseline", "batch_size=128",
        "rl_training_config.mini_batch_size=128",
        extra_env={"MODEL_PATH": "/models/7b", "NUM_BATCHES": "314"},
    )
    self.assertEqual(captured["num_batches"], "314")
    self.assertEqual(cfg["batch_size"], "128")
    self.assertEqual(cfg["rl_training_config.mini_batch_size"], "128")

  def test_7b_requires_explicit_weight_path(self):
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_aime_7b_full.sh"), "baseline", "0"],
        env=self.env, text=True, capture_output=True,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("Set MODEL_PATH", result.stderr)
    self.assertFalse((self.root / "capture.json").exists())

  def test_eval_default_and_7b_use_correct_model_without_launching(self):
    for script, expected in (
        ("run_aime_final_eval.sh", "deepseek_r1_distill_qwen_1p5b"),
        ("run_aime_7b_eval.sh", "deepseek_r1_distill_qwen_7b"),
    ):
      output = self.root / expected
      result = subprocess.run(
          ["bash", str(SCRIPTS / script), "--run-root", str(self.root),
           "--output-dir", str(output), "--dry-run"],
          env=dict(self.env, MODEL_PATH="/models/test"),
          text=True, capture_output=True,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      command = shlex.split(result.stdout)
      self.assertEqual(command[command.index("--model_config") + 1], expected)
      self.assertEqual(command[command.index("--max_generation_steps") + 1], "8192")
      self.assertFalse(output.exists())


if __name__ == "__main__":
  unittest.main()
