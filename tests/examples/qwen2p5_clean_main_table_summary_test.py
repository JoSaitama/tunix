# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import importlib.util

from absl.testing import absltest


def _load_summary_module():
  module_path = (
      Path(__file__).resolve().parents[2]
      / "examples/dpo/summarize_qwen2p5_clean_main_table.py"
  )
  spec = importlib.util.spec_from_file_location(
      "summarize_qwen2p5_clean_main_table",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5CleanMainTableSummaryTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.summary = _load_summary_module()

  def test_aggregate_rows_computes_mean_and_std(self):
    per_run = {
        "vanilla_dpo_seed0": {
            "logical_variant": "vanilla_dpo",
            "display_name": "Vanilla DPO",
            "seed": 0,
            "metrics": {"clean_test_acc": 0.50},
        },
        "vanilla_dpo_seed1": {
            "logical_variant": "vanilla_dpo",
            "display_name": "Vanilla DPO",
            "seed": 1,
            "metrics": {"clean_test_acc": 0.55},
        },
        "vanilla_dpo_seed2": {
            "logical_variant": "vanilla_dpo",
            "display_name": "Vanilla DPO",
            "seed": 2,
            "metrics": {"clean_test_acc": 0.60},
        },
        "self_inf_seed0": {
            "logical_variant": "self_inf",
            "display_name": "Self-DTV (ours)",
            "seed": 0,
            "metrics": {"clean_test_acc": 0.62},
        },
        "self_inf_seed1": {
            "logical_variant": "self_inf",
            "display_name": "Self-DTV (ours)",
            "seed": 1,
            "metrics": {"clean_test_acc": 0.63},
        },
        "self_inf_seed2": {
            "logical_variant": "self_inf",
            "display_name": "Self-DTV (ours)",
            "seed": 2,
            "metrics": {"clean_test_acc": 0.61},
        },
    }

    rows = self.summary.aggregate_rows(
        per_run_metrics=per_run,
        columns=("clean_test_acc",),
    )

    self.assertLen(rows, 2)
    vanilla_metric = rows[0]["metrics"]["clean_test_acc"]
    self.assertAlmostEqual(vanilla_metric["mean"], 0.55)
    self.assertAlmostEqual(vanilla_metric["std"], 0.0408248290, places=6)
    self.assertEqual(rows[1]["display_name"], "Self-DTV (ours)")
    self.assertEqual(rows[1]["seeds"], [0, 1, 2])

  def test_render_latex_table_bolds_best_mean(self):
    rows = [
        {
            "display_name": "Vanilla DPO",
            "metrics": {
                "clean_test_acc": {"mean": 0.55, "std": 0.01},
            },
        },
        {
            "display_name": "Self-DTV (ours)",
            "metrics": {
                "clean_test_acc": {"mean": 0.57, "std": 0.02},
            },
        },
    ]

    latex = self.summary.render_latex_table(
        rows,
        columns=("clean_test_acc",),
    )

    self.assertIn("\\textbf{0.5700 $\\pm$ 0.0200}", latex)
    self.assertIn("Test Acc", latex)


if __name__ == "__main__":
  absltest.main()
