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
import json
import tempfile

from absl.testing import absltest


def _load_summary_module():
  module_path = (
      Path(__file__).resolve().parents[2]
      / "examples/dpo/summarize_qwen2p5_clean_tables.py"
  )
  spec = importlib.util.spec_from_file_location(
      "summarize_qwen2p5_clean_tables",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5CleanTablesTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.summary_mod = _load_summary_module()

  def test_build_table_rows_merges_clean_and_benchmark_metrics(self):
    clean_metrics = {
        "vanilla_dpo": {"clean_val_acc_auc": 0.5629},
        "random_pair_filtering": {"clean_val_acc_auc": 0.5596},
        "reward_based_filtering": {"clean_val_acc_auc": 0.5641},
        "self_inf": {"clean_val_acc_auc": 0.5659},
    }
    clean_test = {
        "vanilla_dpo": 0.5555,
        "random_pair_filtering": 0.5484,
        "reward_based_filtering": 0.5675,
        "self_inf": 0.5696,
    }
    benchmark_scores = {
        "vanilla_dpo": {
            "rewardbench_overall": 0.8123,
            "ifeval_prompt_strict": 0.4410,
            "mt_bench_avg_score": 7.12,
            "alpacaeval2_lc_win_rate": 31.4,
            "arena_hard_score": 28.2,
        },
        "random_pair_filtering": {
            "rewardbench_overall": 0.8012,
            "ifeval_prompt_strict": 0.4350,
            "mt_bench_avg_score": 7.05,
            "alpacaeval2_lc_win_rate": 30.1,
            "arena_hard_score": 27.8,
        },
        "reward_based_filtering": {
            "rewardbench_overall": 0.8299,
            "ifeval_prompt_strict": 0.4520,
            "mt_bench_avg_score": 7.21,
            "alpacaeval2_lc_win_rate": 32.3,
            "arena_hard_score": 29.0,
        },
        "self_inf": {
            "rewardbench_overall": 0.8344,
            "ifeval_prompt_strict": 0.4690,
            "mt_bench_avg_score": 7.34,
            "alpacaeval2_lc_win_rate": 33.8,
            "arena_hard_score": 30.7,
        },
    }

    rows = self.summary_mod.build_table_rows(
        clean_metrics=clean_metrics,
        clean_test_acc=clean_test,
        benchmark_scores=benchmark_scores,
    )

    self.assertLen(rows, 4)
    self.assertEqual(rows[0]["method"], "vanilla_dpo")
    self.assertEqual(rows[-1]["display_name"], "Self-inf (ours)")
    self.assertAlmostEqual(rows[-1]["arena_hard_score"], 30.7)
    self.assertAlmostEqual(rows[-1]["rewardbench_overall"], 0.8344)
    self.assertAlmostEqual(rows[-1]["ifeval_prompt_strict"], 0.4690)

  def test_render_latex_table_handles_missing_scores(self):
    rows = [
        {
            "method": "vanilla_dpo",
            "display_name": "Vanilla DPO",
            "clean_val_acc_auc": 0.5629,
            "clean_test_acc": 0.5555,
            "rewardbench_overall": None,
            "ifeval_prompt_strict": None,
            "mt_bench_avg_score": None,
            "alpacaeval2_lc_win_rate": None,
            "arena_hard_score": None,
        },
        {
            "method": "self_inf",
            "display_name": "Self-inf (ours)",
            "clean_val_acc_auc": 0.5659,
            "clean_test_acc": 0.5696,
            "rewardbench_overall": 0.8344,
            "ifeval_prompt_strict": 0.4690,
            "mt_bench_avg_score": 7.34,
            "alpacaeval2_lc_win_rate": 33.8,
            "arena_hard_score": 30.7,
        },
    ]

    latex = self.summary_mod.render_latex_table(
        rows,
        columns=(
            "clean_val_acc_auc",
            "clean_test_acc",
            "rewardbench_overall",
            "ifeval_prompt_strict",
        ),
    )

    self.assertIn("--", latex)
    self.assertIn("\\textbf{0.5659}", latex)
    self.assertIn("RewardBench Overall", latex)
    self.assertIn("IFEval Prompt Strict", latex)

  def test_load_clean_test_metrics_filters_clean_rows(self):
    payload = {
        "results": [
            {
                "variant": "vanilla_dpo",
                "dataset": "clean",
                "metrics": {"rewards_accuracy": 0.5555},
            },
            {
                "variant": "vanilla_dpo",
                "dataset": "global_flip20",
                "metrics": {"rewards_accuracy": 0.1234},
            },
        ]
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
      path = Path(tmp_dir) / "test_matrix.json"
      path.write_text(json.dumps(payload))
      metrics = self.summary_mod.load_clean_test_metrics(path)

    self.assertEqual(metrics, {"vanilla_dpo": 0.5555})

  def test_load_benchmark_scores_merges_rewardbench_and_ifeval(self):
    benchmark_payload = {
        "scores": {
            "self_inf": {
                "ifeval_prompt_strict": 0.469,
                "ifbench_prompt_loose": 0.337,
                "mt_bench_avg_score": 4.95,
            }
        }
    }
    rewardbench_payload = {
        "scores": {
            "self_inf": {
                "rewardbench_overall": 0.8344,
                "rewardbench_chat": 0.8123,
            }
        }
    }
    rewardbench_v2_payload = {
        "scores": {
            "self_inf": {
                "rewardbench2_macro_score": 0.6123,
                "rewardbench2_weighted_score": 0.5988,
            }
        }
    }
    livebench_payload = {
        "scores": {
            "self_inf": {
                "livebench_if_score": 12.5,
                "livebench_if_prompt_strict": 0.22,
            }
        }
    }
    safety_payload = {
        "scores": {
            "self_inf": {
                "xstest_overall_accuracy": 0.71,
                "harmbench_inverted_micro_asr_lower": 0.83,
            }
        }
    }
    wildbench_payload = {
        "scores": {
            "self_inf": {
                "wildbench_adjusted_score": 18.0,
                "wildbench_task_macro_score": 16.0,
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
      benchmark_path = Path(tmp_dir) / "benchmark_summary.json"
      rewardbench_path = Path(tmp_dir) / "rewardbench_summary.json"
      rewardbench_v2_path = Path(tmp_dir) / "rewardbench_v2_summary.json"
      livebench_path = Path(tmp_dir) / "livebench_summary.json"
      safety_path = Path(tmp_dir) / "safety_summary.json"
      wildbench_path = Path(tmp_dir) / "wildbench_summary.json"
      benchmark_path.write_text(json.dumps(benchmark_payload))
      rewardbench_path.write_text(json.dumps(rewardbench_payload))
      rewardbench_v2_path.write_text(json.dumps(rewardbench_v2_payload))
      livebench_path.write_text(json.dumps(livebench_payload))
      safety_path.write_text(json.dumps(safety_payload))
      wildbench_path.write_text(json.dumps(wildbench_payload))
      scores = self.summary_mod.load_benchmark_scores(
          benchmark_path,
          rewardbench_path,
          rewardbench_v2_path,
          livebench_path,
          safety_path,
          wildbench_path,
      )

    self.assertAlmostEqual(scores["self_inf"]["rewardbench_overall"], 0.8344)
    self.assertAlmostEqual(scores["self_inf"]["rewardbench_chat"], 0.8123)
    self.assertAlmostEqual(scores["self_inf"]["ifeval_prompt_strict"], 0.469)
    self.assertAlmostEqual(scores["self_inf"]["ifbench_prompt_loose"], 0.337)
    self.assertAlmostEqual(scores["self_inf"]["rewardbench2_macro_score"], 0.6123)
    self.assertAlmostEqual(scores["self_inf"]["livebench_if_score"], 12.5)
    self.assertAlmostEqual(scores["self_inf"]["xstest_overall_accuracy"], 0.71)
    self.assertAlmostEqual(
        scores["self_inf"]["harmbench_inverted_micro_asr_lower"], 0.83
    )
    self.assertAlmostEqual(scores["self_inf"]["wildbench_adjusted_score"], 18.0)


if __name__ == "__main__":
  absltest.main()
