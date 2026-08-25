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


def _load_rewardbench_module():
  module_path = (
      Path(__file__).resolve().parents[2]
      / "examples/dpo/eval_qwen2p5_rewardbench_v2.py"
  )
  spec = importlib.util.spec_from_file_location(
      "eval_qwen2p5_rewardbench_v2",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5RewardBenchV2Test(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mod = _load_rewardbench_module()

  def test_summarize_rewardbench_v2_rows_reports_macro_and_weighted_scores(self):
    scored_rows = [
        {
            "id": "0",
            "subset": "Factuality",
            "num_correct": 1,
            "scores": [0.4, 0.1, -0.3],
        },
        {
            "id": "1",
            "subset": "Factuality",
            "num_correct": 1,
            "scores": [-0.2, 0.5, 0.1],
        },
        {
            "id": "tied:0",
            "subset": "Ties",
            "num_correct": 2,
            "scores": [0.2, 0.1, -0.4],
        },
    ]

    summary = self.mod._summarize_rewardbench_v2_rows(  # pylint: disable=protected-access
        scored_rows=scored_rows,
    )

    self.assertAlmostEqual(summary["rewardbench2_factuality"], 0.5)
    self.assertAlmostEqual(summary["rewardbench2_ties"], 0.3)
    self.assertAlmostEqual(summary["rewardbench2_macro_score"], 0.4)
    self.assertAlmostEqual(summary["rewardbench2_weighted_score"], (0.5 * 2 + 0.3) / 3)


if __name__ == "__main__":
  absltest.main()
