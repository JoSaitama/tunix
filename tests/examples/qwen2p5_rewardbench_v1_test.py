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
      / "examples/dpo/eval_qwen2p5_rewardbench_v1.py"
  )
  spec = importlib.util.spec_from_file_location(
      "eval_qwen2p5_rewardbench_v1",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5RewardBenchV1Test(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mod = _load_rewardbench_module()

  def test_calculate_section_scores_uses_official_example_weights(self):
    subset_accuracies = {
        "alpacaeval-easy": 0.5,
        "alpacaeval-length": 0.6,
        "alpacaeval-hard": 0.7,
        "mt-bench-easy": 0.8,
        "mt-bench-med": 0.9,
        "math-prm": 0.25,
    }
    example_counts = {
        "alpacaeval-easy": 100,
        "alpacaeval-length": 95,
        "alpacaeval-hard": 95,
        "mt-bench-easy": 28,
        "mt-bench-med": 40,
        "math-prm": 984,
    }
    subset_mapping = {
        "Chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "Reasoning": ["math-prm"],
    }

    scores = self.mod._calculate_section_scores(  # pylint: disable=protected-access
        subset_accuracies=subset_accuracies,
        example_counts=example_counts,
        subset_mapping=subset_mapping,
    )

    self.assertAlmostEqual(
        scores["chat"],
        (
            0.5 * 100
            + 0.6 * 95
            + 0.7 * 95
            + 0.8 * 28
            + 0.9 * 40
        )
        / (100 + 95 + 95 + 28 + 40),
    )
    self.assertAlmostEqual(scores["reasoning"], 0.25)


if __name__ == "__main__":
  absltest.main()
