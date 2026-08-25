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


def _load_benchmark_module():
  module_path = (
      Path(__file__).resolve().parents[2]
      / "examples/dpo/eval_qwen2p5_clean_benchmarks.py"
  )
  spec = importlib.util.spec_from_file_location(
      "eval_qwen2p5_clean_benchmarks",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5CleanBenchmarksTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mod = _load_benchmark_module()

  def test_zero_temperature_disables_top_p_sampling(self):
    generator = object.__new__(self.mod.TunixChatGenerator)
    generator._top_k = 50
    generator._top_p = 0.95

    temperature, top_k, top_p = generator._resolve_sampling_controls(
        temperature=0.0
    )

    self.assertEqual(temperature, 0.0)
    self.assertIsNone(top_k)
    self.assertIsNone(top_p)

  def test_positive_temperature_keeps_sampling_controls(self):
    generator = object.__new__(self.mod.TunixChatGenerator)
    generator._top_k = 50
    generator._top_p = 0.95

    temperature, top_k, top_p = generator._resolve_sampling_controls(
        temperature=0.7
    )

    self.assertEqual(temperature, 0.7)
    self.assertEqual(top_k, 50)
    self.assertEqual(top_p, 0.95)

  def test_normalize_benchmark_accepts_ifeval_aliases(self):
    self.assertEqual(self.mod._normalize_benchmark("ifeval"), "ifeval")  # pylint: disable=protected-access
    self.assertEqual(self.mod._normalize_benchmark("if_eval"), "ifeval")  # pylint: disable=protected-access
    self.assertEqual(self.mod._normalize_benchmark("ifbench"), "ifbench")  # pylint: disable=protected-access

  def test_compute_ifeval_scores_aggregates_prompt_and_instruction_accuracy(self):
    strict_rows = [
        {
            "follow_all_instructions": True,
            "follow_instruction_list": [True, True],
        },
        {
            "follow_all_instructions": False,
            "follow_instruction_list": [True, False],
        },
    ]
    loose_rows = [
        {
            "follow_all_instructions": True,
            "follow_instruction_list": [True, True],
        },
        {
            "follow_all_instructions": True,
            "follow_instruction_list": [True, True],
        },
    ]

    scores = self.mod._compute_ifeval_scores(strict_rows, loose_rows)  # pylint: disable=protected-access

    self.assertAlmostEqual(scores["ifeval_prompt_strict"], 0.5)
    self.assertAlmostEqual(scores["ifeval_instruction_strict"], 0.75)
    self.assertAlmostEqual(scores["ifeval_prompt_loose"], 1.0)
    self.assertAlmostEqual(scores["ifeval_instruction_loose"], 1.0)


if __name__ == "__main__":
  absltest.main()
