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


def _load_module():
  module_path = (
      Path(__file__).resolve().parents[2]
      / "examples/dpo/eval_qwen2p5_wildbench.py"
  )
  spec = importlib.util.spec_from_file_location(
      "eval_qwen2p5_wildbench",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5WildBenchTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mod = _load_module()

  def test_summarize_completed_scores_matches_official_scaling(self):
    rows = [
        {"session_id": "s0", "primary_tag": "Editing"},
        {"session_id": "s1", "primary_tag": "Editing"},
        {"session_id": "s2", "primary_tag": "Coding & Debugging"},
    ]
    tempdir = Path(self.create_tempdir().full_path)
    result_path = tempdir / "wildbench.json"
    result_path.write_text(
        """
[
  {"session_id": "s0", "score": 6, "model_output": "aaaa"},
  {"session_id": "s1", "score": 7, "model_output": "bbbbbb"},
  {"session_id": "s2", "score": 5, "model_output": "cc"}
]
"""
    )

    summary = self.mod._summarize_completed_scores(rows, result_path)  # pylint: disable=protected-access

    self.assertAlmostEqual(summary["wildbench_raw_score"], 6.0)
    self.assertAlmostEqual(summary["wildbench_adjusted_score"], 20.0)
    self.assertAlmostEqual(summary["wildbench_task_macro_score"], 15.0)
    self.assertAlmostEqual(summary["wildbench_avg_length"], 4.0)


if __name__ == "__main__":
  absltest.main()
