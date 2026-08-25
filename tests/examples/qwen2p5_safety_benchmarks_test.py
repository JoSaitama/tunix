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
      / "examples/dpo/score_safety_generation.py"
  )
  spec = importlib.util.spec_from_file_location(
      "score_safety_generation",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5SafetyBenchmarksTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mod = _load_module()

  def test_load_rows_supports_jsonl(self):
    tempdir = Path(self.create_tempdir().full_path)
    path = tempdir / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')

    rows = self.mod._load_rows(path)  # pylint: disable=protected-access

    self.assertLen(rows, 2)
    self.assertEqual(rows[1]["a"], 2)


if __name__ == "__main__":
  absltest.main()
