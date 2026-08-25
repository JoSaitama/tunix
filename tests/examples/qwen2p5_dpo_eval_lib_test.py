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
from types import SimpleNamespace
import importlib.util
import sys

from absl.testing import absltest


def _load_eval_module():
  module_path = (
      Path(__file__).resolve().parents[2] / "examples/dpo/qwen2p5_dpo_eval_lib.py"
  )
  spec = importlib.util.spec_from_file_location(
      "qwen2p5_dpo_eval_lib",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class Qwen2p5DPOEvalLibTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.eval_lib = _load_eval_module()

  def test_resolve_metrics_prefix_prefers_direct_trainer_attr(self):
    trainer = SimpleNamespace(metrics_prefix="direct", config=None)
    self.assertEqual(self.eval_lib._resolve_metrics_prefix(trainer), "direct")

  def test_resolve_metrics_prefix_falls_back_to_config(self):
    trainer = SimpleNamespace(config=SimpleNamespace(metrics_prefix="from_config"))
    self.assertEqual(
        self.eval_lib._resolve_metrics_prefix(trainer), "from_config"
    )

  def test_resolve_metrics_prefix_falls_back_to_training_config(self):
    trainer = SimpleNamespace(
        training_config=SimpleNamespace(metrics_prefix="legacy")
    )
    self.assertEqual(self.eval_lib._resolve_metrics_prefix(trainer), "legacy")


if __name__ == "__main__":
  absltest.main()
