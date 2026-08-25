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
import tempfile
import importlib.util

from absl.testing import absltest


def _load_helper_module():
  module_path = (
      Path(__file__).resolve().parents[2]
      / "examples/dpo/qwen2p5_clean_main_table_lib.py"
  )
  spec = importlib.util.spec_from_file_location(
      "qwen2p5_clean_main_table_lib",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5CleanMainTableLibTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.helper = _load_helper_module()

  def _make_run(self, runs_root: Path, run_name: str) -> None:
    run_dir = runs_root / run_name
    (run_dir / "exported_model").mkdir(parents=True)
    (run_dir / "tensorboard").mkdir()

  def test_discover_clean_main_table_runs_uses_legacy_seed0_fallback(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      repo_root = Path(tmp_dir)
      runs_root = repo_root / "runs"
      runs_root.mkdir()
      self._make_run(
          runs_root,
          (
              "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
              "random_pair_filtering_clean_lora_full_20260417_013847"
          ),
      )
      self._make_run(
          runs_root,
          (
              "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
              "random_pair_filtering_filt10_clean_lora_full_seed1_20260422_010203"
          ),
      )

      rows = self.helper.discover_clean_main_table_runs(
          repo_root=repo_root,
          run_ts="20260422_010203",
          legacy_run_ts="20260417_013847",
          methods=("random_pair_filtering_filt10",),
          seeds=(0, 1),
      )

    self.assertLen(rows, 2)
    self.assertEqual(rows[0]["run_key"], "random_pair_filtering_filt10_seed0")
    self.assertEqual(rows[0]["source"], "legacy_seed0")
    self.assertEqual(rows[0]["source_variant"], "random_pair_filtering")
    self.assertEqual(rows[1]["run_key"], "random_pair_filtering_filt10_seed1")
    self.assertEqual(rows[1]["source"], "seeded")
    self.assertEqual(rows[1]["source_variant"], "random_pair_filtering_filt10")

  def test_discover_clean_main_table_runs_requires_seeded_run_for_filt5(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      repo_root = Path(tmp_dir)
      runs_root = repo_root / "runs"
      runs_root.mkdir()
      self._make_run(
          runs_root,
          (
              "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
              "random_pair_filtering_filt5_clean_lora_full_seed2_20260422_010203"
          ),
      )

      rows = self.helper.discover_clean_main_table_runs(
          repo_root=repo_root,
          run_ts="20260422_010203",
          legacy_run_ts="20260417_013847",
          methods=("random_pair_filtering_filt5",),
          seeds=(2,),
      )

    self.assertLen(rows, 1)
    self.assertEqual(rows[0]["run_key"], "random_pair_filtering_filt5_seed2")
    self.assertEqual(rows[0]["source"], "seeded")

  def test_discover_clean_main_table_runs_can_disable_legacy_fallback(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      repo_root = Path(tmp_dir)
      runs_root = repo_root / "runs"
      runs_root.mkdir()
      self._make_run(
          runs_root,
          (
              "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
              "vanilla_dpo_clean_lora_full_20260417_013847"
          ),
      )

      with self.assertRaises(SystemExit):
        self.helper.discover_clean_main_table_runs(
            repo_root=repo_root,
            run_ts="20260424_010203",
            legacy_run_ts="20260417_013847",
            allow_legacy_fallback=False,
            methods=("vanilla_dpo",),
            seeds=(0,),
        )

  def test_discover_clean_main_table_runs_respects_profile(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      repo_root = Path(tmp_dir)
      runs_root = repo_root / "runs"
      runs_root.mkdir()
      self._make_run(
          runs_root,
          (
              "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
              "vanilla_dpo_clean_lora_smoke_seed0_20260424_010203"
          ),
      )

      rows = self.helper.discover_clean_main_table_runs(
          repo_root=repo_root,
          run_ts="20260424_010203",
          legacy_run_ts=None,
          allow_legacy_fallback=False,
          methods=("vanilla_dpo",),
          seeds=(0,),
          profile="smoke",
      )

    self.assertLen(rows, 1)
    self.assertEqual(
        rows[0]["run_name"],
        "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
        "vanilla_dpo_clean_lora_smoke_seed0_20260424_010203",
    )

  def test_expected_run_specs_builds_seeded_keys(self):
    specs = self.helper.expected_run_specs(
        methods=("vanilla_dpo", "self_dtv"),
        seeds=(0, 2),
    )

    self.assertEqual(
        specs,
        [
            {
                "logical_variant": "vanilla_dpo",
                "seed": 0,
                "run_key": "vanilla_dpo_seed0",
            },
            {
                "logical_variant": "vanilla_dpo",
                "seed": 2,
                "run_key": "vanilla_dpo_seed2",
            },
            {
                "logical_variant": "self_inf",
                "seed": 0,
                "run_key": "self_inf_seed0",
            },
            {
                "logical_variant": "self_inf",
                "seed": 2,
                "run_key": "self_inf_seed2",
            },
        ],
    )


if __name__ == "__main__":
  absltest.main()
