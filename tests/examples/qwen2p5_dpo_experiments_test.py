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
from omegaconf import OmegaConf


def _load_experiment_module():
  module_path = (
      Path(__file__).resolve().parents[2]
      / "examples/dpo/qwen2p5_dpo_experiments.py"
  )
  spec = importlib.util.spec_from_file_location(
      "qwen2p5_dpo_experiments",
      module_path,
  )
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class Qwen2p5DpoExperimentsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.exp = _load_experiment_module()

  def test_build_dataset_module_spec_uses_static_corruption_for_train(self):
    module_spec = self.exp.build_dataset_module_spec(
        subset="train",
        corruption_config="tail50_flip20",
        profile="full",
    )

    self.assertIn("flip_scope='tail_fraction'", module_spec)
    self.assertIn("flip_ratio=0.2", module_spec)
    self.assertIn("flip_tail_fraction=0.5", module_spec)
    self.assertIn("subset='train'", module_spec)

  def test_build_dataset_module_spec_injects_train_shuffle_seed_only_for_train(self):
    train_module_spec = self.exp.build_dataset_module_spec(
        subset="train",
        corruption_config="clean",
        profile="full",
        train_shuffle_seed=2,
    )
    eval_module_spec = self.exp.build_dataset_module_spec(
        subset="eval",
        corruption_config="clean",
        profile="full",
        train_shuffle_seed=2,
    )

    self.assertIn("shuffle_seed=2", train_module_spec)
    self.assertNotIn("shuffle_seed=", eval_module_spec)

  def test_build_dataset_module_spec_keeps_eval_clean(self):
    module_spec = self.exp.build_dataset_module_spec(
        subset="eval",
        corruption_config="global_flip40",
        profile="smoke",
    )

    self.assertIn("subset='eval'", module_spec)
    self.assertIn("flip_scope='none'", module_spec)
    self.assertIn("flip_ratio=0.0", module_spec)
    self.assertIn("limit=64", module_spec)

  def test_prepare_launch_config_disables_late_flip_and_encodes_corruption(self):
    config_path = str(
        Path(__file__).resolve().parents[2]
        / "examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = f"{tmp_dir}/config.yaml"
      self.exp.prepare_launch_config(
          config_path=config_path,
          output_path=output_path,
          run_root="/tmp/dpo_reward_based_filtering_global_flip20",
          run_name="test-run",
          profile="smoke",
          variant="reward_based_filtering",
          corruption_config="global_flip20",
          sft_model_path="/tmp/sft_model",
          curation_threshold="",
          curation_keep_ratio="0.85",
          self_influence_dot_threshold="",
          dpo_seed=2,
          curation_seed=19,
          train_shuffle_seed=5,
      )
      cfg = OmegaConf.load(output_path)

    self.assertEqual(cfg.actor_model_config.model_path, "/tmp/sft_model")
    self.assertEqual(cfg.reference_model_config.model_path, "/tmp/sft_model")
    self.assertEqual(cfg.dpo_config.late_flip_ratio, 0.0)
    self.assertIsNone(cfg.dpo_config.late_flip_start_step)
    self.assertTrue(cfg.dpo_config.use_dynamic_batch_curation)
    self.assertEqual(cfg.dpo_config.curation_variant, "reward_based_filtering")
    self.assertEqual(cfg.dpo_config.curation_keep_ratio, 0.85)
    self.assertEqual(cfg.actor_model_config.rng_seed, 2)
    self.assertEqual(cfg.reference_model_config.rng_seed, 2)
    self.assertEqual(cfg.dpo_config.curation_seed, 19)
    self.assertIn("shuffle_seed=5", cfg.train_data_module)
    self.assertIn("flip_scope='global'", cfg.train_data_module)
    self.assertIn("flip_ratio=0.2", cfg.train_data_module)
    self.assertIn("limit=512", cfg.train_data_module)
    self.assertIn("flip_scope='none'", cfg.eval_data_module)
    self.assertEqual(
        cfg.training_config.checkpoint_root_directory,
        "/tmp/dpo_reward_based_filtering_global_flip20/checkpoints",
    )
    self.assertEqual(cfg.training_config.metrics_logging_options.run_name, "test-run")

  def test_run_metadata_includes_corruption_config(self):
    run_root = self.exp.build_run_root(
        repo_root="/repo",
        variant="self_inf_batch",
        corruption_config="tail50_flip40",
        ft_mode="lora",
        profile="full",
        run_ts="20260413_120000",
    )
    run_name = self.exp.build_run_name(
        variant="self_inf_batch",
        corruption_config="tail50_flip40",
        ft_mode="lora",
        profile="full",
        run_ts="20260413_120000",
    )

    self.assertIn("self_inf_tail50_flip40_lora_full_20260413_120000", run_root)
    self.assertIn(
        "self_inf-tail50_flip40-lora-full-20260413_120000",
        run_name,
    )

  def test_seeded_run_metadata_encodes_seed(self):
    run_root = self.exp.build_run_root(
        repo_root="/repo",
        variant="reward_based_filtering_5pct",
        corruption_config="clean",
        ft_mode="lora",
        profile="full",
        run_ts="20260422_010203",
        seed=2,
    )
    run_name = self.exp.build_run_name(
        variant="reward_based_filtering_5pct",
        corruption_config="clean",
        ft_mode="lora",
        profile="full",
        run_ts="20260422_010203",
        seed=2,
    )
    parsed = self.exp.parse_run_dir_name(Path(run_root).name)

    self.assertIn(
        "reward_based_filtering_filt5_clean_lora_full_seed2_20260422_010203",
        run_root,
    )
    self.assertIn(
        "reward_based_filtering_filt5-clean-lora-full-seed2-20260422_010203",
        run_name,
    )
    self.assertIsNotNone(parsed)
    self.assertEqual(parsed["variant"], "reward_based_filtering_filt5")
    self.assertEqual(parsed["seed"], 2)
    self.assertEqual(
        parsed["display_name"], "Reward-based Filtering (5%)"
    )

  def test_parse_run_dir_name_accepts_named_run_ts(self):
    parsed = self.exp.parse_run_dir_name(
        "dpo_qwen2p5_1p5b_ultrafeedback_from_sft_"
        "vanilla_dpo_clean_lora_smoke_seed0_20260424_true_std_smoke"
    )

    self.assertIsNotNone(parsed)
    self.assertEqual(parsed["variant"], "vanilla_dpo")
    self.assertEqual(parsed["profile"], "smoke")
    self.assertEqual(parsed["seed"], 0)
    self.assertEqual(parsed["run_ts"], "20260424_true_std_smoke")

  def test_prepare_launch_config_uses_variant_specific_keep_ratio_defaults(self):
    config_path = str(
        Path(__file__).resolve().parents[2]
        / "examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = f"{tmp_dir}/config.yaml"
      self.exp.prepare_launch_config(
          config_path=config_path,
          output_path=output_path,
          run_root="/tmp/dpo_random5_clean",
          run_name="random5-run",
          profile="full",
          variant="random_pair_filtering_5pct",
          corruption_config="clean",
          sft_model_path="/tmp/sft_model",
          curation_threshold="",
          curation_keep_ratio="",
          self_influence_dot_threshold="",
          dpo_seed=1,
          curation_seed=1,
          train_shuffle_seed=None,
      )
      cfg = OmegaConf.load(output_path)

    self.assertTrue(cfg.dpo_config.use_dynamic_batch_curation)
    self.assertEqual(cfg.dpo_config.curation_variant, "random_pair_filtering")
    self.assertEqual(cfg.dpo_config.curation_keep_ratio, 0.95)

  def test_prepare_launch_config_supports_legacy_variant_aliases(self):
    config_path = str(
        Path(__file__).resolve().parents[2]
        / "examples/dpo/qwen2p5_1p5b_ultrafeedback_from_sft_lora.yaml"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = f"{tmp_dir}/config.yaml"
      self.exp.prepare_launch_config(
          config_path=config_path,
          output_path=output_path,
          run_root="/tmp/dpo_alias_check",
          run_name="alias-run",
          profile="full",
          variant="baseline",
          corruption_config="clean",
          sft_model_path="/tmp/sft_model",
          curation_threshold="",
          curation_keep_ratio="",
          self_influence_dot_threshold="",
          dpo_seed=None,
          curation_seed=None,
          train_shuffle_seed=None,
      )
      cfg = OmegaConf.load(output_path)

    self.assertFalse(cfg.dpo_config.use_dynamic_batch_curation)


if __name__ == "__main__":
  absltest.main()
