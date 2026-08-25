# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import common
from tunix.sft.dpo import dpo_trainer as dpo_lib
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)
# jax.config.update("jax_debug_nans", True) # useful for debugging NaN


class MySource(grain.RandomAccessDataSource):

  def __init__(self, data):
    self._data = data

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def _dummy_dataset(
    source: MySource,
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    chosen_ids: np.ndarray,
    chosen_mask: np.ndarray,
    rejected_ids: np.ndarray,
    rejected_mask: np.ndarray,
):
  return grain.MapDataset.source(source).map(
      lambda x: dpo_lib.TrainingInput(
          prompt_ids=prompt_ids,
          prompt_mask=prompt_mask,
          chosen_ids=chosen_ids,
          chosen_mask=chosen_mask,
          rejected_ids=rejected_ids,
          rejected_mask=rejected_mask,
      )
  )


def _dummy_string_dataset(
    source: MySource,
    prompts: np.ndarray,
    chosen_responses: np.ndarray,
    rejected_responses: np.ndarray,
    return_dict=False,
):
  ds = grain.MapDataset.source(source)
  if return_dict:
    return ds.map(
        lambda x: {
            "prompts": prompts,
            "chosen_responses": chosen_responses,
            "rejected_responses": rejected_responses,
        }
    )
  else:
    return ds.map(
        lambda x: dpo_lib.DataInput(
            prompts=prompts,
            chosen_responses=chosen_responses,
            rejected_responses=rejected_responses,
        )
    )


class DPOTrainerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="with_ref_model",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
          use_ref_model=True,
      ),
      dict(
          testcase_name="without_ref_model",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
          use_ref_model=False,
      ),
  )
  def test_dpo_trainer(
      self,
      prompt_ids,
      prompt_mask,
      chosen_ids,
      chosen_mask,
      rejected_ids,
      rejected_mask,
      use_ref_model,
  ):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = None
    if use_ref_model:
      ref_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    dpo_config = dpo_lib.DPOTrainingConfig(
        eval_every_n_steps=5,
        max_steps=10,
    )
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_config,
    )
    train_ds = _dummy_dataset(
        MySource(np.arange(10)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
    )
    eval_ds = _dummy_dataset(
        MySource(np.arange(2)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
    )
    dpo_trainer.train(train_ds, eval_ds=eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    for metric_name in [
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margin",
        "rewards/accuracy",
        "log_probs/chosen",
        "log_probs/rejected",
    ]:
      self.assertLen(
          dpo_trainer.metrics_logger.get_metric_history(
              "", metric_name, "train"
          ),
          dpo_trainer._train_steps,
      )
      self.assertLen(
          dpo_trainer.metrics_logger.get_metric_history(
              "", metric_name, "eval"
          ),
          3,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="dataclass_inputs",
          train_ds=_dummy_string_dataset(
              MySource(np.arange(10)),
              prompts=["Tunix", "Parallax"],
              chosen_responses=["PT", "distributed training"],
              rejected_responses=["optimizer library", "quantization"],
          ),
      ),
      dict(
          testcase_name="dict_inputs",
          train_ds=_dummy_string_dataset(
              MySource(np.arange(10)),
              prompts=["Tunix", "Parallax"],
              chosen_responses=["PT", "distributed training"],
              rejected_responses=["optimizer library", "quantization"],
              return_dict=True,
          ),
      ),
  )
  def test_dpo_trainer_with_string_inputs(self, train_ds):
    tokenizer = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    original_ref_variables = jax.tree.map(
        jnp.copy, nnx.state(ref_model, nnx.Param)
    )
    dpo_config = dpo_lib.DPOTrainingConfig(
        eval_every_n_steps=10,
        max_steps=10,
        max_prompt_length=3,
        max_response_length=3,
    )
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_config,
        tokenizer=tokenizer,
    )
    dpo_trainer.train(train_ds, None)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)
    if ref_model is not None:
      ref_variables = nnx.state(ref_model, nnx.Param)
      jax.tree.map_with_path(
          tc.assert_equal, original_ref_variables, ref_variables
      )

    for metric_name in [
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margin",
        "rewards/accuracy",
    ]:
      self.assertLen(
          dpo_trainer.metrics_logger.get_metric_history(
              "", metric_name, "train"
          ),
          dpo_trainer._train_steps,
      )

  def test_dpo_loss_fn(self):
    np.random.seed(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    per_token_logps = np.random.normal(0, 5, size=(8, 4))
    ref_per_token_logps = np.random.normal(0, 5, size=(8, 4)).sum(axis=-1)
    train_example = dpo_lib.TrainExample(
        input_ids=jnp.arange(0, 32).reshape(8, 4),
        positions=jnp.ones((8, 4)),
        attention_mask=jnp.ones((8, 4, 4)),
        ref_chosen_logps=ref_per_token_logps[:4],
        ref_rejected_logps=ref_per_token_logps[4:],
        logits_to_keep=4,
        completion_mask=jnp.ones((8, 4)),
    )

    with mock.patch.object(
        common, "get_per_token_logps", return_value=jnp.array(per_token_logps)
    ):
      loss, _ = dpo_lib.dpo_loss_fn(
          model, train_example, beta=0.1, label_smoothing=0
      )
      np.testing.assert_allclose(loss, 0.753059, atol=1e-5)

      loss, _ = dpo_lib.dpo_loss_fn(
          model, train_example, beta=0.1, label_smoothing=0.3
      )
      np.testing.assert_allclose(loss, 0.925447, atol=1e-5)

  def test_dpo_prepare_inputs_for_strings(self):
    tokenizer = tc.MockVocab()

    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DPOTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
            max_prompt_length=3,
            max_response_length=3,
        ),
        tokenizer=tokenizer,
    )

    # These are random strings, they hold no meaning.
    training_input = dpo_lib.DataInput(
        prompts=["Tunix", "Parallax"],
        chosen_responses=["PT", "distributed training"],
        rejected_responses=["optimizer library", "quantization"],
    )
    out = dpo_trainer._prepare_inputs(training_input)

    expected_input_ids = np.array([
        [0, 1, 14, 1, 16, 0],
        [0, 1, 15, 1, 18, 19],
        [0, 1, 14, 1, 20, 17],
        [0, 1, 15, 1, 21, 0],
    ])
    np.testing.assert_array_equal(out.input_ids, expected_input_ids)
    self.assertEqual(np.sum(out.attention_mask[0]), 14)
    self.assertEqual(np.sum(out.attention_mask[1]), 15)
    self.assertEqual(np.sum(out.attention_mask[2]), 15)
    self.assertEqual(np.sum(out.attention_mask[3]), 14)
    np.testing.assert_allclose(
        out.ref_chosen_logps,
        np.array([-11.21106, -5.985622]),
        atol=1e-1,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        out.ref_rejected_logps,
        np.array([-13.020714, -5.95595]),
        atol=1e-1,
        rtol=1e-2,
    )
    expected_completion_mask = np.array(
        [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]]
    )
    np.testing.assert_array_equal(out.completion_mask, expected_completion_mask)
    self.assertEqual(out.logits_to_keep, 3)

  def test_dpo_prepare_inputs(self):
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    ref_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DPOTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
        ),
    )

    training_input = dpo_lib.TrainingInput(
        prompt_ids=np.array([[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]]),
        prompt_mask=np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        chosen_ids=np.array([[10, 11, 12, 0], [13, 14, 15, 16]]),
        chosen_mask=np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
        rejected_ids=np.array([[20, 21, 22, 0], [23, 0, 0, 0]]),
        rejected_mask=np.array([[1, 1, 1, 0], [1, 0, 0, 0]]),
    )
    out = dpo_trainer._prepare_inputs(training_input)
    expected_input_ids = np.array([
        [1, 2, 3, 4, 5, 10, 11, 12, 0],
        [0, 0, 1, 2, 3, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 20, 21, 22, 0],
        [0, 0, 1, 2, 3, 23, 0, 0, 0],
    ])
    np.testing.assert_array_equal(out.input_ids, expected_input_ids)
    self.assertEqual(np.sum(out.attention_mask[0]), 44)
    self.assertEqual(np.sum(out.attention_mask[1]), 28)
    self.assertEqual(np.sum(out.attention_mask[2]), 44)
    self.assertEqual(np.sum(out.attention_mask[3]), 22)
    np.testing.assert_allclose(
        out.ref_chosen_logps, np.array([-20.536058, -20.905323]), rtol=1e-2
    )
    np.testing.assert_allclose(
        out.ref_rejected_logps, np.array([-18.149311, -8.219014]), rtol=1e-2
    )
    expected_completion_mask = np.array(
        [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
    )
    np.testing.assert_array_equal(out.completion_mask, expected_completion_mask)
    self.assertEqual(out.logits_to_keep, 4)

  def test_aggregate_curated_step_filters_outlier(self):
    grads = {"w": jnp.array([[1.0], [1.0], [10.0], [1.0]])}
    losses = jnp.array([1.0, 1.0, 10.0, 1.0])
    aux = {"metric": jnp.array([1.0, 1.0, 10.0, 1.0])}
    grad_norms = jnp.array([1.0, 1.0, 10.0, 1.0])

    final_grads, final_loss, final_aux = dpo_lib.aggregate_curated_step(
        per_sample_grads=grads,
        per_sample_losses=losses,
        per_sample_aux=aux,
        per_sample_grad_norms=grad_norms,
        curation_threshold=1.0,
    )

    np.testing.assert_allclose(final_grads["w"], np.array([1.0]), atol=1e-6)
    np.testing.assert_allclose(final_loss, 1.0, atol=1e-6)
    np.testing.assert_allclose(final_aux["metric"], 1.0, atol=1e-6)
    np.testing.assert_allclose(final_aux["dbc/num_samples_kept"], 3.0, atol=1e-6)
    np.testing.assert_allclose(
        final_aux["dbc/num_samples_filtered"], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(final_aux["dbc/keep_ratio"], 0.75, atol=1e-6)

  def test_aggregate_self_influence_curated_step_filters_anti_aligned_sample(
      self,
  ):
    grads = {"w": jnp.array([[1.0], [1.0], [1.0], [-0.1]])}
    losses = jnp.array([1.0, 1.0, 1.0, 10.0])
    aux = {"metric": jnp.array([1.0, 1.0, 1.0, 10.0])}
    grad_norms = jnp.array([1.0, 1.0, 1.0, 0.1])

    final_grads, final_loss, final_aux = (
        dpo_lib.aggregate_self_influence_curated_step(
            per_sample_grads=grads,
            per_sample_losses=losses,
            per_sample_aux=aux,
            per_sample_grad_norms=grad_norms,
            dot_threshold=0.0,
        )
    )

    np.testing.assert_allclose(final_grads["w"], np.array([1.0]), atol=1e-6)
    np.testing.assert_allclose(final_loss, 1.0, atol=1e-6)
    np.testing.assert_allclose(final_aux["metric"], 1.0, atol=1e-6)
    np.testing.assert_allclose(final_aux["dbc/num_samples_kept"], 3.0, atol=1e-6)
    np.testing.assert_allclose(
        final_aux["dbc/num_samples_filtered"], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(final_aux["dbc/keep_ratio"], 0.75, atol=1e-6)
    np.testing.assert_allclose(
        final_aux["dbc/self_inf_dot_threshold"], 0.0, atol=1e-6
    )

  def test_aggregate_random_curated_step_keeps_requested_count_deterministically(
      self,
  ):
    grads = {"w": jnp.array([[1.0], [2.0], [3.0], [4.0]])}
    losses = jnp.array([1.0, 2.0, 3.0, 4.0])
    aux = {"metric": jnp.array([1.0, 2.0, 3.0, 4.0])}
    grad_norms = jnp.array([1.0, 2.0, 3.0, 4.0])

    final_grads, final_loss, final_aux = dpo_lib.aggregate_random_curated_step(
        per_sample_grads=grads,
        per_sample_losses=losses,
        per_sample_aux=aux,
        per_sample_grad_norms=grad_norms,
        curation_keep_ratio=0.5,
        curation_seed=7,
        window_seed=17,
    )
    repeated_grads, repeated_loss, repeated_aux = (
        dpo_lib.aggregate_random_curated_step(
            per_sample_grads=grads,
            per_sample_losses=losses,
            per_sample_aux=aux,
            per_sample_grad_norms=grad_norms,
            curation_keep_ratio=0.5,
            curation_seed=7,
            window_seed=17,
        )
    )
    changed_seed_grads, changed_seed_loss, changed_seed_aux = (
        dpo_lib.aggregate_random_curated_step(
            per_sample_grads=grads,
            per_sample_losses=losses,
            per_sample_aux=aux,
            per_sample_grad_norms=grad_norms,
            curation_keep_ratio=0.5,
            curation_seed=8,
            window_seed=17,
        )
    )

    scores = np.asarray(
        jax.random.uniform(
            jax.random.fold_in(jax.random.PRNGKey(7), 17), shape=(4,)
        )
    )
    kept_indices = np.argsort(scores)[-2:]
    expected_mean = np.mean(np.array([1.0, 2.0, 3.0, 4.0])[kept_indices])

    np.testing.assert_allclose(final_grads["w"], np.array([expected_mean]), atol=1e-6)
    np.testing.assert_allclose(final_loss, expected_mean, atol=1e-6)
    np.testing.assert_allclose(final_aux["metric"], expected_mean, atol=1e-6)
    np.testing.assert_allclose(final_aux["dbc/num_samples_kept"], 2.0, atol=1e-6)
    np.testing.assert_allclose(final_aux["dbc/keep_ratio"], 0.5, atol=1e-6)
    np.testing.assert_allclose(
        final_aux["dbc/random_keep_ratio_target"], 0.5, atol=1e-6
    )
    np.testing.assert_allclose(final_grads["w"], repeated_grads["w"], atol=1e-6)
    np.testing.assert_allclose(final_loss, repeated_loss, atol=1e-6)
    np.testing.assert_allclose(final_aux["metric"], repeated_aux["metric"], atol=1e-6)
    self.assertNotAlmostEqual(
        float(changed_seed_grads["w"][0]),
        float(final_grads["w"][0]),
    )
    self.assertNotAlmostEqual(float(changed_seed_loss), float(final_loss))
    self.assertNotAlmostEqual(
        float(changed_seed_aux["metric"]),
        float(final_aux["metric"]),
    )

  def test_aggregate_reward_margin_curated_step_keeps_top_margin_samples(self):
    grads = {"w": jnp.array([[1.0], [4.0], [-2.0], [9.0]])}
    losses = jnp.array([1.0, 4.0, 2.0, 9.0])
    aux = {
        "metric": jnp.array([1.0, 4.0, 2.0, 9.0]),
        "rewards/margin": jnp.array([0.1, 0.4, -0.2, 0.9]),
    }
    grad_norms = jnp.array([1.0, 4.0, 2.0, 9.0])

    final_grads, final_loss, final_aux = (
        dpo_lib.aggregate_reward_margin_curated_step(
            per_sample_grads=grads,
            per_sample_losses=losses,
            per_sample_aux=aux,
            per_sample_grad_norms=grad_norms,
            curation_keep_ratio=0.5,
        )
    )

    np.testing.assert_allclose(final_grads["w"], np.array([6.5]), atol=1e-6)
    np.testing.assert_allclose(final_loss, 6.5, atol=1e-6)
    np.testing.assert_allclose(final_aux["metric"], 6.5, atol=1e-6)
    np.testing.assert_allclose(final_aux["dbc/num_samples_kept"], 2.0, atol=1e-6)
    np.testing.assert_allclose(final_aux["dbc/keep_ratio"], 0.5, atol=1e-6)
    np.testing.assert_allclose(
        final_aux["dbc/reward_margin_mean"], 0.3, atol=1e-6
    )
    np.testing.assert_allclose(
        final_aux["dbc/reward_margin_cutoff"], 0.4, atol=1e-6
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="outlier_l2",
          curation_variant="outlier_l2",
          curation_threshold=1e6,
          curation_keep_ratio=1.0,
          self_influence_dot_threshold=0.0,
          expected_metrics=(
              "dbc/num_samples_total",
              "dbc/num_samples_kept",
              "dbc/keep_ratio",
              "dbc/grad_norm_cutoff",
          ),
      ),
      dict(
          testcase_name="self_inf_batch",
          curation_variant="self_inf_batch",
          curation_threshold=3.0,
          curation_keep_ratio=1.0,
          self_influence_dot_threshold=-1e6,
          expected_metrics=(
              "dbc/num_samples_total",
              "dbc/num_samples_kept",
              "dbc/keep_ratio",
              "dbc/self_inf_dot_mean",
          ),
      ),
      dict(
          testcase_name="random_pair_filtering",
          curation_variant="random_pair_filtering",
          curation_threshold=3.0,
          curation_keep_ratio=1.0,
          self_influence_dot_threshold=0.0,
          expected_metrics=(
              "dbc/num_samples_total",
              "dbc/num_samples_kept",
              "dbc/keep_ratio",
              "dbc/random_keep_ratio_target",
          ),
      ),
      dict(
          testcase_name="reward_based_filtering",
          curation_variant="reward_based_filtering",
          curation_threshold=3.0,
          curation_keep_ratio=1.0,
          self_influence_dot_threshold=0.0,
          expected_metrics=(
              "dbc/num_samples_total",
              "dbc/num_samples_kept",
              "dbc/keep_ratio",
              "dbc/reward_margin_mean",
          ),
      ),
  )
  def test_curated_trainer_matches_grad_accumulation_when_no_samples_filtered(
      self,
      curation_variant,
      curation_threshold,
      curation_keep_ratio,
      self_influence_dot_threshold,
      expected_metrics,
  ):
    prompt_ids = np.arange(0, 10).reshape(2, 5)
    prompt_mask = np.ones((2, 5))
    chosen_ids = np.arange(10, 20).reshape(2, 5)
    chosen_mask = np.ones((2, 5))
    rejected_ids = np.arange(20, 30).reshape(2, 5)
    rejected_mask = np.ones((2, 5))
    train_ds = _dummy_dataset(
        MySource(np.arange(4)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
    )
    baseline_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    curated_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    baseline_ref = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    curated_ref = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))

    baseline_trainer = dpo_lib.DPOTrainer(
        model=baseline_model,
        ref_model=baseline_ref,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DPOTrainingConfig(
            eval_every_n_steps=10,
            max_steps=2,
            gradient_accumulation_steps=2,
        ),
    )
    curated_trainer = dpo_lib.CuratedDPOTrainer(
        model=curated_model,
        ref_model=curated_ref,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DPOTrainingConfig(
            eval_every_n_steps=10,
            max_steps=2,
            gradient_accumulation_steps=2,
            use_dynamic_batch_curation=True,
            curation_variant=curation_variant,
            curation_threshold=curation_threshold,
            curation_keep_ratio=curation_keep_ratio,
            self_influence_dot_threshold=self_influence_dot_threshold,
        ),
    )

    baseline_trainer.train(train_ds, None)
    curated_trainer.train(train_ds, None)

    jax.tree.map_with_path(
        functools.partial(tc.assert_close, atol=1e-6, rtol=1e-6),
        nnx.state(baseline_model, nnx.Param),
        nnx.state(curated_model, nnx.Param),
    )
    self.assertEqual(baseline_trainer.train_steps, curated_trainer.train_steps)
    self.assertEqual(baseline_trainer.iter_steps, curated_trainer.iter_steps)
    for metric_name in expected_metrics:
      self.assertLen(
          curated_trainer.metrics_logger.get_metric_history(
              "", metric_name, "train"
          ),
          curated_trainer.train_steps,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="self_inf_alias",
          variant="self-inf-batch",
          expected="self_inf_batch",
      ),
      dict(
          testcase_name="random_pair_filtering_alias",
          variant="random-pair-filtering",
          expected="random_pair_filtering",
      ),
      dict(
          testcase_name="reward_based_filtering_alias",
          variant="reward-based-filtering",
          expected="reward_based_filtering",
      ),
  )
  def test_dpo_training_config_normalizes_curation_variant_alias(
      self, variant, expected
  ):
    config = dpo_lib.DPOTrainingConfig(
        eval_every_n_steps=10,
        max_steps=20,
        curation_variant=variant,
    )

    self.assertEqual(config.curation_variant, expected)


if __name__ == "__main__":
  absltest.main()
