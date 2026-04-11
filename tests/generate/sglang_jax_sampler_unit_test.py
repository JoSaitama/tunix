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

import asyncio
import threading
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.generate import sglang_jax_sampler


class _DummyTokenizer:

  def encode(self, text, **kwargs):
    del kwargs
    return [len(text), len(text) + 1]

  def decode(self, ids, **kwargs):
    del kwargs
    return ",".join(str(i) for i in ids)

  def bos_id(self):
    return 101

  def eos_id(self):
    return 102

  def pad_id(self):
    return 0


class _DummySamplingParams:

  def __init__(self):
    self.max_new_tokens = None
    self.n = None
    self.temperature = None
    self.stop_token_ids = None
    self.skip_special_tokens = None
    self.top_p = None
    self.top_k = None

  def convert_to_dict(self):
    return {
        "max_new_tokens": self.max_new_tokens,
        "n": self.n,
        "temperature": self.temperature,
        "stop_token_ids": self.stop_token_ids,
        "skip_special_tokens": self.skip_special_tokens,
        "top_p": self.top_p,
        "top_k": self.top_k,
    }


class SglangJaxSamplerUnitTest(absltest.TestCase):

  def _build_sampler(self, outputs):
    sampler = object.__new__(sglang_jax_sampler.SglangJaxSampler)
    sampler.tokenizer = _DummyTokenizer()
    sampler.args = {"context_length": 32}
    sampler._engine_lock = threading.RLock()
    sampler.engine = mock.Mock()
    sampler.engine.get_default_sampling_params.return_value = (
        _DummySamplingParams()
    )
    sampler.engine.generate.return_value = outputs
    return sampler

  def test_multisampling_flattens_outputs_prompt_major(self):
    sampler = self._build_sampler(
        outputs=[
            {
                "output_ids": [[11, 12], [13]],
                "text": ["a-0", "a-1"],
            },
            {
                "output_ids": [[21], [22, 23]],
                "text": ["b-0", "b-1"],
            },
        ]
    )

    output = sampler(
        input_strings=["aa", "bbbb"],
        max_generation_steps=4,
        max_prompt_length=4,
        multi_sampling=2,
        pad_output=True,
    )

    self.assertEqual(output.text, ["a-0", "a-1", "b-0", "b-1"])
    self.assertEqual(output.tokens.shape, (4, 4))
    self.assertEqual(output.padded_prompt_tokens.shape, (4, 4))
    np.testing.assert_array_equal(
        output.padded_prompt_tokens[0], output.padded_prompt_tokens[1]
    )
    np.testing.assert_array_equal(
        output.padded_prompt_tokens[2], output.padded_prompt_tokens[3]
    )

  def test_close_shuts_down_engine(self):
    sampler = self._build_sampler(outputs=[])

    sampler.close()

    sampler.engine.shutdown.assert_called_once_with()

  def test_flush_cache_delegates_to_engine(self):
    sampler = self._build_sampler(outputs=[])

    sampler.flush_cache()

    sampler.engine.flush_cache.assert_called_once_with()

  def test_release_memory_occupation_offloads_weights_to_host(self):
    sampler = self._build_sampler(outputs=[])
    model_runner = mock.Mock()
    model_runner.model = object()

    with (
        mock.patch.object(
            sglang_jax_sampler.SglangJaxSampler,
            '_model_runner',
            new_callable=mock.PropertyMock,
            return_value=model_runner,
        ),
        mock.patch.object(
            sglang_jax_sampler.nnx,
            'split',
            return_value=('model_def', 'device_state'),
        ),
        mock.patch.object(
            sglang_jax_sampler.jax,
            'device_get',
            return_value='host_state',
        ),
        mock.patch.object(
            sglang_jax_sampler.nnx,
            'merge',
            return_value='host_model',
        ) as merge_mock,
        mock.patch.object(
            sglang_jax_sampler.jax.tree_util,
            'tree_flatten',
            return_value=(['host_leaf'], None),
        ),
    ):
      sampler.release_memory_occupation()

    merge_mock.assert_called_once_with('model_def', 'host_state')
    self.assertEqual(model_runner.model, 'host_model')
    self.assertEqual(model_runner.model_state_leaves, ['host_leaf'])
    self.assertTrue(sampler._weights_offloaded_to_host)

  def test_resume_memory_occupation_reshards_host_weights(self):
    sampler = self._build_sampler(outputs=[])
    sampler._weights_offloaded_to_host = True

    with (
        mock.patch.object(
            sglang_jax_sampler.SglangJaxSampler,
            'transformer_state',
            new_callable=mock.PropertyMock,
            return_value='host_state',
        ),
        mock.patch.object(
            sampler,
            'update_params',
        ) as update_params_mock,
    ):
      sampler.resume_memory_occupation()

    update_params_mock.assert_called_once_with(
        updated_weights='host_state', filter_types=None
    )

  def test_update_params_passes_hook_functions_to_transfer(self):
    sampler = self._build_sampler(outputs=[])
    sampler.mappings = {'layers.*.attn.k_bias': ('dst', None)}
    sampler.to_hf_transpose_keys = {'lm_head.w': (1, 0)}
    sampler.to_hf_hook_fns = {'layers.*.attn.k_bias': mock.Mock()}
    model_runner = mock.Mock()

    with (
        mock.patch.object(
            sglang_jax_sampler.SglangJaxSampler,
            'transformer_state',
            new_callable=mock.PropertyMock,
            return_value='dst_state',
        ),
        mock.patch.object(
            sglang_jax_sampler.SglangJaxSampler,
            '_model_runner',
            new_callable=mock.PropertyMock,
            return_value=model_runner,
        ),
        mock.patch.object(
            sglang_jax_sampler.utils,
            'transfer_state_with_mappings',
            return_value='new_state',
        ) as transfer_mock,
        mock.patch.object(
            sglang_jax_sampler.jax.tree_util,
            'tree_flatten',
            return_value=(['leaf'], None),
        ),
    ):
      sampler.update_params(updated_weights='src_state')

    transfer_mock.assert_called_once_with(
        src_state='src_state',
        dst_state='dst_state',
        key_mappings=sampler.mappings,
        key_mapping_hook_fns=sampler.to_hf_hook_fns,
        transpose_keys=sampler.to_hf_transpose_keys,
        reshard_fn=sampler._reshard_params_to_engine,
    )
    self.assertEqual(model_runner.model_state_leaves, ['leaf'])

  def test_reshard_params_to_engine_moves_arrays_to_host_first(self):
    source = {'w': np.arange(4, dtype=np.float32).reshape(2, 2)}
    dst_shardings = mock.sentinel.dst_shardings

    with mock.patch.object(
        sglang_jax_sampler.reshard,
        'reshard_pytree',
        return_value='resharded',
    ) as reshard_mock:
      result = sglang_jax_sampler.SglangJaxSampler._reshard_params_to_engine(
          source, dst_shardings
      )

    self.assertEqual(result, 'resharded')
    passed_source, passed_dst = reshard_mock.call_args.args
    self.assertIsInstance(passed_source['w'], np.ndarray)
    self.assertIs(passed_dst, dst_shardings)

  def test_close_cancels_tokenizer_manager_asyncio_tasks(self):
    sampler = self._build_sampler(outputs=[])
    loop = asyncio.new_event_loop()

    async def _wait_forever():
      await asyncio.Event().wait()

    task = loop.create_task(_wait_forever())
    sampler.engine.loop = loop
    sampler.engine.tokenizer_manager = mock.Mock(asyncio_tasks={task})

    try:
      sampler.close()
    finally:
      if not task.done():
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
      loop.close()

    self.assertTrue(task.cancelled())
    self.assertEqual(sampler.engine.tokenizer_manager.asyncio_tasks, set())
    sampler.engine.shutdown.assert_called_once_with()


if __name__ == "__main__":
  absltest.main()
