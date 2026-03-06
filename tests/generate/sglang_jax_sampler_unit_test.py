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

from absl.testing import absltest
import numpy as np
from tunix.generate import sglang_jax_sampler


class SglangJaxSamplerUnitTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._sampler = sglang_jax_sampler.SglangJaxSampler.__new__(
        sglang_jax_sampler.SglangJaxSampler
    )

  def test_normalize_output_ids_truncates_and_flattens(self):
    output_ids = [[1, 2, 3, 4, 5]]
    normalized = self._sampler._normalize_output_ids(
        output_ids, max_generation_steps=4
    )
    np.testing.assert_array_equal(
        normalized,
        np.array([1, 2, 3, 4], dtype=np.int32),
    )

  def test_normalize_output_text_extracts_scalar(self):
    output_text = ["first", "second"]
    normalized = self._sampler._normalize_output_text(output_text)
    self.assertEqual(normalized, "first")


if __name__ == "__main__":
  absltest.main()
