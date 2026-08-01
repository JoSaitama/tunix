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
from unittest import mock

from absl.testing import absltest
from tunix.oss import utils


class HfPipelineTest(absltest.TestCase):

  @mock.patch.object(utils.hf, 'hf_hub_download', autospec=True)
  @mock.patch.object(utils.hf, 'list_repo_files', autospec=True)
  def test_complete_local_model_skips_huggingface_calls(
      self, list_repo_files, hf_hub_download
  ):
    with tempfile.TemporaryDirectory() as temp_dir:
      model_dir = Path(temp_dir)
      (model_dir / 'config.json').write_text('{}', encoding='utf-8')
      (model_dir / 'model.safetensors').touch()

      result = utils.hf_pipeline('organization/model', temp_dir)

    self.assertEqual(result, temp_dir)
    list_repo_files.assert_not_called()
    hf_hub_download.assert_not_called()

  @mock.patch.object(utils.hf, 'hf_hub_download', autospec=True)
  @mock.patch.object(utils.hf, 'list_repo_files', autospec=True)
  def test_incomplete_local_model_uses_huggingface(
      self, list_repo_files, hf_hub_download
  ):
    list_repo_files.return_value = ['config.json', 'model.safetensors']
    with tempfile.TemporaryDirectory() as temp_dir:
      Path(temp_dir, 'config.json').write_text('{}', encoding='utf-8')

      result = utils.hf_pipeline('organization/model', temp_dir)

    self.assertEqual(result, temp_dir)
    list_repo_files.assert_called_once_with('organization/model', token=None)
    self.assertEqual(hf_hub_download.call_count, 2)


if __name__ == '__main__':
  absltest.main()
