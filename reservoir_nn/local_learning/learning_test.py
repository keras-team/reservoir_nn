# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for local learning."""

from absl.testing import absltest
import numpy as np

from reservoir_nn.local_learning import learning


class LearningTest(absltest.TestCase):

  def test_hebbian_learning_works(self):
    batch_size = 10
    hidden_size = 20
    activation = np.random.random((batch_size, hidden_size))
    weight = np.random.random((hidden_size, hidden_size))
    params = {'eta': 0.1}
    new_weight = learning.LocalLearning(learning_rule='hebbian')(params,
                                                                 activation,
                                                                 weight)
    self.assertEqual(weight.shape, new_weight.shape)

  def test_oja_learning_works(self):
    batch_size = 10
    hidden_size = 20
    activation = np.random.random((batch_size, hidden_size))
    weight = np.random.random((hidden_size, hidden_size))
    params = {'eta': 0.1}
    new_weight = learning.LocalLearning(learning_rule='oja')(params, activation,
                                                             weight)
    self.assertEqual(weight.shape, new_weight.shape)

  def test_contrastive_hebbian_learning_works(self):
    batch_size = 10
    hidden_size = 20
    activation = np.random.random((batch_size, hidden_size))
    expected = np.random.random((batch_size, hidden_size))
    weight = np.random.random((hidden_size, hidden_size))
    params = {'eta': 0.1}
    new_weight = learning.LocalLearning(learning_rule='contrastive_hebbian')(
        params, activation, weight, expected)
    self.assertEqual(weight.shape, new_weight.shape)


if __name__ == '__main__':
  absltest.main()
