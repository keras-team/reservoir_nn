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

"""Tests for sideways_time."""

from absl.testing import absltest
import numpy as np
from reservoir_nn.models import sideways_time
import tensorflow as tf


class SidewaysTimeTest(absltest.TestCase):

  def test_sidewise_time_layer_outputs_correct_shape(self):
    batch_size = 4
    maxlen = 3
    embed_dim = 2
    input_layer = tf.keras.Input((maxlen, embed_dim))
    test_layer = sideways_time.SidewaysTime(order=2)(input_layer)
    model = tf.keras.Model(input_layer, test_layer)

    input_data = np.ones((batch_size, maxlen, embed_dim), dtype=np.float32)
    output_data = model.predict(input_data)
    feature_size = maxlen * embed_dim
    expected_shape = (batch_size,
                      int(feature_size + feature_size * (feature_size + 1) / 2))
    self.assertEqual(output_data.shape, expected_shape)

  def test_sideways_time_classifier_outputs_correct_shape(self):
    weights = np.ones((42, 42))
    num_classes = 2
    batch_size = 5
    vocab_size = 50
    maxlen = 20
    embed_dim = 10
    model = sideways_time.sideways_time_reservoir_classifier(
        reservoir_weight=weights,
        num_classes=num_classes,
        embed_dim=embed_dim,
        maxlen=maxlen,
        vocab_size=vocab_size)

    x_data = np.ones((batch_size, maxlen))
    y_data = np.ones(batch_size)
    model.fit(x_data, y_data)
    result = model(x_data)
    self.assertEqual(result.shape, (batch_size, num_classes))


if __name__ == '__main__':
  absltest.main()
