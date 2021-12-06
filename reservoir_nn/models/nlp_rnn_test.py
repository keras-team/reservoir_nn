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

"""Tests for nlp_rnn."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from reservoir_nn.models import nlp_rnn

LAYER_NAMES = frozenset(["DenseReservoirRNNTemporal", "RNN", "LSTM"])


class NlpRnnTest(parameterized.TestCase):

  @parameterized.parameters(LAYER_NAMES)
  def test_lstm_language_model_outputs_correct_shape(self, model_name):
    weights = np.ones((42, 42))
    batch_size = 5
    vocab_size = 50
    maxlen = 20
    embed_dim = 10
    model = nlp_rnn.recurrent_reservoir_language_model(
        layer_name=model_name,
        reservoir_weight=weights,
        embed_dim=embed_dim,
        vocab_size=vocab_size)

    x_data = np.ones((batch_size, maxlen))
    model.fit(x_data, x_data)

    result = model.predict(x_data)

    self.assertEqual(result.shape, (batch_size, maxlen, vocab_size))

  @parameterized.parameters(LAYER_NAMES)
  def test_lstm_classifier_outputs_correct_shape(self, model_name):
    weights = np.ones((42, 42))
    num_classes = 2
    batch_size = 5
    vocab_size = 50
    maxlen = 20
    embed_dim = 10
    model = nlp_rnn.recurrent_reservoir_nlp_classifier(
        layer_name=model_name,
        reservoir_weight=weights,
        num_classes=num_classes,
        embed_dim=embed_dim,
        vocab_size=vocab_size)

    x_data = np.ones((batch_size, maxlen))
    y_data = np.ones(batch_size)
    model.fit(x_data, y_data)

    result = model(x_data)

    self.assertEqual(result.shape, (batch_size, num_classes))


if __name__ == "__main__":
  absltest.main()
