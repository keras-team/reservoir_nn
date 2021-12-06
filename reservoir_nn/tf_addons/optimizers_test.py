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

"""Tests for reservoir_nn.tf_addons.optimizers."""

from absl.testing import absltest
from reservoir_nn.tf_addons import optimizers
import tensorflow as tf


class SerializableAdamWTest(absltest.TestCase):

  def test_it_serializes_scalar_weight_decay(self):
    opt = optimizers.SerializableAdamW(weight_decay=1.0)
    self.assertEqual(opt.get_config()['weight_decay'], 1.0)

  def test_it_serializes_tensor_weight_decay(self):
    opt = optimizers.SerializableAdamW(weight_decay=tf.Variable(1.0))
    self.assertEqual(opt.get_config()['weight_decay'], 1.0)

  def test_it_serializes_callable_weight_decay(self):
    opt = optimizers.SerializableAdamW(weight_decay=lambda: tf.Variable(1.0))
    self.assertEqual(opt.get_config()['weight_decay'], 1.0)


if __name__ == '__main__':
  absltest.main()
