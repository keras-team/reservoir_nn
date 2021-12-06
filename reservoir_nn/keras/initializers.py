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

"""Code to update keras initializers.

Function is designed to imitate the previous behavior of
tf.keras.initializers.RandomNormal with a set seed for deterministic
initialization. Behavior of this keras function was changed in cl/392092094 and
going forward tf.random.Generator is recommended
to set a seed for the whole experiment.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class FixedRandomInitializer(tf.keras.initializers.Initializer):
  """Creates a random initializer with a fixed seed."""

  def __init__(self, seed, mean=0., stddev=1.):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed

  def __call__(self, shape, dtype=None):
    return tf.random.stateless_normal(
        shape,
        mean=self.mean,
        stddev=self.stddev,
        dtype=dtype,
        seed=[self.seed, 0])

  def get_config(self):
    return {'seed': self.seed, 'mean': self.mean, 'stddev': self.stddev}
