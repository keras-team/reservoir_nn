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

"""Module for using tensorflow_addons optimizers."""

import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers


class SerializableAdamW(tfa_optimizers.AdamW):
  """Makes the optimizer AdamW serializable."""

  def get_config(self):
    config = tf.keras.optimizers.Adam.get_config(self)

    config.update({
        "weight_decay": self._fixed_serialize_hyperparameter("weight_decay"),
    })

    return config

  def _fixed_serialize_hyperparameter(self, hyperparameter_name):
    """Serialize a hyperparameter that can be a float, callable, or Tensor."""
    value = self._hyper[hyperparameter_name]

    # First resolve the callable
    if callable(value):
      value = value()

    if isinstance(value, tf.keras.optimizers.schedules.LearningRateSchedule):
      return tf.keras.optimizers.schedules.serialize(value)

    if tf.is_tensor(value):
      return tf.keras.backend.get_value(value)

    return value
