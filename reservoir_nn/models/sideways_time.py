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

"""Sideways time models.

This is inspired by Gauthier et al. (2021).
Link: https://arxiv.org/pdf/2106.07688.pdf
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from reservoir_nn.keras import reservoir_registry
import tensorflow as tf


class SidewaysTime(tf.keras.layers.Layer):
  """Add nonlinear features as in Gauthier et al. (2021)."""

  def __init__(self, order: int = 2):
    super(SidewaysTime, self).__init__()
    if order != 2:
      raise ValueError(
          "Sideways time for orders other than 2 is not yet implemented.")
    self.order = order

  def call(self, data: tf.Tensor) -> tf.Tensor:
    input_shape = tf.shape(data)
    batch_size = input_shape[0]
    maxlen = input_shape[1]
    embed_dim = input_shape[2]

    # Concatenate embeddings within each sequence in batch.
    data_linear = tf.reshape(data, (-1, maxlen * embed_dim))

    # Take outer product.
    # data_nonlinear has shape
    # (batch_size, embed_dim * maxlen, embed_dim * maxlen)
    data_nonlinear = tf.einsum("ki,kj->kij", data_linear, data_linear)

    # Take upper triangle of outer product matrix, so each pair of features
    # is only considered once.
    triangular_mask = self.get_triangular_mask(batch_size, maxlen, embed_dim)
    data_nonlinear = tf.gather_nd(data_nonlinear, tf.where(triangular_mask))
    data_nonlinear = tf.reshape(data_nonlinear, (batch_size, -1))

    # Concatenate linear and nonlinear features.
    result = tf.concat([data_linear, data_nonlinear], -1)
    result = tf.ensure_shape(result,
                             self.compute_output_shape(data.shape.as_list()))
    return result

  def get_triangular_mask(self, batch_size, maxlen, embed_dim):
    i = tf.range(maxlen * embed_dim)[:, tf.newaxis]
    j = tf.range(maxlen * embed_dim)
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, maxlen * embed_dim, maxlen * embed_dim))
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1),
         tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    return tf.tile(mask, mult)

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0]
    maxlen = input_shape[1]
    embed_dim = input_shape[2]
    feature_dim = maxlen * embed_dim
    return (batch_size, int(feature_dim + feature_dim * (feature_dim + 1) / 2))


def sideways_time_reservoir_classifier(
    reservoir_weight: np.ndarray,
    num_classes: int = 2,
    vocab_size: int = 20000,
    embed_dim: int = 128,
    maxlen: int = 80,
    reservoir_params: Optional[Dict[str, Any]] = None,
    reservoir_base: str = "DenseReservoir") -> tf.keras.Model:
  """Build a reservoir-base SidewaysTime model for sentence classification.

  Args:
    reservoir_weight: weights to use in the reservoir of the model.
    num_classes: the number of target classes for sentence classification.
    vocab_size: number of tokens in the vocabulary.
    embed_dim: size of word embeddings to train for each vocab entry.
    maxlen: maximum sequence length (number of tokens in a sentence).
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a Correct argument for the reservoir base, e.g.
      common options include {
      'recurrence_degree': 3,
      'keep_memory': True,
      'trainable_reservoir': True,
      'use_bias': True,
      'activation_within_recurrence': True,
      'kernel_local_learning': 'hebbian',
      'kernel_local_learning_params': {'eta': 0.1},
      'recurrent_kernel_local_learning': 'hebbian',
      'recurrent_kernel_local_learning_params': {'eta': 0.1},
      'state_discount': 1.0, }. If variable not included in the params, the
        default values are used.)
    reservoir_base: the reservoir base to use.

  Returns:
    A reservoir model instance.
  """
  reservoir_layer = reservoir_registry.get_reservoir(reservoir_base)
  if reservoir_params is None:
    reservoir_params = {}

  inputs = tf.keras.layers.Input((maxlen,))
  x = tf.keras.layers.Embedding(
      input_dim=vocab_size, output_dim=embed_dim)(
          inputs)
  x = SidewaysTime()(x)
  logging.info("x.shape=%s", str(x.shape))

  # Reservoir sandwich
  x = tf.keras.layers.Dense(reservoir_weight.shape[0])(x)
  x = reservoir_layer(reservoir_weight, **reservoir_params)(x)
  outputs = tf.keras.layers.Dense(num_classes)(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(
      "adam",
      loss=loss_fn,
      metrics=[
          tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
          tf.keras.metrics.SparseCategoricalCrossentropy(
              name="sparse_categorical_crossentropy")
      ])
  return model
