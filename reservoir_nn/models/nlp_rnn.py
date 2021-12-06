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

"""RNN-based Reservior Models for Natural Language Processing."""

from typing import Any, Dict, Optional

from immutabledict import immutabledict
import numpy as np
from reservoir_nn.keras import layers
import tensorflow as tf

KERAS_RNN_LAYERS = immutabledict({
    "LSTM": tf.keras.layers.LSTM,
    "RNN": tf.keras.layers.SimpleRNN,
})

# These are reservoir models for time series data.
RNN_RESERVOIR_LAYER_NAMES = frozenset({"DenseReservoirRNNTemporal"})


def recurrent_reservoir_language_model(
    reservoir_weight: Optional[np.ndarray] = None,
    vocab_size: int = 20000,
    embed_dim: int = 128,
    reservoir_params: Optional[Dict[str, Any]] = None,
    layer_name: str = "DenseReservoirRNNTemporal",
    units: int = 200,
    learning_rate: float = 0.001) -> tf.keras.Sequential:
  """Build an RNN model for language modeling.

  Args:
    reservoir_weight: weights to use in the reservoir of the model. Ignored for
      keras models. Required for reservoir models
    vocab_size: number of tokens in the vocabulary.
    embed_dim: size of word embeddings to train for each vocab entry (input to
      LSTM cells).
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
        'state_discount': 1.0,
        }
      If variable not included in the params, the default values are used.)
      Ignored for keras models.
    layer_name: the kind of RNN layer to build. Must be in KERAS_RNN_LAYERS or
      RNN_RESERVOIR_LAYER_NAMES.
    units: The number of units in RNN cells for keras models.
    learning_rate: The learning rate for the model.

  Returns:
    RNN language model instance.
  """
  rnn_layer = get_rnn_layer(
      reservoir_weight=reservoir_weight,
      reservoir_params=reservoir_params,
      layer_name=layer_name,
      units=units,
      return_sequences=True)
  model = tf.keras.Sequential([
      tf.keras.layers.Input((None,)),
      tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim),
      rnn_layer,
      tf.keras.layers.Dense(vocab_size)
  ])

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=loss_fn)
  return model


def recurrent_reservoir_nlp_classifier(
    reservoir_weight: Optional[np.ndarray],
    num_classes: int = 2,
    vocab_size: int = 20000,
    embed_dim: int = 128,
    reservoir_params: Optional[Dict[str, Any]] = None,
    layer_name: str = "DenseReservoirRNNTemporal",
    units: int = 200,
    learning_rate: float = 0.001) -> tf.keras.Model:
  """Build an RNN model for NLP sentence classification tasks.

  Args:
    reservoir_weight: weights to use in the reservoir of the model. Ignored for
      keras models. Must be set for reservoir models.
    num_classes: the number of target classes for sentence classification.
    vocab_size: number of tokens in the vocabulary.
    embed_dim: size of word embeddings to train for each vocab entry (input to
      LSTM cells).
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
        'state_discount': 1.0,
        }
        If variable not included in the params, the default values are used.)
        Ignored for keras models.
    layer_name: the kind of RNN layer to build. Must be in KERAS_RNN_LAYERS or
      RNN_RESERVOIR_LAYER_NAMES.
    units: The number of units in RNN cells for keras models.
    learning_rate: The learning rate for the model.

  Returns:
    Sentence classification RNN model instance.
  """
  rnn_layer = get_rnn_layer(
      reservoir_weight=reservoir_weight,
      reservoir_params=reservoir_params,
      layer_name=layer_name,
      units=units,
      return_sequences=False)

  inputs = tf.keras.layers.Input((None,))
  x = tf.keras.layers.Embedding(
      input_dim=vocab_size, output_dim=embed_dim)(
          inputs)

  x = rnn_layer(x)
  outputs = tf.keras.layers.Dense(num_classes)(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=loss_fn,
      metrics=[
          tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
          tf.keras.metrics.SparseCategoricalCrossentropy(
              name="sparse_categorical_crossentropy")
      ])
  return model


def get_rnn_layer(reservoir_weight: Optional[np.ndarray] = None,
                  reservoir_params: Optional[Dict[str, Any]] = None,
                  layer_name: str = "LSTM",
                  units: int = 200,
                  return_sequences: bool = False) -> tf.keras.layers.Layer:
  """Build an RNN layer.

  Args:
    reservoir_weight: weights to use in the reservoir of the model. Ignored for
      keras models. Must be set for reservoir models.
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a correct argument for the reservoir base, e.g.
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
        default values are used.) Ignored for keras models.
    layer_name: the kind of RNN layer to build. Must be in KERAS_RNN_LAYERS or
      RNN_RESERVOIR_LAYER_NAMES.
    units: The number of units in RNN cells for keras models.
    return_sequences: When true, the layer returns outputs for each item in
      sequence.

  Returns:
    An rnn layer.

  Raises:
    ValueError when layer_name is not in KERAS_RNN_LAYERS or
      RNN_RESERVOIR_LAYER_NAMES.
  """
  if layer_name in RNN_RESERVOIR_LAYER_NAMES:
    if reservoir_params is None:
      reservoir_params = {}

    layer_cell = layers.DenseReservoirRecurrentCell(
        weight=reservoir_weight, **reservoir_params)
    return tf.keras.layers.RNN(layer_cell, return_sequences=return_sequences)

  elif layer_name in KERAS_RNN_LAYERS:
    return KERAS_RNN_LAYERS[layer_name](
        units, return_sequences=return_sequences)
  else:
    valid_layer_names = set(
        list(KERAS_RNN_LAYERS.keys()) + list(RNN_RESERVOIR_LAYER_NAMES))
    raise ValueError(f"Layer name '{layer_name}' not supported. " +
                     f"Must be one of: {valid_layer_names}")
