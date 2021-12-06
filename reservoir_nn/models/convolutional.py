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

"""Convolutional layers."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from reservoir_nn.keras import initializers
from reservoir_nn.keras import reservoir_registry
import tensorflow as tf


# comparing to a very small float (1e-12) is safer than comparing to zero:
_VERY_SMALL_FLOAT = 1e-12


def inception_inspired_reservoir_model(
    input_shape: Tuple[int, int, int],
    reservoir_weight: np.ndarray,
    num_output_channels: int,
    seed: Optional[int] = None,
    num_filters: int = 32,
    reservoir_base: str = 'DenseReservoir',
    reservoir_params: Optional[Dict[str, Any]] = None,
    final_activation: Optional[str] = 'sigmoid',
    task: str = 'segmentation',
) -> tf.keras.Model:
  """Builds a simple recurrent reservoir model with inception-style head.

  The model is an SRN in the sense that a copy of the output of a first
  reservoir is passed through a set of trainable weights and then through
  a second identical reservoir.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
    reservoir_weight:  Weight matrix to be assigned to the fixed layers.
    num_output_channels:  how many output channels to use.
    seed:  int seed to use to get a deterministic set of "random" weights.
    num_filters: how many filters to include in each layer of the inception
      block.
    reservoir_base: the reservoir base to use. Default is 'DenseReservoir'.
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a Correct argument for the reservoir base,
      e.g. common options include {
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
      }. If variable not included in the params, the default values are used.)
    final_activation: 'sigmoid', 'softmax', 'tanh', or None.
    task: which task this model is used for (options includes: 'segmentation',
      'classification')

  Returns:
    A simple recurrent reservoir model with convolutional head

  Raises:
    ValueError: if task not in accepted tasks (segmentation, classification).
  """

  if task not in ['segmentation', 'classification']:
    raise ValueError(
        f'Task not defined in accepted tasks (segmentation, classification). Got {task}'
    )

  # Create a sequential keras model

  if reservoir_params is None:
    reservoir_params = {}
  reservoir_params['weight'] = reservoir_weight

  inputs = tf.keras.layers.Input(input_shape)

  if seed:
    kernel_initializer = initializers.FixedRandomInitializer(seed=seed)
  else:
    kernel_initializer = tf.keras.initializers.RandomNormal()

  # Inception 'stem'
  x = tf.keras.layers.Conv2D(
      num_filters, 8, padding='same', input_shape=input_shape,
      activation='elu')(
          inputs)

  x = tf.keras.layers.MaxPooling2D(
      pool_size=(3, 3), strides=(1, 1), padding='same')(
          x)

  x = tf.keras.layers.Conv2D(
      num_filters, 1, activation='elu', padding='same')(
          x)

  x = tf.keras.layers.Conv2D(
      num_filters, 3, activation='elu', padding='same')(
          x)

  x = tf.keras.layers.MaxPooling2D(
      pool_size=(3, 3), strides=(1, 1), padding='same')(
          x)

  x = tf.keras.layers.Conv2D(
      num_filters, 1, activation='elu', padding='same')(
          x)
  x = tf.keras.layers.Conv2D(
      num_filters, 3, activation='elu', padding='same')(
          x)

  # Inception block
  incepta = tf.keras.layers.Conv2D(
      num_filters, [1, 1], strides=(1, 1), activation='elu', padding='same')(
          x)
  incepta = tf.keras.layers.Conv2D(
      num_filters, [5, 5], strides=(1, 1), activation='elu', padding='same')(
          incepta)
  inceptb = tf.keras.layers.Conv2D(
      num_filters, [1, 1], strides=(1, 1), activation='elu', padding='same')(
          x)
  inceptb = tf.keras.layers.Conv2D(
      num_filters, [3, 3], strides=(1, 1), activation='elu', padding='same')(
          inceptb)
  inceptc = tf.keras.layers.MaxPooling2D(
      pool_size=(3, 3), strides=(1, 1), padding='same')(
          x)
  inceptc = tf.keras.layers.Conv2D(
      num_filters, [1, 1], strides=(1, 1), activation='elu', padding='same')(
          inceptc)
  inceptd = tf.keras.layers.Conv2D(
      num_filters, [1, 1], strides=(1, 1), activation='elu', padding='same')(
          x)

  y = tf.concat([incepta, inceptb, inceptc, inceptd], -1)

  # Dense layer
  y = tf.keras.layers.Dense(reservoir_weight.shape[0], activation='elu')(y)

  # The first reservoir layer
  y = reservoir_registry.get_reservoir(reservoir_base)(**reservoir_params)(y)

  # Trainable layer in between reservoirs
  y = tf.keras.layers.Dense(reservoir_weight.shape[0], activation='elu')(y)

  # The second fixed reservoir layer
  y = reservoir_registry.get_reservoir(reservoir_base)(**reservoir_params)(y)

  # Create outputs.
  if task == 'classification':
    y = tf.keras.layers.Flatten()(y)

  outputs = tf.keras.layers.Dense(
      units=num_output_channels,
      activation=final_activation,
      kernel_initializer=kernel_initializer)(
          y)

  model = tf.keras.models.Model(inputs, outputs)

  return model


def unet_reservoir_sandwich_model(
    input_shape: Tuple[int, ...],
    reservoir_weight: np.ndarray,
    num_output_channels: int,
    reservoir_base: str = 'DenseReservoir',
    reservoir_params: Optional[Dict[str, Any]] = None,
    final_activation: Optional[str] = 'sigmoid',
    noise_layer_stddev: float = 0.0,
    task: str = 'segmentation',
    match_tutorial: bool = False,
) -> tf.keras.Model:
  """Builds a reservoir-on-the-bottom network with a unet backbone.

  Args:
    input_shape:  tuple describing the shape of the input (e.g., (32,32))
    reservoir_weight:  reservoir weights to use.
    num_output_channels:  how many output channels to use.
    reservoir_base: the reservoir base to use. Default is 'DenseReservoir'.
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a Correct argument for the reservoir base,
      e.g. common options include {
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
      }. If variable not included in the params, the default values are used.)
    final_activation: 'sigmoid', 'softmax', 'tanh', or None.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.
    task: which task this model is used for (options includes: 'segmentation',
      'classification')
    match_tutorial: Whether to remove all the layers around and including the
      reservoir that aren't included in the keras unet tutorial. Note that this
      will substantially reduce the number of parameters and the ablation
      library is recommended for tighter comparison.

  Returns:
    the model

  Raises:
    ValueError: if task not in accepted tasks (segmentation, classification).
  """

  if task not in ['segmentation', 'classification']:
    raise ValueError(
        f'Task not defined in accepted tasks (segmentation, classification). Got {task}'
    )

  if reservoir_params is None:
    reservoir_params = {}
  reservoir_params['weight'] = reservoir_weight

  # This is to decide whether to use Conv1d, Conv2d, or Conv3d
  num_dims = len(input_shape) - 1

  inputs = tf.keras.Input(shape=input_shape)

  # The first half of the network is downsampling the input
  if num_dims == 2:
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
  elif num_dims == 3:
    x = tf.keras.layers.Conv3D(32, 3, strides=2, padding='same')(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  previous_block_activation = x  # Set aside residual

  for filters in [64, 128, 256]:
    if num_dims == 2:
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
      x = tf.keras.layers.BatchNormalization(name=f'cross_layer_{filters}')(x)
    elif num_dims == 3:
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv3D(
          filters, 3, groups=int(filters / 2), padding='same')(
              x)
      x = tf.keras.layers.TimeDistributed(
          tf.keras.layers.Conv2D(filters, (1, 1), padding='same'))(
              x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv3D(
          filters, 3, groups=int(filters / 2), padding='same')(
              x)
      x = tf.keras.layers.TimeDistributed(
          tf.keras.layers.Conv2D(filters, (1, 1), padding='same'))(
              x)
      x = tf.keras.layers.BatchNormalization(name=f'cross_layer_{filters}')(x)

    # Layers that will "cross" the u
    if filters == 64:
      crossu_64 = x
    if filters == 128:
      crossu_128 = x
    if filters == 256:
      crossu_256 = x

    if num_dims == 2:
      x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
      residual = tf.keras.layers.Conv2D(
          filters, 1, strides=2, padding='same')(
              previous_block_activation)
    elif num_dims == 3:
      x = tf.keras.layers.MaxPooling3D(3, strides=2, padding='same')(x)
      residual = tf.keras.layers.Conv3D(
          filters, 1, strides=2, padding='same')(
              previous_block_activation)

    x = tf.keras.layers.add([x, residual])

    previous_block_activation = x  # Set aside the next residual

  if not match_tutorial:
    # Add a reservoir sandwich at the bottom:
    x = tf.keras.layers.Dense(reservoir_weight.shape[0], activation='elu')(x)

    x = reservoir_registry.get_reservoir(reservoir_base)(**reservoir_params)(x)

    if noise_layer_stddev > _VERY_SMALL_FLOAT:
      x = tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev)(x)

    if num_dims == 2:
      x = tf.keras.layers.Conv2D(256, 1, 1, padding='same')(x)
    elif num_dims == 3:
      x = tf.keras.layers.Conv3D(256, 1, 1, padding='same')(x)

  # The second half of the network is upsampling the inputs
  for filters in [256, 128, 64, 32]:
    x = tf.keras.layers.Activation('relu')(x)
    if num_dims == 2:
      x = tf.keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.UpSampling2D(2)(x)
    elif num_dims == 3:
      x = tf.keras.layers.Conv3DTranspose(filters, 3, padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv3DTranspose(filters, 3, padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.UpSampling3D(2)(x)

    # Project residual
    if num_dims == 2:
      residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
      residual = tf.keras.layers.Conv2D(filters, 1, padding='same')(residual)
    elif num_dims == 3:
      residual = tf.keras.layers.UpSampling3D(2)(previous_block_activation)
      residual = tf.keras.layers.Conv3D(filters, 1, padding='same')(residual)

    if filters == 256:
      x = tf.keras.layers.add([x, residual, crossu_256])
    if filters == 128:
      x = tf.keras.layers.add([x, residual, crossu_128])
    if filters == 64:
      x = tf.keras.layers.add([x, residual, crossu_64])
    else:
      x = tf.keras.layers.add([x, residual])
    previous_block_activation = x

  # Per-pixel classification layer
  # Note: output activation function is tanh to allow for contrastive labels
  # with minimal data munging (to try to prevent mistakes in the somewhat-tricky
  # contrastive label creation).
  if task == 'segmentation':
    if num_dims == 2:
      outputs = tf.keras.layers.Conv2D(
          num_output_channels, 3, activation=final_activation, padding='same')(
              x)
    elif num_dims == 3:
      outputs = tf.keras.layers.Conv3D(
          num_output_channels, 3, activation=final_activation, padding='same')(
              x)
  elif task == 'classification':
    outputs = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(units=num_output_channels)(outputs)

  model = tf.keras.Model(inputs, outputs)
  return model
