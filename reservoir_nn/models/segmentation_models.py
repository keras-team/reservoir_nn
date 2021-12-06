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

"""Fly segmentation models."""

import enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from reservoir_nn.keras import blocks
from reservoir_nn.keras import initializers
from reservoir_nn.keras import layers
from reservoir_nn.keras import reservoir_registry
from reservoir_nn.utils import weight_transforms
import tensorflow as tf


# comparing to a very small float (1e-12) is safer than comparing to zero:
_VERY_SMALL_FLOAT = 1e-12


def minimal_reservoir_model(
    input_shape: Tuple[int, int, int],
    reservoir_weight: np.ndarray,
    num_classes: int = 1,
    reservoir_base: str = 'DenseReservoir',
    trainable_reservoir: bool = False,
    recurrence_degree: int = 0,
    reservoir_activation=tf.keras.activations.relu,
    reservoir_use_bias: bool = False,
    noise_layer_stddev: float = 0.0,
    deterministic_initialization: bool = False,
    seed: int = 42,
    apply_batchnorm: bool = False,
    embedding_layer_dropout_rate: float = 0.0,
    reservoir_layer_dropout_rate: float = 0.0,
    embedding_kernel_regularizer: tf.keras.regularizers.Regularizer = None,
    reservoir_kernel_regularizer: tf.keras.regularizers.Regularizer = None,
    output_kernel_regularizer: tf.keras.regularizers.Regularizer = None
) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds a reservoir model for image segmentation.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
      The input layer is built to fit the size of the image.
    reservoir_weight: Weight matrix to be assigned to the reservoir layer.
    num_classes: The number of classes in the labels. Note that labels must be
      in a shape of (height, width, optional_color_channels, num_classes).
    reservoir_base: The reservoir base to be used. Default to be DenseReservoir.
    trainable_reservoir: If True, the reservoir layer is trainable. If False,
      weights are set non-trainable. A trainable DenseReservoir is similar to a
      Dense layer.
    recurrence_degree: The degree of recurrence in the reservoir layer. 0 for
      non-recurrence.
    reservoir_activation: Activation function used for the input and output of
      the reservoir layer.
    reservoir_use_bias: Whether the reservoir layer uses a bias vector.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.
    deterministic_initialization: Whether to use a fix seed to initialize the
      trainable parameters.
    seed: Seed to use if deterministic initialization.
    apply_batchnorm: Apply Batch Normalization after each layer.
    embedding_layer_dropout_rate: The fraction of the output units of the
      embedding layer to drop each training step.
    reservoir_layer_dropout_rate: The fraction of the output units of the
      reservoir layer to drop each training step.
    embedding_kernel_regularizer: Regularizer function that is applied to the
      kernel weights matrix of the embedding layer.
    reservoir_kernel_regularizer: Regularizer function that is applied to the
      kernel weights matrix of the reservoir layer.
    output_kernel_regularizer: Regularizer function that is applied to the
      kernel weights matrix of the output layer.

  Returns:
    The reservoir model as a keras model object.
  """
  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='reservoir_segmentation_model')

  # Input layer should have the shape of the input images
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))
  model.add(tf.keras.layers.Flatten(name='flatten_input'))

  if deterministic_initialization:
    kernel_initializer = initializers.FixedRandomInitializer(seed=seed)
  else:
    kernel_initializer = tf.keras.initializers.RandomNormal()

  # Trainable input embedding
  model.add(
      tf.keras.layers.Dense(
          reservoir_weight.shape[0],
          activation=reservoir_activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=embedding_kernel_regularizer,
          name='input_dense'))

  if apply_batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  # comparing rate to a small float of 1e-12 is safer than comparing to zero
  if embedding_layer_dropout_rate > _VERY_SMALL_FLOAT:
    model.add(tf.keras.layers.Dropout(rate=embedding_layer_dropout_rate))

  # The reservoir layer
  reservoir_layer_class = reservoir_registry.get_reservoir(reservoir_base)(
      weight=reservoir_weight,
      activation=reservoir_activation,
      use_bias=reservoir_use_bias,
      kernel_regularizer=reservoir_kernel_regularizer,
      trainable_reservoir=trainable_reservoir,
      recurrence_degree=recurrence_degree,
      name='reservoir_layer',
      input_shape=(None, reservoir_weight.shape[0]))

  model.add(reservoir_layer_class)

  if apply_batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  # comparing rate to a small float of 1e-12 is safer than comparing to zero
  if reservoir_layer_dropout_rate > _VERY_SMALL_FLOAT:
    model.add(tf.keras.layers.Dropout(rate=reservoir_layer_dropout_rate))

  # Noise regularization
  if noise_layer_stddev > _VERY_SMALL_FLOAT:
    model.add(
        tf.keras.layers.GaussianNoise(
            stddev=noise_layer_stddev,
            name='gaussian_noise',
        ))

  activation = 'sigmoid' if num_classes == 1 else 'relu'
  # Trainable output embedding
  model.add(
      tf.keras.layers.Dense(
          input_shape[0] * input_shape[1],
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=output_kernel_regularizer,
          name='output_dense'))

  # reshape the output layer
  model.add(tf.keras.layers.Reshape(input_shape, name='reshape_output'))

  if num_classes > 1:
    model.add(
        tf.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding='same',
            activation='softmax',
            name='output_convolution',
        ))

  return model


def build_simple_recurrent_model(
    input_shape: Tuple[int, int],
    reservoir_weight: np.ndarray,
    reservoir_activation=tf.keras.activations.relu,
    deterministic_initialization: bool = False,
    apply_batchnorm: bool = False,
    noise_layer_stddev: float = 0.1,
    dropout_rate: float = 0,
) -> tf.keras.Model:
  """Builds a reservoir model for image segmentation.

  Args:
    input_shape: (image_height, image_width) of the input image. The input layer
      is built to fit the size of the image.
    reservoir_weight: Weight matrix to be assigned to the fixed reservoir layer.
    reservoir_activation: Activation function used for the input and output of
      the reservoir layer.
    deterministic_initialization: Whether to use a fix seed to initialize the
      trainable parameters.
    apply_batchnorm: Apply Batch Normalization after each layer.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.
    dropout_rate:  The dropout layer randomly sets input units to 0 with a
      frequency of dropout_rate at each step during training time, which helps
      prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate)
      such that the sum over all inputs is unchanged.

  Returns:
    The reservoir model as a keras model object.
  """

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='simple_recurrent_segmentation_model')

  # Input layer should have the shape of a single square image tile
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))
  model.add(tf.keras.layers.Flatten(name='flatten_input'))

  if deterministic_initialization:
    kernel_initializer = tf.keras.initializers.RandomNormal(seed=42)
  else:
    kernel_initializer = tf.keras.initializers.RandomNormal()

  # Trainable input embedding
  model.add(
      tf.keras.layers.Dense(
          reservoir_weight.shape[0],
          activation=reservoir_activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-7, l2=1e-6),
          name='input_dense'))

  if dropout_rate > _VERY_SMALL_FLOAT:
    model.add(tf.keras.layers.Dropout(dropout_rate))

  if apply_batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  # Noise regularization
  if noise_layer_stddev > _VERY_SMALL_FLOAT:
    model.add(
        tf.keras.layers.GaussianNoise(
            stddev=noise_layer_stddev, name='gaussian_noise_1'))

  # The first fixed reservoir layer
  model.add(
      layers.DenseReservoir(
          reservoir_weight,
          activation=reservoir_activation,
          trainable=False,
          name='reservoir_dense'))

  if apply_batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  # Trainable layer in between reservoir steps
  model.add(
      tf.keras.layers.Dense(reservoir_weight.shape[0], name='hidden_dense'))

  if apply_batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  # Noise regularization
  if noise_layer_stddev > _VERY_SMALL_FLOAT:
    model.add(
        tf.keras.layers.GaussianNoise(
            stddev=noise_layer_stddev, name='gaussian_noise_2'))

  # The second reservoir layer is a copy of the first, that's what makes this
  # simple recurrent
  model.add(
      layers.DenseReservoir(
          reservoir_weight,
          activation=reservoir_activation,
          trainable=False,
          name='reservoir_dense_copy'))

  if apply_batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  # Trainable output layer
  model.add(
      tf.keras.layers.Dense(
          input_shape[0] * input_shape[1],
          activation='sigmoid',
          kernel_initializer=kernel_initializer,
          name='output_dense'))

  # Noise regularization
  if noise_layer_stddev > _VERY_SMALL_FLOAT:
    model.add(
        tf.keras.layers.GaussianNoise(
            stddev=noise_layer_stddev, name='gaussian_noise_3'))

  # reshape the output layer
  model.add(tf.keras.layers.Reshape(input_shape, name='reshape_output'))

  return model


def convolution_model(
    input_shape: Tuple[int, int, int],
    conv_filters: Tuple[int, ...],
    conv_kernel_sizes: Tuple[int, ...],
    conv_strides: Tuple[int, ...],
    upsample_conv_filters: Optional[Tuple[int, ...]] = None,
    upsample_conv_kernel_sizes: Optional[Tuple[int, ...]] = None,
    upsample_conv_strides: Optional[Tuple[int, ...]] = None,
    num_classes: int = 1,
    upsample_layer_dropout_rate: float = 0.0,
    noise_layer_stddev: float = 0.0,
):
  """Builds a model composed of DownSample and UpSample convolution layers.

  Note that the Dropout layers are implemented only if the input rate is larger
  than `_VERY_SMALL_FLOAT`. The same for the Noise layer that it is implemented
  only if the input Gaussian stddev is larger than `_VERY_SMALL_FLOAT`.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
      The input layer is built to fit the size of the image.
    conv_filters: The number of output filters of the layers in the DownSample
      convolution block.
    conv_kernel_sizes: The height and width of the 2D convolution windows in the
      layers of the DownSample block.
    conv_strides: The strides along the height and width of the MaxPoolings in
      the DownSample block.
    upsample_conv_filters: The number of output filters of the layers in the
      UpSample convolution block. If not provided, the reversed tuple of
      `conv_filters` is used.
    upsample_conv_kernel_sizes: The height and width of the 2D convolution
      windows in the layers of the UpSample block. If not provided, the reversed
      tuple of `conv_kernel_sizes` is used.
    upsample_conv_strides: The strides along the height and width of the
      Conv2DTranspose layers in the UpSample block. If not provided, the
      reversed tuple of `conv_strides` is used.
    num_classes: The number of classes in the labels. Note that labels must be
      in a shape of (height, width, optional_color_channels, num_classes).
    upsample_layer_dropout_rate: The fraction of the output units of the
      upsample convolution layers to drop each training step.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.

  Returns:
    The keras model object.
  """

  # Tuple parameters for convolution blocks must have the same num of elements
  if (len(conv_filters) != len(conv_kernel_sizes) or
      len(conv_filters) != len(conv_strides)):
    raise ValueError(
        f'`conv_filters`, `conv_kernel_sizes`, and `conv_strides` must have '
        f'the same number of elements, but got '
        f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

  # Tuple parameters for UpSample conv block
  upsample_conv_filters = upsample_conv_filters or conv_filters[::-1]
  upsample_conv_kernel_sizes = (
      upsample_conv_kernel_sizes or conv_kernel_sizes[::-1])

  # Check if UpSampling matches DownSampling
  if upsample_conv_strides:
    if tf.math.reduce_prod(upsample_conv_strides) != tf.math.reduce_prod(
        conv_strides):
      raise ValueError(
          f'Product of UpSample factors of {upsample_conv_strides} does not '
          f'match with that of DownSample factors of {conv_strides}')
  else:
    upsample_conv_strides = conv_strides[::-1]

  # Tuple parameters for UpSample block must have the same num of elements
  if (len(upsample_conv_filters) != len(upsample_conv_kernel_sizes) or
      len(upsample_conv_filters) != len(upsample_conv_strides)):
    raise ValueError(
        f'`upsample_conv_filters`, `upsample_conv_kernel_sizes`, and '
        f'`upsample_conv_strides` must have the same number of elements, but '
        f'got {len(upsample_conv_filters)}, {len(upsample_conv_kernel_sizes)}, '
        f'{len(upsample_conv_strides)}')

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='convolution_segmentation_model')

  # The input layer
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))

  # The DownSample Convolution block
  model.add(
      blocks.DownSampleConvolution(
          conv_filters=conv_filters,
          conv_kernel_sizes=conv_kernel_sizes,
          max_poolings=(True,) * len(conv_filters),
          conv_strides=conv_strides,
          activation='relu',
          apply_batchnorm=True,
          leaky=True,
          name='convolution_head',
      ))

  # Noise regularization after the DownSample block
  if noise_layer_stddev > _VERY_SMALL_FLOAT:
    model.add(tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev))

  # Upsampling convolution layers
  model.add(
      blocks.UpSampleConvolution(
          conv_filters=upsample_conv_filters,
          conv_kernel_sizes=upsample_conv_kernel_sizes,
          conv_strides=upsample_conv_strides,
          apply_batchnorm=True,
          dropout_rate=upsample_layer_dropout_rate,
          name='upsample_convolution',
      ))

  final_activation = 'sigmoid' if num_classes == 1 else 'softmax'

  # Last convolution layer for segmentation
  model.add(
      tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=1,
          padding='same',
          activation=final_activation,
          name='final_convolution_layer',
      ))

  return model


def convolution_reservoir_model(
    input_shape: Tuple[int, int, int],
    reservoir_weights: Tuple[np.ndarray, ...],
    reservoir_recurrence_degrees: Tuple[int, ...],
    trainable_reservoir: Tuple[bool, ...],
    conv_filters: Tuple[int, ...],
    conv_kernel_sizes: Tuple[int, ...],
    conv_strides: Tuple[int, ...],
    num_classes: int = 1,
    reservoir_base: str = 'DenseReservoir',
    reservoir_use_bias: bool = False,
    upsample_convolution: bool = False,
    embedding_layer_dropout_rate: float = 0.0,
    reservoir_layer_dropout_rate: float = 0.0,
    upsample_layer_dropout_rate: float = 0.0,
    noise_layer_stddev: float = 0.0,
    deterministic_initialization: bool = False,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
  """Builds a convolution, reservoir model for image segmentation.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
      The input layer is built to fit the size of the image.
    reservoir_weights: The weight matrices to be assigned to the reservoir
      layers that are implemented in a series.
    reservoir_recurrence_degrees: The degrees of recurrence in the corresponding
      reservoir layers. 0 for non-recurrence.
    trainable_reservoir: Whether the corresponding reservoir is trainable. If
      True, the reservoir layer is trainable. If False, weights are set
      non-trainable. A trainable DenseReservoir is similar to a Dense layer.
    conv_filters: The number of output filters of the layers in each convolution
      block. The tuple is reversed in the UpSampleConvolution block. Note the
      reversal is also applied to `conv_kernel_sizes` and `conv_strides`.
    conv_kernel_sizes: The height and width of the 2D convolution windows in the
      layers of each block.
    conv_strides: The strides along the height and width of the MaxPoolings in
      the DownSample and of the Conv2DTranspose layers in the UpSample block.
    num_classes: The number of classes in the labels. Note that labels must be
      in a shape of (height, width, optional_color_channels, num_classes).
    reservoir_base: The reservoir base to be used. Default to be DenseReservoir.
    reservoir_use_bias: Whether the reservoir layer(s) use(s) a bias vector.
    upsample_convolution: If True, adds `num_conv_layers` upsample convolution
      layers after the reservoir.
    embedding_layer_dropout_rate: The fraction of the output units of the
      embedding layer to drop each training step.
    reservoir_layer_dropout_rate: The fraction of the output units of the
      reservoir layers to drop each training step.
    upsample_layer_dropout_rate: The fraction of the output units of upsample
      convolution layers to drop each training step.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.
    deterministic_initialization: Whether to use a fix seed to initialize the
      trainable parameters.
    apply_batchnorm: Apply Batch Normalization after each layer.

  Returns:
    The reservoir model as a keras model object.
  """
  # Check if each reservoir has a corresponding recurrence degree
  if len(reservoir_weights) != len(reservoir_recurrence_degrees):
    raise ValueError(
        f'reservoir_weights has {len(reservoir_weights)} elements, but '
        f'reservoir_recurrence_degrees has {len(reservoir_recurrence_degrees)}')

  # Check if the reservoirs meet the recurrence requirement
  for recurrence_degree, weight in zip(reservoir_recurrence_degrees,
                                       reservoir_weights):
    if recurrence_degree > 0 and weight.shape[0] != weight.shape[1]:
      raise ValueError(
          f'There is a reservoir that does not meet the recurrence requirement.'
          f' It must be square, but the input got shape: {weight.shape}')

  # Check if the weight matrices are correctly chained
  for i in range(len(reservoir_weights) - 1):
    num_columns = reservoir_weights[i].shape[1]
    num_rows = reservoir_weights[i + 1].shape[0]
    if num_columns != num_rows:
      raise ValueError(
          f'Reservoir weight matrix {i} with {num_columns} columns cannot be '
          f'chained to matrix {i+1} with {num_rows} rows')

  # Tuple parameters for convolution blocks must have the same num of elements
  if (len(conv_filters) != len(conv_kernel_sizes) or
      len(conv_filters) != len(conv_strides)):
    raise ValueError(
        f'`conv_filters`, `conv_kernel_sizes`, and `conv_strides` must have '
        f'the same number of elements, but got '
        f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='convolution_segmentation_model')

  # The input layer
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))

  # Convolution head
  num_convolution_layers = len(conv_filters)
  model.add(
      blocks.DownSampleConvolution(
          conv_filters=conv_filters,
          conv_kernel_sizes=conv_kernel_sizes,
          max_poolings=(True,) * num_convolution_layers,
          conv_strides=conv_strides,
          activation='relu',
          apply_batchnorm=apply_batchnorm,
          leaky=True,
          name='convolution_head',
      ))

  # flatten the conv output
  model.add(tf.keras.layers.Flatten(name='flatten_conv_output'))

  if deterministic_initialization:
    kernel_initializer = tf.keras.initializers.RandomNormal(seed=42)
  else:
    kernel_initializer = tf.keras.initializers.RandomNormal()

  # Trainable input embedding
  model.add(
      tf.keras.layers.Dense(
          reservoir_weights[0].shape[0],
          activation=tf.keras.activations.relu,
          kernel_initializer=kernel_initializer,
          name='input_dense',
      ))

  if apply_batchnorm:
    model.add(tf.keras.layers.BatchNormalization())

  # comparing rate to a small float of 1e-12 is safer than comparing to zero
  if embedding_layer_dropout_rate > _VERY_SMALL_FLOAT:
    model.add(tf.keras.layers.Dropout(rate=embedding_layer_dropout_rate))

  # The reservoir layer(s)

  for i, (reservoir_weight, recurrence_degree, trainable) in enumerate(
      zip(reservoir_weights, reservoir_recurrence_degrees,
          trainable_reservoir)):
    reservoir_layer_class = reservoir_registry.get_reservoir(reservoir_base)(
        weight=reservoir_weight,
        activation=tf.keras.activations.relu,
        use_bias=reservoir_use_bias,
        trainable_reservoir=trainable,
        recurrence_degree=recurrence_degree,
        name=f'reservoir_layer_{i}',
    )
    model.add(reservoir_layer_class)

    if apply_batchnorm:
      model.add(tf.keras.layers.BatchNormalization())

    # comparing rate to a small float of 1e-12 is safer than comparing to zero
    if reservoir_layer_dropout_rate > _VERY_SMALL_FLOAT:
      model.add(tf.keras.layers.Dropout(rate=reservoir_layer_dropout_rate))

  # Noise regularization
  if noise_layer_stddev > _VERY_SMALL_FLOAT:
    model.add(
        tf.keras.layers.GaussianNoise(
            stddev=noise_layer_stddev,
            name='gaussian_noise',
        ))

  final_activation = 'sigmoid' if num_classes == 1 else 'softmax'

  if not upsample_convolution:
    # Trainable output
    model.add(
        tf.keras.layers.Dense(
            input_shape[0] * input_shape[1] * num_classes,
            activation=final_activation,
            kernel_initializer=kernel_initializer,
            name='output_dense'))

    # reshape the output layer
    model.add(
        tf.keras.layers.Reshape(
            (*input_shape[:2], num_classes),
            name='reshape_output',
        ))

    return model

  # if upsample conv layers are to be added
  height, width = (input_shape[0], input_shape[1])
  for conv_stride in conv_strides:
    height = (height - 1) // conv_stride + 1
    width = (width - 1) // conv_stride + 1

  # Trainable reverse-embedding layer
  model.add(
      tf.keras.layers.Dense(
          height * width * conv_filters[-1],
          activation='relu',
          kernel_initializer=kernel_initializer,
          name='reverse_embedding'))
  # Reshape the reverse-embedding layer
  model.add(
      tf.keras.layers.Reshape(
          (height, width, conv_filters[-1]),
          name='reshape_reverse_embedding',
      ))

  # Upsampling convolution layers
  model.add(
      blocks.UpSampleConvolution(
          conv_filters=conv_filters[::-1],
          conv_kernel_sizes=conv_kernel_sizes[::-1],
          conv_strides=conv_strides[::-1],
          apply_batchnorm=apply_batchnorm,
          dropout_rate=upsample_layer_dropout_rate,
      ))

  # Last convolution layer for segmentation
  model.add(
      tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=1,
          padding='same',
          activation=final_activation,
          name='last_convolution_layer',
      ))

  return model


def selective_sensor_model(
    input_shape: Tuple[int, int, int],
    reservoir_weights: Tuple[np.ndarray, ...],
    conv_filters: Tuple[int, ...],
    conv_kernel_sizes: Tuple[int, ...],
    conv_strides: Tuple[int, ...],
    reservoir_base: str = 'DenseReservoir',
    num_classes: int = 1,
    num_sensors_per_channel: int = 10,
    reservoir_layer_dropout_rate: float = 0.0,
    upsample_layer_dropout_rate: float = 0.0,
    noise_layer_stddev: float = 0.0,
    reservoir_params_set: Optional[Tuple[Dict[str, Any]]] = None,
) -> tf.keras.Model:
  """Builds a model using a SelectiveSensor to send input to the reservoirs.

  Model: Input => Convolution Head => SelectiveSensor => Reservoirs => Output
  The output of the Convo Head is reshaped into (batch, indices, channels). The
  dim of the output of SelectiveSensor (conv_filters * num_sensors_per_channel)
  must match with the first reservoir's number of rows.

  Note that the Dropout layers are implemented only if the input rate is larger
  than `_VERY_SMALL_FLOAT`. The same for the Noise layer that it is implemented
  only if the input Gaussian stddev is larger than `_VERY_SMALL_FLOAT`.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
      The input layer is built to fit the size of the image.
    reservoir_weights: The weight matrices to be assigned to the reservoir
      layers that are implemented in a series. A recurrent reservoir must be
      square.
    conv_filters: The number of output filters of the layers in each convolution
      block. The tuple is reversed in the UpSampleConvolution block. Note the
      reversal is also applied to `conv_kernel_sizes` and `conv_strides`.
    conv_kernel_sizes: The height and width of the 2D convolution windows in the
      layers of each block.
    conv_strides: The strides along the height and width of the MaxPoolings in
      the DownSample and of the Conv2DTranspose layers in the UpSample block.
    reservoir_base: The reservoir layer name to use. Default is DenseReservoir.
    num_classes: The number of classes in the labels. Note that labels must be
      in a shape of (height, width, optional_color_channels, num_classes).
    num_sensors_per_channel: The number of sensory neurons that are connected to
      one input channel.
    reservoir_layer_dropout_rate: The fraction of the output units of the
      reservoir layers to drop each training step.
    upsample_layer_dropout_rate: The fraction of the output units of the
      upsample convolution layers to drop each training step
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.
    reservoir_params_set: the parameters to initialize the reservoirs. (Any
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

  Returns:
    The reservoir model as a keras model object.
  """

  if reservoir_params_set is None:
    reservoir_params_set = tuple([{} for _ in reservoir_weights])

  if len(reservoir_params_set) != len(reservoir_weights):
    raise ValueError(
        f'reservoir_weights should have equal number of elements as '
        f'reservoir_params_set, but got {len(reservoir_weights)} vs '
        f'{len(reservoir_params_set)}.')

  for params_set, weight in zip(reservoir_params_set, reservoir_weights):
    params_set['weight'] = weight

  # Tuple parameters for convolution blocks must have the same num of elements
  if (len(conv_filters) != len(conv_kernel_sizes) or
      len(conv_filters) != len(conv_strides)):
    raise ValueError(
        f'`conv_filters`, `conv_kernel_sizes`, and `conv_strides` must have '
        f'the same number of elements, but got '
        f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

  # Check if the weight matrices are correctly chained
  for i in range(len(reservoir_weights) - 1):
    num_columns = reservoir_weights[i].shape[1]
    num_rows = reservoir_weights[i + 1].shape[0]
    if num_columns != num_rows:
      raise ValueError(
          f'Reservoir weight matrix {i} with {num_columns} columns cannot be '
          f'chained to matrix {i+1} with {num_rows} rows')

  # Check if the Fly Sensor matches the first reservoir
  num_sensors = conv_filters[-1] * num_sensors_per_channel
  if num_sensors != reservoir_weights[0].shape[0]:
    raise ValueError(
        f'The number of sensors = the last element of conv_filters times '
        f'num_sensors_per_channel ={conv_filters[-1]}*{num_sensors_per_channel}'
        f' = {num_sensors} does not match with the number  of rows of the first'
        f' reservoir, which is {reservoir_weights[0].shape[0]}')

  # Check if the reservoirs meet the recurrence requirement
  reservoir_recurrence_degrees = [
      params['recurrence_degree'] for params in reservoir_params_set
  ]
  for recurrence_degree, weight in zip(reservoir_recurrence_degrees,
                                       reservoir_weights):
    if recurrence_degree > 0 and weight.shape[0] != weight.shape[1]:
      raise ValueError(
          f'There is a reservoir that does not meet the recurrence requirement.'
          f' It must be square, but the input got shape: {weight.shape}')

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='convolution_segmentation_model')

  # The input layer
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))

  # Convolution head
  model.add(
      blocks.DownSampleConvolution(
          conv_filters=conv_filters,
          conv_kernel_sizes=conv_kernel_sizes,
          max_poolings=(True,) * len(conv_filters),
          conv_strides=conv_strides,
          activation='relu',
          apply_batchnorm=True,
          leaky=True,
          name='convolution_head',
      ))

  # Fly Sensor layer to send input to reservoirs
  model.add(
      layers.SelectiveSensor(
          num_sensors_per_channel=num_sensors_per_channel,
          name='selective_sensor',
      ))

  # The reservoir layer(s)

  for i, reservoir_params in enumerate(reservoir_params_set):
    reservoir_layer_class = reservoir_registry.get_reservoir(reservoir_base)(
        **reservoir_params)

    model.add(reservoir_layer_class)

    model.add(tf.keras.layers.BatchNormalization())

    # comparing rate to a small float of 1e-12 is safer than comparing to zero
    if reservoir_layer_dropout_rate > _VERY_SMALL_FLOAT:
      model.add(tf.keras.layers.Dropout(rate=reservoir_layer_dropout_rate))

    # Noise regularization after the first reservoir
    if i == 0:
      if noise_layer_stddev > _VERY_SMALL_FLOAT:
        model.add(tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev))

  # Upsampling convolution layers
  model.add(
      blocks.UpSampleConvolution(
          conv_filters=conv_filters[::-1],
          conv_kernel_sizes=conv_kernel_sizes[::-1],
          conv_strides=conv_strides[::-1],
          apply_batchnorm=True,
          dropout_rate=upsample_layer_dropout_rate,
          name='upsample_convolution',
      ))

  final_activation = 'sigmoid' if num_classes == 1 else 'softmax'

  # Last convolution layer for segmentation
  model.add(
      tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=1,
          padding='same',
          activation=final_activation,
          name='final_convolution_layer',
      ))

  return model


def sparse_sensor_model(
    input_shape: Tuple[int, int, int],
    reservoir_weights: Tuple[np.ndarray, ...],
    reservoir_recurrence_degrees: Tuple[int, ...],
    trainable_reservoir: Tuple[bool, ...],
    conv_filters: Tuple[int, ...],
    conv_kernel_sizes: Tuple[int, ...],
    conv_strides: Tuple[int, ...],
    reservoir_base: str = 'SparseReservoir',
    num_classes: int = 1,
    num_sensors: int = 100,
    reservoir_use_bias: bool = False,
    reservoir_layer_dropout_rate: float = 0.0,
    upsample_layer_dropout_rate: float = 0.0,
    noise_layer_stddev: float = 0.0,
) -> tf.keras.Model:
  """Builds a model using a SparseSensor to send input to the reservoirs.

  Model architecture:
  Input => Downsample => SparseSensor => Reservoirs => Upsample => Output

  Note that the Dropout layers are implemented only if the input rate is larger
  than `_VERY_SMALL_FLOAT`. The same for the Noise layer that it is implemented
  only if the input Gaussian stddev is larger than `_VERY_SMALL_FLOAT`.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
      The input layer is built to fit the size of the image.
    reservoir_weights: The weight matrices to be assigned to the reservoir
      layers that are implemented in a series. A recurrent reservoir must be
      square.
    reservoir_recurrence_degrees: The recurrence degrees of the reservoirs. The
      number of elements must be the same as that of reservoir_weights.
    trainable_reservoir: Whether the corresponding reservoir is trainable. If
      True, the reservoir layer is trainable. If False, weights are set
      non-trainable. A trainable DenseReservoir is similar to a Dense layer.
    conv_filters: The number of output filters of the layers in each convolution
      block. The tuple is reversed in the UpSampleConvolution block. Note the
      reversal is also applied to `conv_kernel_sizes` and `conv_strides`.
    conv_kernel_sizes: The height and width of the 2D convolution windows in the
      layers of each block.
    conv_strides: The strides along the height and width of the MaxPoolings in
      the DownSample and of the Conv2DTranspose layers in the UpSample block.
    reservoir_base: The reservoir layer name to use. Default is DenseReservoir.
    num_classes: The number of classes in the labels. Note that labels must be
      in a shape of (height, width, num_classes).
    num_sensors: The number of sensory neurons to connect to the Input.
    reservoir_use_bias: Whether the reservoir layers use a bias vector.
    reservoir_layer_dropout_rate: The fraction of the output units of the
      reservoir layers to drop each training step.
    upsample_layer_dropout_rate: The fraction of the output units of upsample
      convolution layers to drop each training step.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.

  Returns:
    The sparse sensor model.
  """

  # Check if the weight matrices are correctly chained
  for i in range(len(reservoir_weights) - 1):
    num_columns = reservoir_weights[i].shape[1]
    num_rows = reservoir_weights[i + 1].shape[0]
    if num_columns != num_rows:
      raise ValueError(
          f'Reservoir weight matrix {i} with {num_columns} columns cannot be '
          f'chained to matrix {i+1} with {num_rows} rows')

  # Check if the first reservoir can be a SparseSensor
  sensor_weight = reservoir_weights[0]
  sensor_shape = sensor_weight.shape
  if sensor_shape[0] != conv_filters[-1] or sensor_shape[1] != num_sensors:
    raise ValueError(
        f'For the first reservoir_weight to be used to build SparseSensor, '
        f'it is expected to have shape = (conv_filters[-1], num_sensors) = '
        f'({conv_filters[-1]}, {num_sensors}). Got shape = {sensor_shape}.')

  # Check if the reservoirs (except the 1st one) meet the recurrence requirement
  for recurrence_degree, weight in zip(reservoir_recurrence_degrees[1:],
                                       reservoir_weights[1:]):
    if recurrence_degree > 0 and weight.shape[0] != weight.shape[1]:
      raise ValueError(
          f'There is a reservoir that does not meet the recurrence requirement.'
          f' It must be square, but the input got shape: {weight.shape}')

  # Tuple parameters for convolution blocks must have the same num of elements
  if (len(conv_filters) != len(conv_kernel_sizes) or
      len(conv_filters) != len(conv_strides)):
    raise ValueError(
        f'`conv_filters`, `conv_kernel_sizes`, and `conv_strides` must have '
        f'the same number of elements, but got '
        f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='sparsensor_segmentation_model')

  # The input layer
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))

  # Convolution head
  model.add(
      blocks.DownSampleConvolution(
          conv_filters=conv_filters,
          conv_kernel_sizes=conv_kernel_sizes,
          max_poolings=(True,) * len(conv_filters),
          conv_strides=conv_strides,
          activation='relu',
          apply_batchnorm=True,
          leaky=True,
          name='convolution_head',
      ))

  # SparseSensor layer to send input to reservoirs
  model.add(
      layers.SparseSensor(
          num_input_channels=conv_filters[-1],
          num_sensors=num_sensors,
          weight=sensor_weight,
          name='sparse_sensor',
      ))

  # Noise regularization after the sensor
  if noise_layer_stddev > _VERY_SMALL_FLOAT:
    model.add(tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev))

  for i, (reservoir_weight, recurrence_degree, trainable) in enumerate(
      zip(reservoir_weights[1:], reservoir_recurrence_degrees[1:],
          trainable_reservoir[1:])):

    # The reservoir layer(s)
    reservoir_layer_class = reservoir_registry.get_reservoir(reservoir_base)(
        weight=reservoir_weight,
        activation=tf.keras.activations.relu,
        use_bias=reservoir_use_bias,
        recurrence_degree=recurrence_degree,
        trainable_reservoir=trainable,
        name=f'reservoir_layer_{i}',
    )
    model.add(reservoir_layer_class)

    model.add(tf.keras.layers.BatchNormalization())

    # Comparing rate to a small float of 1e-12 is safer than comparing to zero
    if reservoir_layer_dropout_rate > _VERY_SMALL_FLOAT:
      model.add(tf.keras.layers.Dropout(rate=reservoir_layer_dropout_rate))

  # Compute output shape of the reservoirs with `conv_strides` = (2, 2)
  height, width = (input_shape[0], input_shape[1])
  for conv_stride in conv_strides:
    height = (height - 1) // conv_stride + 1
    width = (width - 1) // conv_stride + 1

  # Reshape
  model.add(tf.keras.layers.Reshape((height, width, -1)))

  # Upsampling convolution layers
  model.add(
      blocks.UpSampleConvolution(
          conv_filters=conv_filters[::-1],
          conv_kernel_sizes=conv_kernel_sizes[::-1],
          conv_strides=conv_strides[::-1],
          apply_batchnorm=True,
          dropout_rate=upsample_layer_dropout_rate,
          name='upsample_convolution',
      ))

  final_activation = 'sigmoid' if num_classes == 1 else 'softmax'

  # Last convolution layer for segmentation
  model.add(
      tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=1,
          padding='same',
          activation=final_activation,
          name='final_convolution_layer',
      ))

  return model


@enum.unique
class ConvBlockType(enum.Enum):
  UPSAMPLE = 'UpSample'
  DOWNSAMPLE = 'DownSample'


def convolution_reservoir_alternating_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    add_flies: Tuple[bool, ...],
    reservoir_weights: Tuple[np.ndarray, ...],
    reservoir_recurrence_degrees: Tuple[int, ...],
    conv_block_types: Tuple[str, ...],
    conv_block_filters: Tuple[Tuple[int, ...], ...],
    conv_block_kernel_sizes: Tuple[Tuple[int, ...], ...],
    conv_block_strides: Tuple[Tuple[int, ...], ...],
    reservoir_base: str = 'SparseReservoir',
    reservoir_use_bias: bool = False,
    reservoir_layer_dropout_rate: float = 0.0,
    upsample_layer_dropout_rate: float = 0.0,
    noise_layer_stddev: float = 0.0,
) -> tf.keras.Model:
  """Builds a model of alternating convolution blocks and reservoirs.

  Architecture: convolution => reservoir => convolution => reservoir ...

  Args:
    input_shape: (height, width, num_channels) of the input image.
    num_classes: The number of classes in the labels. Note that labels must be
      in a shape of (height, width, num_classes).
    add_flies: Whether to add reservoirs after the convolution blocks.
    reservoir_weights: The weight matrices to be assigned to the reservoir
      layers. They must be square so that their outputs have the same
      `num_channels` with that of the corresponding inputs.
    reservoir_recurrence_degrees: The recurrence degrees of the reservoirs. The
      number of elements must be the same as that of reservoir_weights.
    conv_block_types: if 'DownSample', a `DownSampleConvolution` block is used.
      Otherwise, a `UpSampleConvolution` block is used.
    conv_block_filters: Tuple of tuples of numbers of filters for the layers of
      the convolution blocks.
    conv_block_kernel_sizes: Tuple of tuples of kernel sizes of the layers of
      the convolution blocks.
    conv_block_strides: Tuple of tuples of strides along the height and width of
      the MaxPoolings in the DownSample and of the Conv2DTranspose layers in the
      UpSample blocks.
    reservoir_base: which reservoir base to use. Default is SparseReservoir.
    reservoir_use_bias: Whether the reservoir layers use a bias vector.
    reservoir_layer_dropout_rate: The fraction of the output units of the
      reservoir layers to drop each training step.
    upsample_layer_dropout_rate: The fraction of the output units of upsample
      convolution layers to drop each training step.
    noise_layer_stddev: Standard deviation of the Gaussian noise layers. If it's
      non-zero, one noise layer is implemented after each convolution block.

  Returns:
    The convolution reservoir alternating model.
  """
  # Downsampling and upsampling must not change input size
  downsample, upsample = 1, 1
  for conv_block_type, conv_strides in zip(conv_block_types,
                                           conv_block_strides):
    # Validate the block type
    conv_block_type = ConvBlockType(conv_block_type)
    if conv_block_type == ConvBlockType.DOWNSAMPLE:
      for conv_stride in conv_strides:
        downsample *= conv_stride
    else:
      for conv_stride in conv_strides:
        upsample *= conv_stride
  if downsample != upsample:
    raise ValueError(
        f'Input size must not change. It gets downsampled {downsample} folds, '
        f'but being upsampled {upsample} folds')

  # Tuple parameters must have the same num of elements
  if (len(reservoir_weights) != len(reservoir_recurrence_degrees) or
      len(reservoir_weights) != len(add_flies) or
      len(reservoir_weights) != len(conv_block_types) or
      len(conv_block_types) != len(conv_block_filters) or
      len(conv_block_types) != len(conv_block_kernel_sizes) or
      len(conv_block_types) != len(conv_block_strides)):
    raise ValueError(
        f'`reservoir_weights`, `reservoir_recurrence_degrees`, `add_flies`, '
        f'`conv_block_type`, `conv_block_filters`, `conv_block_kernel_sizes`, '
        f' and `conv_block_strides` must have the same number of elements, but'
        f'got {len(reservoir_weights)}, {len(reservoir_recurrence_degrees)}, '
        f'{len(add_flies)}, {len(conv_block_types)}, {len(conv_block_filters)},'
        f' {len(conv_block_kernel_sizes)} and {len(conv_block_strides)}.')

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(
      name='alternate_convolution_reservoir_model')

  # The input layer
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))

  for i, (
      conv_block_type,
      conv_filters,
      conv_kernel_sizes,
      conv_strides,
      add_reservoir,
      weight,
      recurrence_degree,
  ) in enumerate(
      zip(
          conv_block_types,
          conv_block_filters,
          conv_block_kernel_sizes,
          conv_block_strides,
          add_flies,
          reservoir_weights,
          reservoir_recurrence_degrees,
      )):
    # Verify the reservoirs are square
    shape = weight.shape
    if add_reservoir and shape[0] != shape[1]:
      raise ValueError(
          f'All used reservoirs must be square, but # {i} got shape of {shape}')

    # Tuples of the convolution block must have the same length
    if (len(conv_filters) != len(conv_kernel_sizes) or
        len(conv_filters) != len(conv_strides)):
      raise ValueError(
          f'Corresponding tuple elements of `conv_block_filters`, '
          f'`conv_block_kernel_sizes` and `conv_block_strides` must have the '
          f'same length, but those at position {i} have lengths of '
          f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

    # Last layer of the convolution block must connect with the reservoir
    if add_reservoir and conv_filters[-1] != shape[0]:
      raise ValueError(
          f'Convolution block must connect with reservoir, but block {i} '
          f'has the last number of conv_filters = {conv_filters[-1]} while '
          f'reservoir {i} has shape = {shape})')

    # Add a convolution block
    conv_block_type = ConvBlockType(conv_block_type)
    if conv_block_type == ConvBlockType.DOWNSAMPLE:
      model.add(
          blocks.DownSampleConvolution(
              conv_filters=conv_filters,
              conv_kernel_sizes=conv_kernel_sizes,
              max_poolings=(True,) * len(conv_filters),
              conv_strides=conv_strides,
              activation=tf.keras.activations.relu,
              apply_batchnorm=True,
              leaky=True,
              name=f'convolution_block_{i}',
          ))
    else:
      model.add(
          blocks.UpSampleConvolution(
              conv_filters=conv_filters,
              conv_kernel_sizes=conv_kernel_sizes,
              conv_strides=conv_strides,
              apply_batchnorm=True,
              dropout_rate=upsample_layer_dropout_rate,
              name=f'convolution_block_{i}',
          ))

    # Noise regularization after each convolution block
    if noise_layer_stddev > _VERY_SMALL_FLOAT:
      model.add(tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev))

    # Add a sparse reservoir
    if add_reservoir:
      model.add(
          reservoir_registry.get_reservoir(reservoir_base)(
              weight=weight,
              activation=tf.keras.activations.relu,
              use_bias=reservoir_use_bias,
              recurrence_degree=recurrence_degree,
              trainable_reservoir=True,
              name=f'reservoir_layer_{i}',
          ))

      # Comparing rate to a small float of 1e-12 is safer than comparing to zero
      if reservoir_layer_dropout_rate > _VERY_SMALL_FLOAT:
        model.add(tf.keras.layers.Dropout(rate=reservoir_layer_dropout_rate))

    model.add(tf.keras.layers.BatchNormalization())

  final_activation = 'sigmoid' if num_classes == 1 else 'softmax'

  # Last convolution layer for segmentation
  model.add(
      tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=1,
          padding='same',
          activation=final_activation,
          name='final_convolution_layer',
      ))

  return model


def convolution_multi_reservoirs(
    input_shape: Tuple[int, int, int],
    reservoir_weights: Tuple[np.ndarray, ...],
    reservoir_recurrence_degrees: Tuple[int, ...],
    trainable_reservoir: Tuple[bool, ...],
    conv_filters: Tuple[int, ...],
    conv_kernel_sizes: Tuple[int, ...],
    conv_strides: Tuple[int, ...],
    upsample_conv_filters: Optional[Tuple[int, ...]] = None,
    upsample_conv_kernel_sizes: Optional[Tuple[int, ...]] = None,
    upsample_conv_strides: Optional[Tuple[int, ...]] = None,
    reservoir_base: str = 'SparseReservoir',
    local_learning: str = 'none',
    num_classes: int = 1,
    reservoir_use_bias: bool = False,
    reservoir_layer_dropout_rate: float = 0.0,
    upsample_layer_dropout_rate: float = 0.0,
    noise_layer_stddev: float = 0.0,
):
  """Builds a model using multiple reservoirs.

  Model: Input => Fly Reservoirs => Convolution => Output

  Note that the Dropout layers are implemented only if the input rate is larger
  than `_VERY_SMALL_FLOAT`. The same for the Noise layer that it is implemented
  only if the input Gaussian stddev is larger than `_VERY_SMALL_FLOAT`.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
      The input layer is built to fit the size of the image.
    reservoir_weights: The weight matrices to be assigned to the reservoir
      layers that are implemented in a series. A recurrent reservoir must be
      square.
    reservoir_recurrence_degrees: The recurrence degrees of the reservoirs. The
      number of elements must be the same as that of reservoir_weights.
    trainable_reservoir: Whether the corresponding reservoir is trainable. If
      True, the reservoir layer is trainable. If False, weights are set
      non-trainable. A trainable FlyDense is similar to a Dense layer.
    conv_filters: The number of output filters of the layers in the DownSample
      convolution block.
    conv_kernel_sizes: The height and width of the 2D convolution windows in the
      layers of the DownSample block.
    conv_strides: The strides along the height and width of the MaxPoolings in
      the DownSample block.
    upsample_conv_filters: The number of output filters of the layers in the
      UpSample convolution block. If not provided, the reversed tuple of
      `conv_filters` is used.
    upsample_conv_kernel_sizes: The height and width of the 2D convolution
      windows in the layers of the UpSample block. If not provided, the reversed
      tuple of `conv_kernel_sizes` is used.
    upsample_conv_strides: The strides along the height and width of the
      Conv2DTranspose layers in the UpSample block. If not provided, the
      reversed tuple of `conv_strides` is used.
    reservoir_base: The reservoir layer name to use. Default is SparseReservoir.
    local_learning: local learning rules to apply to kernels of the reservoir,
      options include ('none', 'hebbian', 'oja', 'contrastive_hebbian')
    num_classes: The number of classes in the labels. Note that labels must be
      in a shape of (height, width, optional_color_channels, num_classes).
    reservoir_use_bias: Whether the reservoir layer(s) use(s) a bias vector.
    reservoir_layer_dropout_rate: The fraction of the output units of the
      reservoir layers to drop each training step.
    upsample_layer_dropout_rate: The fraction of the output units of the
      upsample convolution layers to drop each training step.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.

  Returns:
    The reservoir model as a keras model object.
  """

  # Tuple parameters for reservoirs must have the same number of elements:
  if (len(reservoir_weights) != len(reservoir_recurrence_degrees) or
      len(reservoir_weights) != len(trainable_reservoir)):
    raise ValueError(
        f'`reservoir_weights`, `reservoir_recurrence_degrees`, and '
        f'`trainable_reservoir` must have the same number of elements, but got '
        f'{len(reservoir_weights)}, {len(reservoir_recurrence_degrees)}, and '
        f'{len(trainable_reservoir)}')

  # Tuple parameters for convolution blocks must have the same num of elements
  if (len(conv_filters) != len(conv_kernel_sizes) or
      len(conv_filters) != len(conv_strides)):
    raise ValueError(
        f'`conv_filters`, `conv_kernel_sizes`, and `conv_strides` must have '
        f'the same number of elements, but got '
        f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

  # Tuple parameters for UpSample conv block
  upsample_conv_filters = upsample_conv_filters or conv_filters[::-1]
  upsample_conv_kernel_sizes = (
      upsample_conv_kernel_sizes or conv_kernel_sizes[::-1])

  # Check if UpSampling matches DownSampling
  if upsample_conv_strides:
    if tf.math.reduce_prod(upsample_conv_strides) != tf.math.reduce_prod(
        conv_strides):
      raise ValueError(
          f'Product of UpSample factors of {upsample_conv_strides} does not '
          f'match with that of DownSample factors of {conv_strides}')
  else:
    upsample_conv_strides = conv_strides[::-1]

  # Tuple parameters for UpSample block must have the same num of elements
  if (len(upsample_conv_filters) != len(upsample_conv_kernel_sizes) or
      len(upsample_conv_filters) != len(upsample_conv_strides)):
    raise ValueError(
        f'`upsample_conv_filters`, `upsample_conv_kernel_sizes`, and '
        f'`upsample_conv_strides` must have the same number of elements, but '
        f'got {len(upsample_conv_filters)}, {len(upsample_conv_kernel_sizes)}, '
        f'{len(upsample_conv_strides)}')

  # Check if the weight matrices are correctly chained
  for i in range(len(reservoir_weights) - 1):
    num_columns = reservoir_weights[i].shape[1]
    num_rows = reservoir_weights[i + 1].shape[0]
    if num_columns != num_rows:
      raise ValueError(
          f'Reservoir weight matrix {i} with {num_columns} columns cannot be '
          f'chained to matrix {i+1} with {num_rows} rows')

  # Check if the first reservoir connects to the convolution head
  if conv_filters[-1] != reservoir_weights[0].shape[0]:
    raise ValueError(
        f'The last number of convolution filters must be equal to the number of'
        f' rows of the first reservoir, but got {conv_filters[-1]} and '
        f'{reservoir_weights[0].shape[0]}')

  # Check if the reservoirs meet the recurrence requirement
  for recurrence_degree, weight in zip(reservoir_recurrence_degrees,
                                       reservoir_weights):
    if recurrence_degree > 0 and weight.shape[0] != weight.shape[1]:
      raise ValueError(
          f'There is a reservoir that does not meet the recurrence requirement.'
          f' It must be square, but the input got shape: {weight.shape}')

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='multi_reservoirs_segmentation_model')

  # The input layer
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))

  # The Convolution head to connect to the first reservoir
  model.add(
      blocks.DownSampleConvolution(
          conv_filters=conv_filters,
          conv_kernel_sizes=conv_kernel_sizes,
          max_poolings=(True,) * len(conv_filters),
          conv_strides=conv_strides,
          activation='relu',
          apply_batchnorm=True,
          leaky=True,
          name='convolution_head',
      ))

  for i, (reservoir_weight, recurrence_degree, trainable) in enumerate(
      zip(reservoir_weights, reservoir_recurrence_degrees,
          trainable_reservoir)):
    reservoir_layer_class = reservoir_registry.get_reservoir(reservoir_base)(
        weight=reservoir_weight,
        activation=tf.keras.activations.relu,
        use_bias=reservoir_use_bias,
        recurrence_degree=recurrence_degree,
        trainable_reservoir=trainable,
        kernel_local_learning=local_learning,
        name=f'reservoir_layer_{i}',
    )

    model.add(reservoir_layer_class)

    model.add(tf.keras.layers.BatchNormalization())

    # comparing rate to a small float of 1e-12 is safer than comparing to zero
    if reservoir_layer_dropout_rate > _VERY_SMALL_FLOAT:
      model.add(tf.keras.layers.Dropout(rate=reservoir_layer_dropout_rate))

    # Noise regularization after the first reservoir
    if i == 0:
      if noise_layer_stddev > _VERY_SMALL_FLOAT:
        model.add(tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev))

  # Upsampling convolution layers
  model.add(
      blocks.UpSampleConvolution(
          conv_filters=upsample_conv_filters,
          conv_kernel_sizes=upsample_conv_kernel_sizes,
          conv_strides=upsample_conv_strides,
          apply_batchnorm=True,
          dropout_rate=upsample_layer_dropout_rate,
          name='upsample_convolution',
      ))

  final_activation = 'sigmoid' if num_classes == 1 else 'softmax'

  # Last convolution layer for segmentation
  model.add(
      tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=1,
          padding='same',
          activation=final_activation,
          name='final_convolution_layer',
      ))

  return model


def selective_sensor_for_two_contrastive_labels(
    input_shape: Tuple[int, int, int],
    reservoir_weights: Tuple[np.ndarray, ...],
    reservoir_recurrence_degrees: Tuple[int, ...],
    trainable_reservoir: Tuple[bool, ...],
    conv_filters: Tuple[int, ...],
    conv_kernel_sizes: Tuple[int, ...],
    conv_strides: Tuple[int, ...],
    num_sensors_per_channel: int = 10,
    reservoir_base: str = 'SparseReservoir',
    reservoir_use_bias: bool = False,
    reservoir_layer_dropout_rate: float = 0.0,
    upsample_layer_dropout_rate: float = 0.0,
    noise_layer_stddev: float = 0.0,
) -> tf.keras.Model:
  """Builds a model using a SelectiveSensor to send input to the reservoirs.

  Note that this model is very similar to `selective_sensor_model` in module
  `common/models/segmentation_models.py`, except for the final layer. Here all
  three classes (two positive, contrastive classes and one negative class) are
  in the same channel.

  Model: Input => Convolution Head => SelectiveSensor => Reservoirs => Output
  The output of the Convo Head is reshaped into (batch, indices, channels). The
  dim of the output of SelectiveSensor (conv_filters * num_sensors_per_channel)
  must match with the first reservoir's number of rows.

  Note that the Dropout layers are implemented only if the input rate is larger
  than `_VERY_SMALL_FLOAT`. The same for the Noise layer that it is implemented
  only if the input Gaussian stddev is larger than `_VERY_SMALL_FLOAT`.

  Args:
    input_shape: (image_height, image_width, num_channels) of the input image.
      The input layer is built to fit the size of the image.
    reservoir_weights: The weight matrices to be assigned to the reservoir
      layers that are implemented in a series. A recurrent reservoir must be
      square.
    reservoir_recurrence_degrees: The recurrence degrees of the reservoirs. The
      number of elements must be the same as that of reservoir_weights.
    trainable_reservoir: Whether the corresponding reservoir is trainable. If
      True, the reservoir layer is trainable. If False, weights are set
      non-trainable. A trainable FlyDense is similar to a Dense layer.
    conv_filters: The number of output filters of the layers in each convolution
      block. The tuple is reversed in the UpSampleConvolution block. Note the
      reversal is also applied to `conv_kernel_sizes` and `conv_strides`.
    conv_kernel_sizes: The height and width of the 2D convolution windows in the
      layers of each block.
    conv_strides: The strides along the height and width of the MaxPoolings in
      the DownSample and of the Conv2DTranspose layers in the UpSample block.
    num_sensors_per_channel: The number of sensory neurons that are connected to
      one input channel.
    reservoir_base: The reservoir layer name to use. Default is FlyDense.
    reservoir_use_bias: Whether the reservoir layer(s) use(s) a bias vector.
    reservoir_layer_dropout_rate: The fraction of the output units of the
      reservoir layers to drop each training step.
    upsample_layer_dropout_rate: The fraction of the output units of the
      upsample convolution layers to drop each training step
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.

  Returns:
    The reservoir model as a keras model object.
  """
  # Tuple parameters for convolution blocks must have the same num of elements
  if (len(conv_filters) != len(conv_kernel_sizes) or
      len(conv_filters) != len(conv_strides)):
    raise ValueError(
        f'`conv_filters`, `conv_kernel_sizes`, and `conv_strides` must have '
        f'the same number of elements, but got '
        f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

  # Tuple parameters for reservoirs must have the same number of elements:
  if (len(reservoir_weights) != len(reservoir_recurrence_degrees) or
      len(reservoir_weights) != len(trainable_reservoir)):
    raise ValueError(
        f'`reservoir_weights`, `reservoir_recurrence_degrees`, and '
        f'`trainable_reservoir` must have the same number of elements, but got '
        f'{len(reservoir_weights)}, {len(reservoir_recurrence_degrees)}, and '
        f'{len(trainable_reservoir)}')

  # Check if the weight matrices are correctly chained
  for i in range(len(reservoir_weights) - 1):
    num_columns = reservoir_weights[i].shape[1]
    num_rows = reservoir_weights[i + 1].shape[0]
    if num_columns != num_rows:
      raise ValueError(
          f'Reservoir weight matrix {i} with {num_columns} columns cannot be '
          f'chained to matrix {i+1} with {num_rows} rows')

  # Check if the Fly Sensor matches the first reservoir
  num_sensors = conv_filters[-1] * num_sensors_per_channel
  if num_sensors != reservoir_weights[0].shape[0]:
    raise ValueError(
        f'The number of sensors = the last element of conv_filters times '
        f'num_sensors_per_channel ={conv_filters[-1]}*{num_sensors_per_channel}'
        f' = {num_sensors} does not match with the number  of rows of the first'
        f' reservoir, which is {reservoir_weights[0].shape[0]}')

  # Check if the reservoirs meet the recurrence requirement
  for recurrence_degree, weight in zip(reservoir_recurrence_degrees,
                                       reservoir_weights):
    if recurrence_degree > 0 and weight.shape[0] != weight.shape[1]:
      raise ValueError(
          f'There is a reservoir that does not meet the recurrence requirement.'
          f' It must be square, but the input got shape: {weight.shape}')

  # Create a Sequential keras model
  model = tf.keras.models.Sequential(name='convolution_segmentation_model')

  # The input layer
  model.add(tf.keras.layers.InputLayer(input_shape, name='input_layer'))

  # Convolution head
  model.add(
      blocks.DownSampleConvolution(
          conv_filters=conv_filters,
          conv_kernel_sizes=conv_kernel_sizes,
          max_poolings=(True,) * len(conv_filters),
          conv_strides=conv_strides,
          activation='relu',
          apply_batchnorm=True,
          leaky=True,
          name='convolution_head',
      ))

  # Fly Sensor layer to send input to reservoirs
  model.add(
      layers.SelectiveSensor(
          num_sensors_per_channel=num_sensors_per_channel,
          name='selective_sensor',
      ))

  # The reservoir layer(s)

  for i, (reservoir_weight, recurrence_degree, trainable) in enumerate(
      zip(reservoir_weights, reservoir_recurrence_degrees,
          trainable_reservoir)):
    reservoir_layer_class = reservoir_registry.get_reservoir(reservoir_base)(
        weight=reservoir_weight,
        activation=tf.keras.activations.relu,
        use_bias=reservoir_use_bias,
        recurrence_degree=recurrence_degree,
        trainable_reservoir=trainable,
        name=f'reservoir_layer_{i}',
    )

    model.add(reservoir_layer_class)

    model.add(tf.keras.layers.BatchNormalization())

    # comparing rate to a small float of 1e-12 is safer than comparing to zero
    if reservoir_layer_dropout_rate > _VERY_SMALL_FLOAT:
      model.add(tf.keras.layers.Dropout(rate=reservoir_layer_dropout_rate))

    # Noise regularization after the first reservoir
    if i == 0:
      if noise_layer_stddev > _VERY_SMALL_FLOAT:
        model.add(tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev))

  # Upsampling convolution layers
  model.add(
      blocks.UpSampleConvolution(
          conv_filters=conv_filters[::-1],
          conv_kernel_sizes=conv_kernel_sizes[::-1],
          conv_strides=conv_strides[::-1],
          apply_batchnorm=True,
          dropout_rate=upsample_layer_dropout_rate,
          name='upsample_convolution',
      ))

  # Final convolution layer for segmentation
  model.add(
      tf.keras.layers.Conv2D(
          filters=1,
          kernel_size=1,
          padding='same',
          activation='tanh',
          name='final_convolution_layer',
      ))

  return model


## Code for Deeplab inspired models begins here
def deeplab_inspired_encoder(input_tensor: tf.Tensor) -> tf.keras.Model:
  """Creates an encoder backbone for DeepFly inspired by encoder section of UFly.

  Args:
    input_tensor: Keras tensor to use as input image for the model

  Returns:
    Keras model object for the full encoder backbone.
  """

  x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(input_tensor)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  previous_block_activation = x  # Set aside residual
  for filters in [64, 128, 256]:
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(
        3, strides=2, padding='same', name=f'max_pool_{filters}')(
            x)
    residual = tf.keras.layers.Conv2D(
        filters, 1, strides=2, padding='same')(
            previous_block_activation)
    x = tf.keras.layers.add([x, residual], name=f'add_{filters}')
    x = tf.keras.layers.Conv2D(
        filters, 1, strides=1,
        padding='same')(x)  # Allows us to break the graph
    previous_block_activation = x  # Set aside the next residual

  return tf.keras.Model(input_tensor, x)


def downsample_standard(input_tensor) -> tf.keras.Model:
  """Creates a basic encoder backbone for DeepFly that uses 2 down-sampling blocks.

  Note that this is actually bigger than the deeplab inspired encoder as it does
  not use separable convolutions and has more filters.

  Args:
    input_tensor: Keras tensor to use as input image for the model

  Returns:
    Keras model object for the full encoder backbone.
  """

  num_convolution_layers = 2  # conv layers per block

  first_conv_head = blocks.DownSampleConvolution(
      conv_filters=(64, 128),
      conv_kernel_sizes=(3, 3),
      max_poolings=(True,) * num_convolution_layers,
      conv_strides=(2, 2),
      activation='relu',
      apply_batchnorm=True,
      leaky=True,
      name='convolution_head_1')(
          input_tensor)

  outputs = blocks.DownSampleConvolution(
      conv_filters=(256, 512),
      conv_kernel_sizes=(3, 3),
      max_poolings=(True,) * num_convolution_layers,
      conv_strides=(2, 2),
      activation='relu',
      apply_batchnorm=True,
      leaky=False,
      name='convolution_head_2')(
          first_conv_head)

  return tf.keras.Model(input_tensor, outputs)


def _split_model(model: tf.keras.Model,
                 layer_name: str) -> Tuple[tf.keras.Model, tf.keras.Model]:
  """Splits a Keras model into two at a given layer.

  NOTE: If you have a branching, non-sequential model (such as UNet, ResNet etc)
  it is EXTREMELY important that you choose to split the model at a merge point,
  at least one layer before the graph splits again.

  Args:
    model: The original keras model to split.
    layer_name: The layer where we want to split the model. This layer will be
      included in the first of the two returned models.

  Returns:
    Tuple with Keras model objects where the layer specified by layer_name will
    be the lasy layer of the first model, and the output shape of that layer
    will be the input from the second mode.
  """

  # Get the layer index for requested layer to split at from layer name and get
  # all information about the input nodes for each layer.
  layer_inputs = {}
  for idx, layer in enumerate(model.layers):
    if layer.name == layer_name:
      index = idx + 1
    if 'add' in layer.name:
      layer_inputs[str(idx)] = {}
      layer_inputs[str(idx)]['merge'] = True
      layer_inputs[str(idx)]['input0_name'] = layer.input[0].name.split('/')[0]
      layer_inputs[str(idx)]['input1_name'] = layer.input[1].name.split('/')[0]
    else:
      layer_inputs[str(idx)] = {}
      layer_inputs[str(idx)]['merge'] = False
      layer_inputs[str(idx)]['input0_name'] = layer.input.name.split('/')[0]

  outputs = {}

  # Create the first model
  layer_input_1 = tf.keras.Input(model.input_shape[1:])
  x = layer_input_1  # Initialize
  outputs[layer_input_1.name] = x
  layer_inputs[str(1)]['input0_name'] = layer_input_1.name  # Replace input name
  for idx, layer in enumerate(model.layers[1:index]):
    if not layer_inputs[str(idx + 1)]['merge']:
      x = layer(outputs[layer_inputs[str(idx + 1)]['input0_name']])
    else:  # Merge layers require two inputs
      input_0 = outputs[layer_inputs[str(idx + 1)]['input0_name']]
      input_1 = outputs[layer_inputs[str(idx + 1)]['input1_name']]
      x = layer([input_0, input_1])
    # Set aside outputs for later use
    outputs[layer.name] = x

  model1 = tf.keras.models.Model(inputs=layer_input_1, outputs=x)

  # Create the second model
  input_shape_2 = model.layers[index].get_input_shape_at(0)[1:]
  layer_input_2 = tf.keras.Input(shape=input_shape_2)
  x = layer_input_2  # Initialize
  outputs[layer_input_2.name] = x
  layer_inputs[str(index)]['input0_name'] = layer_input_2.name
  for idx, layer in enumerate(model.layers[index:]):
    if not layer_inputs[str(idx + index)]['merge']:
      x = layer(outputs[layer_inputs[str(idx + index)]['input0_name']])
    else:  # Merge layers require two inputs
      input_0 = outputs[layer_inputs[str(idx + index)]['input0_name']]
      input_1 = outputs[layer_inputs[str(idx + index)]['input1_name']]
      x = layer([input_0, input_1])
    # Set aside outputs for later use
    outputs[layer.name] = x

  model2 = tf.keras.models.Model(inputs=layer_input_2, outputs=x)

  return (model1, model2)


def get_backbone_models(
    backbone: str,
    input_shape: Tuple[int, int, int],
    pretrained: bool = False) -> Tuple[tf.keras.Model, tf.keras.Model]:
  """Fetches models to use as the encoder backbones for Deeplab inspired models.

  Args:
    backbone: One of 'resnet50', 'mobilenetv2', 'deeplab_inspired' or
      'downsample_standard'.
    input_shape: Shape in (image_height, image_width, num_channels) of the input
      image to use to initialize the input layer.
    pretrained: Whether or not to include the pretrained weights from training
      on ImageNet. Default is False, which returns random initialization of
      weights. This is only for the Keras out-of-the-box model options.

  Returns:
    Keras model objects for the two parts of the encoder backbone.
  """

  if backbone not in [
      'resnet50', 'mobilenetv2', 'deeplab_inspired', 'downsample_standard'
  ]:
    raise ValueError(
        f'Only `resnet50` and `mobilenetv2`, `deeplab_inspired` and `downsample_standard` have been implemented as backbones so far. Got {backbone}'
    )
  if backbone not in ['resnet50', 'mobilenetv2'] and pretrained:
    raise ValueError(
        'You have requested pretraining for a custom backbone option, which is not implemented'
    )

  backbone_options = {
      'resnet50': {
          'model': tf.keras.applications.ResNet50,
          'layer_1': 'conv3_block4_add',
          'layer_2': 'conv4_block2_add',
      },
      'mobilenetv2': {
          'model': tf.keras.applications.MobileNetV2,
          'layer_1': 'block_12_add',
          'layer_2': 'out_relu',
      },
      'deeplab_inspired': {
          'model': deeplab_inspired_encoder,
          'layer_1': 'add_64',
          'layer_2': 'add_256',
      },
      'downsample_standard': {
          'model': downsample_standard,
          'layer_1': 'convolution_head_1',
          'layer_2': 'convolution_head_2',
      }
  }

  input_layer = tf.keras.Input(shape=input_shape)

  if pretrained:
    backbone_model = backbone_options[backbone]['model'](
        input_tensor=input_layer, weights='imagenet', include_top=True)
  else:
    if backbone in ['deeplab_inspired', 'downsample_standard']:
      backbone_model = backbone_options[backbone]['model'](
          input_tensor=input_layer)
    else:
      backbone_model = backbone_options[backbone]['model'](
          input_tensor=input_layer, weights=None, include_top=True)

  # Split the models
  layer_1_models = _split_model(backbone_model,
                                backbone_options[backbone]['layer_1'])
  encoder_part_1 = layer_1_models[0]
  if backbone in ['deeplab_inspired', 'downsample_standard']:
    encoder_part_2 = layer_1_models[1]
  else:  # Remove additional layers for larger models
    layer_2_models = _split_model(layer_1_models[1],
                                  backbone_options[backbone]['layer_2'])
    encoder_part_2 = layer_2_models[0]

  return (encoder_part_1, encoder_part_2)


def get_upsample_layer_fn(input_shape: Tuple[int, int, int],
                          layer_shape: Tuple[int, int, int, int],
                          factor: int) -> tf.keras.layers.Layer:
  """Utility function to get the upsampling layers.

  Args:
    input_shape: Shape in (image_height, image_width, num_channels) of the input
      image.
    layer_shape: Shape in (batch_size, image_height, image_width, num_channels)
      of the previous layer.
    factor: Factor to use for upsampling.

  Returns:
    Keras UpSampling2D layer.
  """

  return tf.keras.layers.UpSampling2D(
      size=(input_shape[0] // factor // layer_shape[1],
            input_shape[1] // factor // layer_shape[2]),
      interpolation='bilinear')


def deeplab_inspired_reservoir_model(
    input_shape: Tuple[int, int, int],
    num_output_channels: int,
    backbone: str,
    pretrained: bool = False,
    reservoir_location: str = 'none',
    reservoir_weight: Optional[np.ndarray] = None,
    reservoir_base: str = 'FlyDense',
    reservoir_params: Optional[Dict[str, Any]] = None,
    reservoir_connection: str = 'dense',
    final_activation: str = 'softmax',
    noise_layer_stddev: float = 0.0,
    aspp_params: Optional[Dict[str, Any]] = None,
) -> tf.keras.Model:
  """Builds a vanilla DeepLabv3+ or a deepFly model.

  Args:
    input_shape: Shape in (image_height, image_width, num_channels) of the input
      images.
    num_output_channels: How many output channels to use.
    backbone: Model to use for the encoder backbone.
    pretrained: Whether or not the backbone layers should be pretrained on
      imagenet.
    reservoir_location: Where to place to reservoir if one is included. Current
      options are 'input_a', 'input_b' or 'both'. 'input_a' places a reservoir
      just after the first half of the encoder backbone ('layer_1') and before
      the next conv layer. 'input_b' places it after the full encoder backbone
      ('layer_2') and before the ASPP layer. 'both' adds two reservoirs, one in
      each location.
    reservoir_weight: The reservoir weights to use in the reservoir part of the
      model
    reservoir_base: The reservoir base to use. Default is 'FlyDense'.
    reservoir_params: The parameters to initialize the reservoir_base. (Any
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
    reservoir_connection: How to connect the reservoir to the rest of the model.
      One of 'dense', 'resize' and 'sparsensor'.
    final_activation: 'sigmoid', 'softmax', or None. Softmax is recommended.
    noise_layer_stddev: Standard deviation of the Gaussian noise layer. If zero,
      the noise layer is not implemented.
    aspp_params: The parameters to use for the ASPP block. If included, all
      arguments for the ASPP class must be used. These are (with defaults) {
      'dilation_rates': (1, 6, 12, 18),
      'dilation_kernel_sizes': (1, 3, 3, 3),
      'conv_num_filters': 256,
      'conv_kernel_size': 1,
      'pooling': 'global',
      'pooling_size': None,
      'padding': 'same',
      'use_separable_conv': True }.

  Returns:
    the model
  """
  if reservoir_params is None:
    reservoir_params = {}
  reservoir_params['weight'] = reservoir_weight
  if aspp_params is None:
    aspp_params = {}

  if backbone not in [
      'resnet50', 'mobilenetv2', 'deeplab_inspired', 'downsample_standard'
  ]:
    raise ValueError(
        f'Only `resnet50` and `mobilenetv2`, `deeplab_inspired` and `downsample_standard` have been implemented as backbones so far. Got {backbone}'
    )

  backbone_models = get_backbone_models(backbone, input_shape, pretrained)

  inputs = tf.keras.Input(shape=input_shape)

  # This sets up half of the encoder backbone which gets passed through 1x1 conv
  first_encoder_section = backbone_models[0](inputs)
  input_a = first_encoder_section  # Set aside for second part of backbone
  pre_reservoir_shape_a = input_a.shape
  if reservoir_location in [
      'input_a', 'both'
  ] and reservoir_weight is not None:  # Add reservoir layer
    if reservoir_connection == 'dense':
      input_a = tf.keras.layers.Dense(
          reservoir_weight.shape[0], activation='elu')(
              input_a)
    elif reservoir_connection in ['resize', 'sparsensor']:
      reservoir_params['weight'] = weight_transforms.resize_weight_matrices(
          [reservoir_weight], (pre_reservoir_shape_a[-1],))[0]
    if reservoir_connection == 'sparsensor':
      input_a = layers.SparseSensor(
          num_input_channels=pre_reservoir_shape_a[-1],
          num_sensors=pre_reservoir_shape_a[-1],
          weight=reservoir_params['weight'],
          name='sparse_sensor_b')(
              input_a)
    input_a = reservoir_registry.get_reservoir(reservoir_base)(
        **reservoir_params)(
            input_a)
    if reservoir_connection == 'sparsensor':
      input_a = tf.keras.layers.Reshape(
          (pre_reservoir_shape_a[-3], pre_reservoir_shape_a[-2], -1))(
              input_a)
    if noise_layer_stddev > np.finfo(float).eps:
      input_a = tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev)(
          input_a)
  input_a = blocks.conv2d_bn(filters=48, num_rows=1, num_cols=1)(input_a)

  # This sets up the full encoder backbone, which continues through the ASPP
  # block before being passed to an upsampling layer
  input_b = backbone_models[1](first_encoder_section)
  pre_reservoir_shape_b = input_a.shape
  if reservoir_location in [
      'input_b', 'both'
  ] and reservoir_weight is not None:  # Add reservoir layer
    if reservoir_connection == 'dense':
      input_b = tf.keras.layers.Dense(
          reservoir_weight.shape[0], activation='elu')(
              input_b)
    elif reservoir_connection in ['resize', 'sparsensor']:
      reservoir_params['weight'] = weight_transforms.resize_weight_matrices(
          [reservoir_weight], (pre_reservoir_shape_b[-1],))[0]
    if reservoir_connection == 'sparsensor':
      input_b = layers.SparseSensor(
          num_input_channels=pre_reservoir_shape_b[-1],
          num_sensors=pre_reservoir_shape_b[-1],
          weight=reservoir_params['weight'],
          name='sparse_sensor_a')(
              input_b)
    input_b = reservoir_registry.get_reservoir(reservoir_base)(
        **reservoir_params)(
            input_b)
    if reservoir_connection == 'sparsensor':
      input_b = tf.keras.layers.Reshape(
          (pre_reservoir_shape_b[-3], pre_reservoir_shape_b[-2], -1))(
              input_b)

    if noise_layer_stddev > np.finfo(float).eps:
      input_b = tf.keras.layers.GaussianNoise(stddev=noise_layer_stddev)(
          input_b)
  input_b = blocks.AtrousSpatialPyramidPooling(**aspp_params)(input_b)

  if backbone in ['downsample_standard', 'deeplab_inspired']:
    input_b = get_upsample_layer_fn(
        input_shape, input_b.shape, factor=4)(
            input_b)
  elif backbone == 'mobilenetv2':
    input_b = get_upsample_layer_fn(
        input_shape, input_b.shape, factor=16)(
            input_b)
  else:
    input_b = get_upsample_layer_fn(
        input_shape, input_b.shape, factor=8)(
            input_b)

  # The two encoder sections are concatenated and passed to the main decoder
  tensor = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])

  # The decoder section
  conv1 = blocks.separable_conv2d_bn(
      filters=256, kernel_size=3, dilation_rate=1)
  conv2 = blocks.separable_conv2d_bn(
      filters=256, kernel_size=3, dilation_rate=1)
  tensor = conv2(conv1(tensor))
  tensor = get_upsample_layer_fn(input_shape, tensor.shape, factor=1)(tensor)
  outputs = tf.keras.layers.Conv2D(
      num_output_channels,
      kernel_size=(1, 1),
      padding='same',
      activation=final_activation)(
          tensor)

  model = tf.keras.Model(inputs, outputs)
  return model
