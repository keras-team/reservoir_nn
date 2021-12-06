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

"""Keras-like blocks of layers."""

import math
from typing import Any, Optional, Tuple

from reservoir_nn.typing import types
import tensorflow as tf


class DownSampleConvolution(tf.keras.layers.Layer):
  """A downsample convolution head block."""

  def __init__(
      self,
      conv_filters: Tuple[int, ...],
      conv_kernel_sizes: Tuple[int, ...],
      max_poolings: Tuple[bool, ...],
      conv_strides: Tuple[int, ...],
      activation: Optional[types.Activation] = None,
      apply_batchnorm: bool = False,
      leaky: bool = False,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Builds a downsample convolution block.

    After each convolution layer, there are options to add a MaxPooling, a
    BatchNormalization, and a LeakyReLU. If conv_stride > 1 then the inputs are
    down-sampled during MaxPooling. And the output will be resized to:
    ```size = ((size - 1) // stride) + 1``` after each convolution layer.

    Note, the number of elements in each of `conv_filters`, `conv_kernel_sizes`,
    `max_poolings`, and `conv_strides` must be the same.

    Args:
      conv_filters: The numbers of filters in the convolution layers.
      conv_kernel_sizes: The height and width of the 2D convolution window in
        the corresponding layers.
      max_poolings: Whether to add a MaxPooling after the convolution layers.
      conv_strides: The strides along the height and width of the corresponding
        MaxPooling layers .
      activation: The activation function, for example tf.nn.relu.
      apply_batchnorm: Whether to add a BatchNormalization after each layer.
      leaky: Whether to add a LeakyReLU after each convolution layer.
      name: The name of the layer.
      **kwargs: Other keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._conv_filters = conv_filters
    self._conv_kernel_sizes = conv_kernel_sizes
    self._max_poolings = max_poolings
    self._conv_strides = conv_strides
    self._activation = tf.keras.activations.get(activation)
    self._apply_batchnorm = apply_batchnorm
    self._leaky = leaky

    if (len(conv_filters) != len(conv_kernel_sizes) or
        len(conv_filters) != len(max_poolings) or
        len(conv_filters) != len(conv_strides)):
      raise ValueError(
          f'The number of elements in `conv_filters`, `conv_kernel_sizes`, '
          f'`max_poolings`, and `conv_strides` must all be the same, but got '
          f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(max_poolings)},'
          f' and {len(conv_strides)}')

    if len(conv_filters) < 1:
      raise ValueError(
          f'Number of convolution layers (number of elements of `conv_filters`)'
          f' must be > 0, but got {len(conv_filters)}')

    self._conv_block = []
    for i, (filters, kernel_size, max_pooling, conv_stride) in enumerate(
        zip(conv_filters, conv_kernel_sizes, max_poolings, conv_strides)):
      self._conv_block.append(
          tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=kernel_size,
              activation=activation,
              padding='same',
              name=f'downsample_conv_layer_{i}',
          ))

      if max_pooling:
        self._conv_block.append(
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(conv_stride, conv_stride),
                padding='same',
            ))
      if apply_batchnorm:
        self._conv_block.append(tf.keras.layers.BatchNormalization())
      if leaky:
        self._conv_block.append(tf.keras.layers.LeakyReLU())

  def call(self, inputs):
    outputs = inputs
    for layer in self._conv_block:
      outputs = layer(outputs)
    return outputs

  def get_config(self):
    config = super().get_config()
    config.update({
        'conv_filters': self._conv_filters,
        'conv_kernel_sizes': self._conv_kernel_sizes,
        'max_poolings': self._max_poolings,
        'conv_strides': self._conv_strides,
        'activation': self._activation,
        'apply_batchnorm': self._apply_batchnorm,
        'leaky': self._leaky,
    })
    return config


class UpSampleConvolution(tf.keras.layers.Layer):
  """An Upsample convolution block."""

  def __init__(
      self,
      conv_filters: Tuple[int, ...],
      conv_kernel_sizes: Tuple[int, ...],
      conv_strides: Tuple[int, ...],
      apply_batchnorm: bool = False,
      dropout_rate: float = 0.0,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Builds an upsample convolution block.

    After each convolution layer, there are options to add a BatchNormalization,
    a Dropout, and finally a ReLu.

    If conv_stride > 1 then the inputs are up-sampled to be of size:
    ```size = size * stride``` after each convolution layer.

    The number of elements in each of `conv_filters`, `conv_kernel_sizes`, and
    `conv_strides` must be the same, which is the number of convolution layers.

    Args:
      conv_filters: The numbers of filters in the convolution layer.
      conv_kernel_sizes: The height and width of the 2D convolution window in
        the corresponding layers.
      conv_strides: The strides of the convolution window along the height and
        width of the corresponding layers.
      apply_batchnorm: Whether to add a BatchNormalization after each layer.
      dropout_rate: The fraction of the output units of the convolution layers
        to drop each training step.
      name: The name of the layer.
      **kwargs: Other keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._conv_filters = conv_filters
    self._conv_kernel_sizes = conv_kernel_sizes
    self._conv_strides = conv_strides
    self._apply_batchnorm = apply_batchnorm
    self._dropout_rate = dropout_rate

    if (len(conv_filters) != len(conv_kernel_sizes) or
        len(conv_filters) != len(conv_strides)):
      raise ValueError(
          f'`conv_filters`, `conv_kernel_sizes`, and `conv_strides` must have '
          f'the same number of elements, but got '
          f'{len(conv_filters)}, {len(conv_kernel_sizes)}, {len(conv_strides)}')

    if len(conv_filters) < 1:
      raise ValueError(
          f'Number of upsample convolution layers (number of elements of '
          f'`conv_filters`) must be > 0, but got {len(conv_filters)}')

    self._upsample_conv = []
    for i, (filters, kernel_size, conv_stride) in enumerate(
        zip(conv_filters, conv_kernel_sizes, conv_strides)):

      self._upsample_conv.append(
          tf.keras.layers.Conv2DTranspose(
              filters=filters,
              kernel_size=kernel_size,
              strides=(conv_stride, conv_stride),
              padding='same',
              name=f'upsample_conv_layer_{i}',
          ))

      if apply_batchnorm:
        self._upsample_conv.append(tf.keras.layers.BatchNormalization())

      self._upsample_conv.append(tf.keras.layers.Dropout(rate=dropout_rate))

      self._upsample_conv.append(tf.keras.layers.ReLU())

  def call(self, inputs):
    outputs = inputs
    for layer in self._upsample_conv:
      outputs = layer(outputs)
    return outputs

  def get_config(self):
    config = super().get_config()
    config.update({
        'conv_filters': self._conv_filters,
        'conv_kernel_sizes': self._conv_kernel_sizes,
        'conv_strides': self._conv_strides,
        'apply_batchnorm': self._apply_batchnorm,
        'dropout_rate': self._dropout_rate,
    })
    return config


class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
  """Atrous Spatial Pyramid Pooling block for the DeepLabv3+ model class."""

  def __init__(
      self,
      dilation_rates: Tuple[int, int, int, int] = (1, 6, 12, 18),
      dilation_kernel_sizes: Tuple[int, int, int, int] = (1, 3, 3, 3),
      conv_num_filters: int = 256,
      conv_kernel_size: int = 1,
      pooling: str = 'global',
      pooling_size: Optional[int] = None,
      padding: str = 'same',
      use_separable_conv=True,
      name: Optional[str] = 'ASPP',
      **kwargs: Any,
  ):
    """Creates an Atrous Spatial Pyramid pooling block.

    This combines a set of parallel atrous convolutions with different rates.
    Concept and default parameters from Chen, L. C., Zhu, Y., Papandreou, G.,
    Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable
    convolution for semantic image segmentation. In Proceedings of the European
    conference on computer vision (ECCV) (pp. 801-818).

    Args:
      dilation_rates: Set of dilation rates to use in the pyramid. By default,
        these are 1, 6, 12, 18.
      dilation_kernel_sizes: Kernel sizes to use for the atrous convolutions.
      conv_num_filters: Number of filters to use for the two conv operations
        outside the pyramid.
      conv_kernel_size: Kernel size to use for the two conv operations outside
        the pyramid.
      pooling: Whether to use global average pooling ('global') or not.
      pooling_size: Size over which to pool if not using global pooling.
      padding: Padding mode for convolutions
      use_separable_conv: Whether to use separable convolutions for the pyramid.
      name: Name of the block.
      **kwargs: Other keyword arguments.
    """

    super().__init__(name=name, **kwargs)
    self._dilation_rates = dilation_rates
    self._dilation_kernel_sizes = dilation_kernel_sizes
    self._conv_num_filters = conv_num_filters
    self._conv_kernel_size = conv_kernel_size
    self._pooling = pooling
    self._pooling_size = pooling_size
    self._padding = padding
    self._use_separable_conv = use_separable_conv

  def build(self, input_shape):

    if len(input_shape) != 4:
      raise ValueError(
          f'Input data must have four dimensions: batch size (or num. examples), '
          f'height, width and channel but got {len(input_shape)}')

    if self._pooling == 'global':
      pool_size = (input_shape[-3], input_shape[-2])
    else:
      pool_size = (self._pooling_size, self._pooling_size)

    self.avg_pool = tf.keras.layers.AveragePooling2D(
        pool_size=pool_size, strides=(1, 1))
    avg_pool_output_shape = ((math.floor(input_shape[-3] - pool_size[0]) + 1),
                             (math.floor(input_shape[-2] - pool_size[1]) + 1))

    self.conv1 = conv2d_bn(
        filters=self._conv_num_filters,
        num_rows=self._conv_kernel_size,
        num_cols=self._conv_kernel_size,
        dilation_rate=1,
        padding=self._padding,
        strides=(1, 1))
    if self._padding == 'same':  # output is same size as input
      conv1_output_shape = avg_pool_output_shape
    else:
      conv1_output_shape = ((avg_pool_output_shape[0] - self._conv_num_filters +
                             1), (avg_pool_output_shape[1] -
                                  self._conv_num_filters + 1))

    # need to upsample back up to the input shape
    self.out_pool = tf.keras.layers.UpSampling2D(
        size=(input_shape[-3] // conv1_output_shape[0],
              input_shape[-2] // conv1_output_shape[1]),
        interpolation='bilinear')

    self.asp = []
    for kernel_size, rate in zip(self._dilation_kernel_sizes,
                                 self._dilation_rates):
      if self._use_separable_conv:
        self.asp.append(
            separable_conv2d_bn(
                filters=self._conv_num_filters,
                kernel_size=kernel_size,
                dilation_rate=rate,
                padding=self._padding,
                name=self._name))
      else:
        self.asp.append(
            conv2d_bn(
                filters=self._conv_num_filters,
                num_rows=kernel_size,
                num_cols=kernel_size,
                dilation_rate=rate,
                padding=self._padding,
                name=self._name))

    self.conv2 = conv2d_bn(
        filters=self._conv_num_filters,
        num_rows=self._conv_kernel_size,
        num_cols=self._conv_kernel_size,
        dilation_rate=1,
        padding=self._padding)

  def call(self, inputs):
    x = self.avg_pool(inputs)
    x = self.conv1(x)
    pyramid = self.out_pool(x)
    for atrous_layer in self.asp:
      pyramid = tf.keras.layers.Concatenate(axis=-1)(
          [pyramid, atrous_layer(inputs)])
    total = self.conv2(pyramid)
    return total

  def get_config(self):
    config = super().get_config()
    config.update({
        'dilation_rates': self._dilation_rates,
        'dilation_kernel_sizes': self._dilation_kernel_sizes,
        'conv_num_filters': self._conv_num_filters,
        'conv_kernel_size': self._conv_kernel_size,
        'pooling': self._pooling,
        'pooling_size': self._pooling_size,
        'padding': self._padding,
    })
    return config


def conv2d_bn(filters,
              num_rows,
              num_cols,
              dilation_rate=(1, 1),
              padding='same',
              strides=(1, 1),
              name=None) -> tf.keras.Sequential:
  """Applies 2D convolution -> batch normalization -> activation.

  Adopted from /third_party/tensorflow/python/keras/applications/inception_v3.py

  Args:
    filters: filters in `Conv2D`.
    num_rows: height of the convolution kernel.
    num_cols: width of the convolution kernel.
    dilation_rate: dilation for atrous convolution (1 = normal conv)
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    name: name of the ops; will become `name + '_conv'` for the convolution and
      `name + '_bn'` for the batch norm layer.

  Returns:
    Keras layers.
  """
  if name is not None:
    bn_name = name + '_bn'
    conv_name = name + '_conv'
  else:
    bn_name = None
    conv_name = None

  return tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          filters, (num_rows, num_cols),
          dilation_rate=dilation_rate,
          strides=strides,
          padding=padding,
          use_bias=False,
          name=conv_name),
      # Note that the batch norm does not scale the input according to the
      #   reference implementation
      tf.keras.layers.BatchNormalization(scale=False, name=bn_name),
      tf.keras.layers.Activation('relu', name=name),
  ])


def separable_conv2d_bn(filters,
                        kernel_size,
                        dilation_rate,
                        padding='same',
                        name=None) -> tf.keras.Sequential:
  """Separable convolution with batch norm between depthwise and pointwise convs.

  Different from tf.keras.layers.SeparableConv2D because it applies and
  activation function in between the depthwise and pointwise conv as in
  DeeplabV3+

  Args:
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      filters. Can be an int if both values are the same.
    dilation_rate: Atrous convolution rate for the depthwise convolution.
    padding: padding mode for convolutions
    name: name of the ops; will become `name + '_depthwise` for the depthwise
      convolution and `name + '_pointwise'` for the pointwise convolution.

  Returns:
    Keras layers.
  """

  if name is not None:
    depth_name = name + '_depthwise'
    point_name = name + '_pointwise'
  else:
    depth_name = None
    point_name = None

  return tf.keras.Sequential([
      tf.keras.layers.DepthwiseConv2D(
          kernel_size=kernel_size,
          strides=1,
          dilation_rate=dilation_rate,
          padding=padding,
          use_bias=False,
          name=depth_name),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(tf.nn.relu),
      tf.keras.layers.Conv2D(
          filters, (1, 1), padding=padding, use_bias=False, name=point_name),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation(tf.nn.relu),
  ])


class InceptionV3Stem(tf.keras.layers.Layer):
  """Inception V3 stem."""

  def __init__(
      self,
      num_filters: Tuple[int, int, int, int, int] = (32, 32, 64, 80, 192),
      padding: str = 'same',
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Creates the inception stem.

    Adopted from
    /third_party/tensorflow/python/keras/applications/inception_v3.py
    (reference implementation)

    Args:
      num_filters: Numbers of filters used in the layers of the stem. By
        default, the filter counts are 32 -> 32 -> 64 -> 80 -> 192.
      padding: Padding in 2D convolutions. The reference implementation uses
        'valid'.
      name: Name of the block.
      **kwargs: Other keyword arguments.

    Returns:
      The inception stem.
    """
    super().__init__(name=name, **kwargs)
    self._num_filters = num_filters
    self._padding = padding

    self.conv_a0 = conv2d_bn(
        num_filters[0], 3, 3, strides=(2, 2), padding=padding)
    self.conv_a1 = conv2d_bn(num_filters[1], 3, 3, padding=padding)
    self.conv_a2 = conv2d_bn(num_filters[2], 3, 3)
    self.pool_a = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

    self.conv_b0 = conv2d_bn(num_filters[3], 1, 1, padding=padding)
    self.conv_b1 = conv2d_bn(num_filters[4], 3, 3, padding=padding)
    self.pool_b = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

  def call(self, inputs):
    x = self.conv_a0(inputs)
    x = self.conv_a1(x)
    x = self.conv_a2(x)
    x = self.pool_a(x)
    x = self.conv_b0(x)
    x = self.conv_b1(x)
    x = self.pool_b(x)

    return x

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_filters': self._num_filters,
        'padding': self._padding,
    })
    return config


class InceptionBlock(tf.keras.layers.Layer):
  """Inception block."""

  def __init__(self,
               num_filters: int,
               name: Optional[str] = None,
               **kwargs: Any):
    """Creates an inception block."""

    super().__init__(name=name, **kwargs)
    self._num_filters = num_filters
    self.path_a = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            num_filters, [1, 1],
            strides=(1, 1),
            activation='elu',
            padding='same'),
        tf.keras.layers.Conv2D(
            num_filters, [5, 5],
            strides=(1, 1),
            activation='elu',
            padding='same')
    ])

    self.path_b = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            num_filters, [1, 1],
            strides=(1, 1),
            activation='elu',
            padding='same'),
        tf.keras.layers.Conv2D(
            num_filters, [3, 3],
            strides=(1, 1),
            activation='elu',
            padding='same')
    ])

    self.path_c = tf.keras.Sequential([
        tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(1, 1), padding='same'),
        tf.keras.layers.Conv2D(
            num_filters, [1, 1],
            strides=(1, 1),
            activation='elu',
            padding='same')
    ])

    self.path_d = tf.keras.layers.Conv2D(
        num_filters, [1, 1], strides=(1, 1), activation='elu', padding='same')

  def call(self, inputs):
    output_a = self.path_a(inputs)
    output_b = self.path_b(inputs)
    output_c = self.path_c(inputs)
    output_d = self.path_d(inputs)
    return tf.concat([output_a, output_b, output_c, output_d], -1)

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_filters': self._num_filters,
    })
    return config


class MobileNetV2BlockLike(tf.keras.layers.Layer):
  """The MobileNet V2 block."""

  def __init__(
      self,
      output_filters: int,
      expansion_factor: int = 1,
      conv_stride: int = 1,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Sets up the block.

    Args:
      output_filters: The number of output feature filters (channels).
      expansion_factor: The expansion factor of the depthwise convolution layer.
      conv_stride: The stride of the convolution window in the depthwise layer.
      name: The name of the block.
      **kwargs: Other keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self._output_filters = output_filters
    self._expansion_factor = expansion_factor
    self._conv_stride = conv_stride

  def build(self, input_shape):

    self._block = []
    self._block.append(
        tf.keras.layers.Conv2D(
            filters=input_shape[-1] * self._expansion_factor,
            kernel_size=1,
            activation=tf.nn.relu6,
            padding='same',
            name='expand_conv',
        ))

    self._block.append(
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.relu6,
            padding='same',
            name='depthwise_conv',
        ))
    self._block.append(
        tf.keras.layers.MaxPooling2D(
            pool_size=(self._conv_stride, self._conv_stride),
            strides=(self._conv_stride, self._conv_stride),
            padding='same',
        ))
    self._block.append(
        tf.keras.layers.Conv2D(
            filters=self._output_filters,
            kernel_size=1,
            activation=None,
            padding='same',
            name='contract_conv',
        ))
    self._block.append(tf.keras.layers.BatchNormalization())

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    output = input_
    for layer in self._block:
      output = layer(output)
    return output

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_filters': self._output_filters,
        'expansion_factor': self._expansion_factor,
        'conv_stride': self._conv_stride,
    })
    return config


# TODO(b/207715483): Clean up redundancy.
def downsample_block(
    filters: int,
    size: int,
    strides: int = 2,
    apply_batchnorm: bool = True,
    name: Optional[str] = None,
) -> tf.keras.Sequential:
  """Creates a convolutional block.

  Conv2D => Batchnorm => LeakyRelu
  Adapted from tensorflow github:
  /tensorflow_examples/models/pix2pix/pix2pix.py

  When size = 4 and strides = 2, it downsamples the input by 1/2.

  For example, when the input is (512, 512, channel), we can create a stack,
  where the output is 16 by 16 images that have 8 channels.

  down_stack = [
      downsample_block(32, 4, 2, apply_batchnorm=False),  # (bs, 256, 256, 32)
      downsample_block(32, 4, 2),  # (bs, 128, 128, 32)
      downsample_block(32, 4, 2),  # (bs, 64, 64, 32)
      downsample_block(16, 4, 2),  # (bs, 32, 32, 16)
      downsample_block( 8, 4, 2),  # (bs, 16, 16, 8)
  ]

  Args:
    filters: Number of filters.
    size: Filter size.
    strides: Striding for the conv layer.
    apply_batchnorm: If True, adds the batchnorm layer.
    name: Name of this block.

  Returns:
    A Conv2D block.
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  block = tf.keras.Sequential(name=name)
  block.add(
      tf.keras.layers.Conv2D(
          filters,
          size,
          strides=strides,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False,
      ))

  if apply_batchnorm:
    block.add(tf.keras.layers.BatchNormalization())

  block.add(tf.keras.layers.LeakyReLU())

  return block


# TODO(b/207715483): Clean up redundancy.
def upsample_block(
    filters: int,
    size: int,
    strides: int = 2,
    apply_batchnorm: bool = True,
    apply_dropout: bool = False,
    name: Optional[str] = None,
) -> tf.keras.Sequential:
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu
  Adapted from tensorflow github:
  /tensorflow_examples/models/pix2pix/pix2pix.py

  Args:
    filters: Number of filters.
    size: Filter size.
    strides: Striding for the Conv2DTranspose layer.
    apply_batchnorm: If True, adds the batchnorm layer.
    apply_dropout: If True, adds the dropout layer.
    name: Name of this block.

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential(name=name)
  result.add(
      tf.keras.layers.Conv2DTranspose(
          filters,
          size,
          strides=strides,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False,
      ))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
