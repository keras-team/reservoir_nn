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

"""Tests for blocks.py."""

import os
from absl.testing import absltest
from absl.testing import parameterized
from reservoir_nn.keras import blocks
import tensorflow as tf


class DownSampleConvolutionTest(absltest.TestCase):

  def test_it_returns_output_of_expected_shape(self):
    image = tf.range(3 * 20 * 40 * 7, dtype=tf.float32)
    image = tf.reshape(image, (3, 20, 40, 7))
    conv_head = blocks.DownSampleConvolution(
        conv_filters=(2, 3, 4),
        conv_kernel_sizes=(1, 2, 3),
        max_poolings=(False, True, True),
        conv_strides=(1, 2, 2),
    )
    expected_output_shape = (3, 5, 10, 4)
    output = conv_head(image)
    self.assertEqual(output.shape, expected_output_shape)

  def test_it_raises_error_with_unmatched_lengths_of_arguments(self):
    with self.assertRaisesRegex(ValueError, 'The number of elements in'):
      blocks.DownSampleConvolution(
          conv_filters=(2, 3, 4, 5),
          conv_kernel_sizes=(1, 2, 3),
          max_poolings=(False, True, True),
          conv_strides=(1, 2, 2),
      )

  def test_it_raises_error_with_arguments_of_length_zero(self):
    with self.assertRaisesRegex(ValueError, 'Number of convolution layers'):
      blocks.DownSampleConvolution(
          conv_filters=(),
          conv_kernel_sizes=(),
          max_poolings=(),
          conv_strides=(),
      )


class UpSampleConvolutionTest(absltest.TestCase):

  def test_it_returns_output_of_expected_shape(self):
    image = tf.range(3 * 4 * 5 * 6, dtype=tf.float32)
    image = tf.reshape(image, (3, 4, 5, 6))
    conv_block = blocks.UpSampleConvolution(
        conv_filters=(2, 3, 4),
        conv_kernel_sizes=(1, 2, 3),
        conv_strides=(1, 2, 3),
    )
    expected_output_shape = (3, 24, 30, 4)
    output = conv_block(image)
    self.assertEqual(output.shape, expected_output_shape)

  def test_it_raises_error_with_unmatched_lengths_of_arguments(self):
    with self.assertRaisesRegex(ValueError, 'the same number of elements, but'):
      blocks.UpSampleConvolution(
          conv_filters=(2, 3, 4, 5),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(1, 2, 2),
      )

  def test_it_raises_error_with_arguments_of_length_zero(self):
    with self.assertRaisesRegex(ValueError, 'Number of upsample convolution'):
      blocks.UpSampleConvolution(
          conv_filters=(),
          conv_kernel_sizes=(),
          conv_strides=(),
      )


class AtrousSpatialPyramidPoolingTest(absltest.TestCase):

  def test_it_returns_output_of_expected_shape(self):
    image = tf.range(2 * 6 * 6 * 3, dtype=tf.float32)
    image = tf.reshape(image, (2, 6, 6, 3))
    aspp = blocks.AtrousSpatialPyramidPooling(image.shape)
    output = aspp(image)
    expected_output_shape = (2, 6, 6, 256)
    self.assertEqual(output.shape, expected_output_shape)

  def test_it_returns_output_of_expected_shape_when_not_global_pooling(self):
    image = tf.range(2 * 6 * 6 * 3, dtype=tf.float32)
    image = tf.reshape(image, (2, 6, 6, 3))
    aspp = blocks.AtrousSpatialPyramidPooling(
        image.shape, pooling='other', pooling_size=4)
    output = aspp(image)
    expected_output_shape = (2, 6, 6, 256)
    self.assertEqual(output.shape, expected_output_shape)


class Conv2dBnTest(absltest.TestCase):

  def test_it_returns_a_sequence_of_layers_in_order(self):
    conv2d_bn_block = blocks.conv2d_bn(4, 2, 2)
    expected_layer_types = [
        tf.keras.layers.Conv2D,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.Activation,
    ]
    for layer, expected_type in zip(conv2d_bn_block.layers,
                                    expected_layer_types):
      self.assertIsInstance(layer, expected_type)


class MobileNetV2BlockLikeTest(parameterized.TestCase):

  @parameterized.parameters((3, 1, (5, 4, 8, 3)), (4, 2, (5, 2, 4, 4)))
  def test_it_returns_output_of_expected_shape(self, output_filters, stride,
                                               expected_shape):
    x = tf.reshape(tf.range(5 * 4 * 8 * 3, dtype=tf.float32), (5, 4, 8, 3))
    x = blocks.MobileNetV2BlockLike(
        output_filters=output_filters, conv_stride=stride)(
            x)
    self.assertEqual(x.shape, expected_shape)


@parameterized.named_parameters(
    ('InceptionV3Stem', blocks.InceptionV3Stem, {}, [42, 42, 3]),
    ('InceptionBlock', blocks.InceptionBlock, {
        'num_filters': 42
    }, [32, 32, 10]),
    ('DownSampleConvolution', blocks.DownSampleConvolution, {
        'conv_filters': (2, 3),
        'conv_kernel_sizes': (1, 3),
        'max_poolings': (True, False),
        'conv_strides': (2, 2)
    }, [8, 16, 9]),
    ('UpSampleConvolution', blocks.UpSampleConvolution, {
        'conv_filters': (2, 3),
        'conv_kernel_sizes': (1, 3),
        'conv_strides': (2, 2)
    }, [2, 3, 4]),
    ('MobileNetV2BlockLike', blocks.MobileNetV2BlockLike, {
        'output_filters': 8,
    }, [2, 3, 4]),
)
class AllLayersTest(parameterized.TestCase):
  """Standard tests for all layers.

  The parameters are `layer_class, kwargs, input_shape`. Omit the batch
  dimension when specifying input_shape.
  """

  def test_get_config_should_be_the_same_before_and_after_reconstruction(
      self, layer_class, kwargs, _):
    layer = layer_class(**kwargs)
    config = layer.get_config()
    reconstructed = layer_class.from_config(config)
    self.assertEqual(reconstructed.get_config(), config)

  def test_layer_should_accept_valid_inputs_without_errors(
      self, layer_class, kwargs, input_shape):
    layer = layer_class(**kwargs)
    batch_size = 3
    batched_shape = (batch_size, *input_shape)
    x = tf.ones(batched_shape, dtype=tf.float32)
    # This line should not raise an error
    layer(x)

  def test_layer_should_reconstruct_with_load_model(self, layer_class, kwargs,
                                                    input_shape):  # pylint:disable=g-doc-args
    """This test makes sure the layer works with load_model."""
    # Create model
    layer = layer_class(**kwargs)
    model = tf.keras.Sequential([layer])
    batch_size = 3
    batched_shape = (batch_size, *input_shape)
    model.build(batched_shape)
    # Export model
    tmp_dir = self.create_tempdir()
    model_dir = os.path.join(tmp_dir, 'test_model')
    model.save(model_dir)
    # Load model back
    reconstructed_model = tf.keras.models.load_model(model_dir)
    inputs = tf.range(
        tf.reduce_prod(input_shape) * batch_size, dtype=tf.float32)
    inputs = tf.reshape(inputs, batched_shape)
    # Models should produce same output
    tf.debugging.assert_near(reconstructed_model(inputs), model(inputs))
    self.assertEqual(reconstructed_model.get_config(), model.get_config())


if __name__ == '__main__':
  absltest.main()
