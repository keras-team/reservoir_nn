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

"""Tests for segmentation_models.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import reservoir_nn.keras.layers as valkyrie_layers
from reservoir_nn.models import segmentation_models
import tensorflow as tf

TrainableParameters = parameterized.named_parameters(
    ('trainable_sparse_reservoir', True),
    ('untrainable_dense_reservoir', False),
)

LayerParameters = parameterized.named_parameters(
    ('dense_reservoir', 'DenseReservoir'),
    ('sparse_reservoir', 'SparseReservoir'),
    ('dense_rnn_reservoir', 'RecurrentDenseReservoir'),
    ('sparse_rnn_reservoir', 'RecurrentSparseReservoir'),
)


class BuildDenselyConnectedReservoirModelTest(parameterized.TestCase):
  """Tests `segmentation_models.minimal_reservoir_model`."""

  @LayerParameters
  def test_all_model_should_produce_prediction_with_correct_shape(self, layer):
    reservoir = np.ones((42, 42))
    batch = 6
    image_size = (2, 3, 1)
    image_set = tf.range(batch * image_size[0] * image_size[1])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        reservoir_base=layer,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_size))

  @LayerParameters
  def test_all_deterministic_initialization_should_produce_same_model(
      self, layer):
    reservoir = np.ones((42, 42))
    image_size = (2, 3, 1)
    model_a = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=True,
        reservoir_base=layer,
    )
    model_b = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=True,
        reservoir_base=layer,
    )

    image = np.ones(shape=(1, 2, 3, 1))

    loss_a = model_a.predict(image)
    loss_b = model_b.predict(image)

    np.testing.assert_allclose(loss_a, loss_b)

  @LayerParameters
  def test_all_random_initialization_should_produce_different_models(
      self, layer):
    reservoir = np.ones((42, 42))
    image_size = (2, 3, 1)
    model_a = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=False,
        reservoir_base=layer,
    )
    model_b = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=False,
        reservoir_base=layer,
    )

    image = np.ones(shape=(1, 2, 3, 1))

    loss_a = model_a.predict(image)
    loss_b = model_b.predict(image)

    # see https://github.com/numpy/numpy/pull/18470
    self.assertFalse(np.allclose(loss_a, loss_b))

  @TrainableParameters
  def test_model_should_produce_prediction_with_correct_shape(self, trainable):
    reservoir = np.ones((42, 42))
    batch = 6
    image_size = (2, 3, 1)
    image_set = tf.range(batch * image_size[0] * image_size[1])
    image_set = tf.reshape(image_set, (batch, *image_size))
    trainable_layer = 'SparseReservoir' if trainable else 'DenseReservoir'
    model = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        trainable_reservoir=trainable,
        reservoir_base=trainable_layer,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_size))

  @TrainableParameters
  def test_deterministic_initialization_should_produce_same_model(
      self, trainable):
    reservoir = np.ones((42, 42))
    image_size = (2, 3, 1)
    trainable_layer = 'SparseReservoir' if trainable else 'DenseReservoir'
    model_a = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=True,
        trainable_reservoir=trainable,
        reservoir_base=trainable_layer,
    )
    model_b = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=True,
        trainable_reservoir=trainable,
        reservoir_base=trainable_layer,
    )

    image = np.ones(shape=(1, 2, 3, 1))

    loss_a = model_a.predict(image)
    loss_b = model_b.predict(image)

    np.testing.assert_allclose(loss_a, loss_b)

  @TrainableParameters
  def test_random_initialization_should_produce_different_models(
      self, trainable):
    reservoir = np.ones((42, 42))
    image_size = (2, 3, 1)
    trainable_layer = 'SparseReservoir' if trainable else 'DenseReservoir'
    model_a = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=False,
        trainable_reservoir=trainable,
        reservoir_base=trainable_layer,
    )
    model_b = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        deterministic_initialization=False,
        trainable_reservoir=trainable,
        reservoir_base=trainable_layer,
    )

    image = np.ones(shape=(1, 2, 3, 1))

    loss_a = model_a.predict(image)
    loss_b = model_b.predict(image)

    # see https://github.com/numpy/numpy/pull/18470
    self.assertFalse(np.allclose(loss_a, loss_b))

  def test_model_should_contain_the_right_layers(self):
    model = segmentation_models.minimal_reservoir_model(
        input_shape=(1, 1, 1),
        reservoir_weight=np.array([[1]]),
        noise_layer_stddev=1.0,
        apply_batchnorm=True,
    )
    expected_layers = (
        tf.keras.layers.Flatten,
        tf.keras.layers.Dense,
        tf.keras.layers.BatchNormalization,
        valkyrie_layers.DenseReservoir,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.GaussianNoise,
        tf.keras.layers.Dense,
        tf.keras.layers.Reshape,
    )
    for layer, expected in zip(model.layers, expected_layers):
      self.assertIsInstance(layer, expected)

  def test_recurrent_model_should_contain_the_right_layers(self):
    model = segmentation_models.build_simple_recurrent_model(
        input_shape=(1, 1),
        reservoir_weight=np.array([[1]]),
        noise_layer_stddev=1.0,
        apply_batchnorm=True,
    )
    expected_layers = (
        tf.keras.layers.Flatten,
        tf.keras.layers.Dense,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.GaussianNoise,
        valkyrie_layers.DenseReservoir,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.Dense,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.GaussianNoise,
        valkyrie_layers.DenseReservoir,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.Dense,
        tf.keras.layers.GaussianNoise,
        tf.keras.layers.Reshape,
    )
    for layer, expected in zip(model.layers, expected_layers):
      self.assertIsInstance(layer, expected)

  def test_model_with_recurrent_reservoir_returns_expected_output_shape(self):
    reservoir = np.ones((42, 42))
    batch = 6
    image_size = (2, 3, 1)
    image_set = tf.range(batch * image_size[0] * image_size[1])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = segmentation_models.minimal_reservoir_model(
        image_size,
        reservoir,
        reservoir_activation='relu',
        recurrence_degree=3,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_size))


class ConvolutionModelTest(absltest.TestCase):

  def test_it_raises_error_with_unmatched_lengths_of_downsample_arguments(self):
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'and `conv_strides` must have'):
      segmentation_models.convolution_model(
          input_shape=input_shape,
          conv_filters=(5, 20),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(1, 2, 3, 4),
      )

  def test_it_raises_error_with_unmatched_lengths_of_upsample_arguments(self):
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, '`upsample_conv_filters`'):
      segmentation_models.convolution_model(
          input_shape=input_shape,
          conv_filters=(2, 3, 4),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(1, 2, 1),
          upsample_conv_filters=(3, 4, 5, 6),
          upsample_conv_kernel_sizes=(4, 5, 6),
          upsample_conv_strides=(1, 2),
      )

  def test_it_raises_error_with_unmatched_downsample_upsample_strides(self):
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'Product of UpSample factors'):
      segmentation_models.convolution_model(
          input_shape=input_shape,
          conv_filters=(5, 20),
          conv_kernel_sizes=(2, 3),
          conv_strides=(1, 2),
          upsample_conv_strides=(3, 4),
      )

  def test_it_should_produce_prediction_with_correct_shape(self):
    batch = 2
    image_shape = (24, 24, 3)
    image_set = tf.range(batch * tf.reduce_prod(image_shape), dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))
    num_classes = 10

    model = segmentation_models.convolution_model(
        input_shape=image_shape,
        conv_filters=(5, 10, 20),
        conv_kernel_sizes=(1, 2, 3),
        conv_strides=(2, 1, 2),
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))


class BuildConvolutionReservoirModelTest(absltest.TestCase):
  """Tests `segmentation_models.convolution_reservoir_model`."""

  def test_downsample_model_should_produce_prediction_with_correct_shape(self):
    reservoirs = (np.ones((42, 42)),)
    reservoir_recurrence_degrees = (0,)
    trainable_reservoir = (True,)
    batch = 6
    image_shape = (20, 20, 1)
    image_set = tf.range(
        batch * image_shape[0] * image_shape[1], dtype=tf.float32)
    image_set = tf.reshape(image_set,
                           (batch, image_shape[0], image_shape[1], 1))
    num_classes = 10

    model = segmentation_models.convolution_reservoir_model(
        image_shape,
        reservoirs,
        reservoir_recurrence_degrees,
        trainable_reservoir,
        conv_filters=(1, 2, 3),
        conv_kernel_sizes=(2, 3, 4),
        conv_strides=(2, 1, 2),
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape,
                     (batch, image_shape[0], image_shape[1], num_classes))

  def test_upsample_model_should_produce_prediction_with_correct_shape(self):
    reservoirs = (np.ones((42, 42)),)
    reservoir_recurrence_degrees = (0,)
    trainable_reservoir = (True,)
    batch = 6
    image_shape = (20, 20, 3)
    image_set = tf.ones((batch, *image_shape))
    num_classes = 5

    model = segmentation_models.convolution_reservoir_model(
        image_shape,
        reservoirs,
        reservoir_recurrence_degrees,
        trainable_reservoir,
        upsample_convolution=True,
        conv_filters=(1, 2, 3),
        conv_kernel_sizes=(2, 3, 4),
        conv_strides=(2, 1, 2),
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))

  def test_it_raises_error_with_incorrectly_chained_reservoir_matrices(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)))
    reservoir_recurrence_degrees = (0, 0)
    trainable_reservoir = (True, True)
    image_shape = (20, 20, 1)

    with self.assertRaisesRegex(ValueError, 'Reservoir weight matrix'):
      segmentation_models.convolution_reservoir_model(
          image_shape,
          reservoirs,
          reservoir_recurrence_degrees,
          trainable_reservoir,
          upsample_convolution=True,
          conv_filters=(1, 2, 3),
          conv_kernel_sizes=(2, 3, 4),
          conv_strides=(3, 4, 5),
      )

  def test_it_raises_error_if_reservoirs_not_match_recurrence_degrees(self):
    reservoirs = (np.ones((3, 3)), np.ones((3, 3)))
    reservoir_recurrence_degrees = (1, 2, 3)
    trainable_reservoir = (True, True, True)
    image_shape = (20, 20, 1)

    with self.assertRaisesRegex(ValueError, 'reservoir_weights has'):
      segmentation_models.convolution_reservoir_model(
          image_shape,
          reservoirs,
          reservoir_recurrence_degrees,
          trainable_reservoir,
          upsample_convolution=True,
          conv_filters=(1, 2, 3),
          conv_kernel_sizes=(2, 3, 4),
          conv_strides=(3, 4, 5),
      )

  def test_it_raises_error_with_unmatched_lengths_of_conv_arguments(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'the same number of elements'):
      segmentation_models.convolution_reservoir_model(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 0, 0),
          trainable_reservoir=(True, False, True),
          upsample_convolution=True,
          conv_filters=(1, 2, 3, 4),
          conv_kernel_sizes=(2, 3, 4),
          conv_strides=(2, 1, 2),
      )

  def test_multi_reservoirs_should_produce_prediction_with_correct_shape(self):
    reservoirs = (np.ones((2, 3)), np.ones((3, 4)), np.ones((4, 5)))
    reservoir_recurrence_degrees = (0, 0, 0)
    trainable_reservoir = (True, True, True)
    batch = 6
    image_shape = (20, 20, 1)
    image_set = tf.ones((batch, *image_shape))
    num_classes = 4

    model = segmentation_models.convolution_reservoir_model(
        image_shape,
        reservoirs,
        reservoir_recurrence_degrees,
        trainable_reservoir,
        upsample_convolution=True,
        conv_filters=(1, 2, 3),
        conv_kernel_sizes=(2, 3, 4),
        conv_strides=(2, 1, 2),
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))

  def test_model_with_recurrent_reservoir_returns_expected_output_shape(self):
    reservoirs = (np.ones((42, 42)),)
    reservoir_recurrence_degrees = (3,)
    trainable_reservoir = (True,)
    batch = 6
    image_shape = (20, 20, 1)
    image_set = tf.range(
        batch * image_shape[0] * image_shape[1], dtype=tf.float32)
    image_set = tf.reshape(image_set,
                           (batch, image_shape[0], image_shape[1], 1))
    num_classes = 2

    model = segmentation_models.convolution_reservoir_model(
        image_shape,
        reservoirs,
        reservoir_recurrence_degrees,
        trainable_reservoir,
        conv_filters=(1, 2, 3),
        conv_kernel_sizes=(2, 3, 4),
        conv_strides=(2, 1, 2),
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape,
                     (batch, image_shape[0], image_shape[1], num_classes))


class BuildFlySensorModelTest(absltest.TestCase):
  """Tests `segmentation_models.selective_sensor_model`."""

  def test_it_raises_error_with_incorrectly_chained_reservoir_matrices(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    image_shape = (20, 20, 1)

    reservoir_recurrence_degrees = (0, 1, 0)
    trainable_reservoir = (True, True, True)
    reservoir_params_set = []
    for i in range(len(trainable_reservoir)):
      reservoir_params_set.append({
          'recurrence_degree': reservoir_recurrence_degrees[i],
          'trainable_reservoir': trainable_reservoir[i]
      })

    with self.assertRaisesRegex(ValueError, 'Reservoir weight matrix'):
      segmentation_models.selective_sensor_model(
          input_shape=image_shape,
          reservoir_weights=reservoirs,
          reservoir_params_set=tuple(reservoir_params_set),
          conv_filters=(1, 2, 1),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
          num_sensors_per_channel=2,
      )

  def test_it_raises_error_if_sensor_not_match_first_reservoir(self):
    reservoirs = (np.ones((12, 23)), np.ones((23, 34)), np.ones((34, 56)))
    num_sensors_per_channel = 7
    input_shape = (20, 20, 1)
    reservoir_recurrence_degrees = (0, 1, 0)
    trainable_reservoir = (True, True, True)
    reservoir_params_set = []
    for i in range(len(trainable_reservoir)):
      reservoir_params_set.append({
          'recurrence_degree': reservoir_recurrence_degrees[i],
          'trainable_reservoir': trainable_reservoir[i]
      })
    with self.assertRaisesRegex(ValueError, 'The number of sensors ='):
      segmentation_models.selective_sensor_model(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_params_set=tuple(reservoir_params_set),
          conv_filters=(1, 2, 3),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
          num_sensors_per_channel=num_sensors_per_channel,
      )

  def test_it_raises_error_if_a_reservoir_is_recurrent_but_not_square(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 14)), np.ones((14, 400)))
    input_shape = (20, 20, 1)
    reservoir_recurrence_degrees = (0, 1, 0)
    trainable_reservoir = (True, True, True)
    reservoir_params_set = []
    for i in range(len(trainable_reservoir)):
      reservoir_params_set.append({
          'recurrence_degree': reservoir_recurrence_degrees[i],
          'trainable_reservoir': trainable_reservoir[i]
      })
    with self.assertRaisesRegex(ValueError, 'There is a reservoir that does'):
      segmentation_models.selective_sensor_model(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_params_set=tuple(reservoir_params_set),
          conv_filters=(1, 2, 2),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
          num_sensors_per_channel=10,
      )

  def test_it_raises_error_with_unmatched_lengths_of_conv_arguments(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    input_shape = (20, 20, 1)
    reservoir_recurrence_degrees = (0, 1, 0)
    trainable_reservoir = (True, True, True)
    reservoir_params_set = []
    for i in range(len(trainable_reservoir)):
      reservoir_params_set.append({
          'recurrence_degree': reservoir_recurrence_degrees[i],
          'trainable_reservoir': trainable_reservoir[i]
      })
    with self.assertRaisesRegex(ValueError, 'the same number of elements'):
      segmentation_models.selective_sensor_model(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_params_set=tuple(reservoir_params_set),
          conv_filters=(5, 10),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(1, 2, 3, 4),
          num_sensors_per_channel=2,
      )

  def test_it_should_produce_prediction_with_correct_shape(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 14)), np.ones((14, 15)))
    batch = 2
    image_shape = (24, 24, 3)
    image_set = tf.range(batch * tf.reduce_prod(image_shape), dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))
    num_classes = 10
    reservoir_recurrence_degrees = (0, 0, 0)
    trainable_reservoir = (True, True, True)
    reservoir_params_set = []
    for i in range(len(trainable_reservoir)):
      reservoir_params_set.append({
          'recurrence_degree': reservoir_recurrence_degrees[i],
          'trainable_reservoir': trainable_reservoir[i]
      })
    model = segmentation_models.selective_sensor_model(
        input_shape=image_shape,
        reservoir_weights=reservoirs,
        reservoir_params_set=tuple(reservoir_params_set),
        conv_filters=(3, 4, 5),
        conv_kernel_sizes=(3, 4, 5),
        conv_strides=(2, 1, 2),
        num_sensors_per_channel=4,
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))

  def test_it_with_recurrence_also_produces_prediction_with_correct_shape(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    batch = 2
    image_shape = (24, 24, 2)
    image_set = tf.range(batch * tf.reduce_prod(image_shape), dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))
    num_classes = 10
    reservoir_recurrence_degrees = (0, 7, 0)
    trainable_reservoir = (True, True, True)
    reservoir_params_set = []
    for i in range(len(trainable_reservoir)):
      reservoir_params_set.append({
          'recurrence_degree': reservoir_recurrence_degrees[i],
          'trainable_reservoir': trainable_reservoir[i]
      })
    model = segmentation_models.selective_sensor_model(
        input_shape=image_shape,
        reservoir_weights=reservoirs,
        reservoir_params_set=tuple(reservoir_params_set),
        conv_filters=(2, 3, 4),
        conv_kernel_sizes=(3, 4, 5),
        conv_strides=(2, 1, 2),
        num_sensors_per_channel=5,
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))


class BuildFlySparseSensorModelTest(absltest.TestCase):
  """Tests `sparse_sensor_model`."""

  def test_it_raises_error_with_incorrectly_chained_reservoir_matrices(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    image_shape = (20, 20, 1)

    with self.assertRaisesRegex(ValueError, 'Reservoir weight matrix'):
      segmentation_models.sparse_sensor_model(
          input_shape=image_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(2, 3, 2),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 3, 4),
          num_sensors=2,
      )

  def test_it_raises_error_if_first_reservoir_cannot_be_sparse_sensor(self):
    reservoirs = (np.ones((12, 23)), np.ones((23, 34)), np.ones((34, 56)))
    num_sensors = 7
    input_shape = (20, 20, 1)

    with self.assertRaisesRegex(ValueError, 'For the first reservoir_weight'):
      segmentation_models.sparse_sensor_model(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(4, 3, 2),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 3, 4),
          num_sensors=num_sensors,
      )

  def test_it_raises_error_if_a_reservoir_is_recurrent_but_not_square(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 14)), np.ones((14, 400)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'There is a reservoir that does'):
      segmentation_models.sparse_sensor_model(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(2, 3, 20),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 3, 4),
          num_sensors=13,
      )

  def test_it_raises_error_with_unmatched_lengths_of_conv_arguments(self):
    reservoirs = (np.ones((10, 13)), np.ones((13, 14)), np.ones((14, 15)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'the same number of elements'):
      segmentation_models.sparse_sensor_model(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 0, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(1, 2, 3, 10),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
          num_sensors=13,
      )

  def test_it_should_produce_prediction_with_correct_shape(self):
    reservoirs = (np.ones((10, 13)), np.ones((13, 14)), np.ones((14, 15)))
    batch = 2
    image_shape = (20, 20, 1)
    image_set = tf.range(
        batch * image_shape[0] * image_shape[1], dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))
    num_classes = 4

    model = segmentation_models.sparse_sensor_model(
        input_shape=image_shape,
        reservoir_weights=reservoirs,
        reservoir_recurrence_degrees=(0, 0, 0),
        trainable_reservoir=(True, True, True),
        conv_filters=(2, 3, 10),
        conv_kernel_sizes=(1, 2, 3),
        conv_strides=(2, 1, 2),
        num_sensors=13,
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))

  def test_it_with_recurrence_also_produces_prediction_with_correct_shape(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    batch = 2
    image_shape = (20, 20, 1)
    image_set = tf.range(
        batch * image_shape[0] * image_shape[1], dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))
    num_classes = 10

    model = segmentation_models.sparse_sensor_model(
        input_shape=image_shape,
        reservoir_weights=reservoirs,
        reservoir_recurrence_degrees=(0, 7, 0),
        trainable_reservoir=(True, True, True),
        conv_filters=(2, 3, 20),
        conv_kernel_sizes=(1, 2, 3),
        conv_strides=(2, 1, 2),
        num_sensors=13,
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))


class AlternateConvolutionFlyTest(absltest.TestCase):

  def test_it_raises_error_with_unmatched_downsampling_upsampling(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    input_shape = (4, 5, 6)

    with self.assertRaisesRegex(ValueError, 'Input size must not change'):
      segmentation_models.convolution_reservoir_alternating_model(
          input_shape=input_shape,
          num_classes=10,
          add_flies=(True, True, True),
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          conv_block_types=('DownSample', 'DownSample', 'UpSample'),
          conv_block_filters=((1, 2), (2, 3), (3, 4)),
          conv_block_kernel_sizes=((2, 3), (4, 5), (6, 7)),
          conv_block_strides=((1, 2), (1, 2), (1, 2)),
      )

  def test_it_raises_error_with_unmatched_lengths_of_tuple_arguments(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    input_shape = (4, 5, 6)
    with self.assertRaisesRegex(ValueError, 'have the same number of elements'):
      segmentation_models.convolution_reservoir_alternating_model(
          input_shape=input_shape,
          num_classes=10,
          add_flies=(True, True, True),
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          conv_block_types=('DownSample', 'DownSample', 'UpSample', 'UpSample'),
          conv_block_filters=((1, 2), (2, 3), (3, 4)),
          conv_block_kernel_sizes=((2, 3), (4, 5), (6, 7)),
          conv_block_strides=((1, 2), (1, 2), (2, 2)),
      )

  def test_it_raises_error_with_non_square_reservoirs(self):
    reservoirs = (np.ones((3, 3)), np.ones((3, 4)), np.ones((5, 5)))
    input_shape = (4, 5, 6)

    with self.assertRaisesRegex(ValueError, 'used reservoirs must be square'):
      segmentation_models.convolution_reservoir_alternating_model(
          input_shape=input_shape,
          num_classes=10,
          add_flies=(False, True, False),
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          conv_block_types=('DownSample', 'DownSample', 'UpSample'),
          conv_block_filters=((1, 2), (2, 3), (3, 4)),
          conv_block_kernel_sizes=((2, 3), (4, 5), (6, 7)),
          conv_block_strides=((1, 2), (1, 2), (2, 2)),
      )

  def test_it_raises_error_with_unmatched_lengths_of_corresponding_conv_tuples(
      self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    input_shape = (4, 5, 6)

    with self.assertRaisesRegex(ValueError, 'Corresponding tuple elements of'):
      segmentation_models.convolution_reservoir_alternating_model(
          input_shape=input_shape,
          num_classes=10,
          add_flies=(True, True, True),
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          conv_block_types=('DownSample', 'DownSample', 'UpSample'),
          conv_block_filters=((1, 2), (2, 3), (3, 4)),
          conv_block_kernel_sizes=((1, 2, 3), (4, 5), (6, 7)),
          conv_block_strides=((1, 2), (1, 2), (2, 2)),
      )

  def test_it_raises_error_if_conv_block_not_connect_reservoir(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    input_shape = (4, 5, 6)

    with self.assertRaisesRegex(ValueError, 'Convolution block must connect'):
      segmentation_models.convolution_reservoir_alternating_model(
          input_shape=input_shape,
          num_classes=10,
          add_flies=(True, True, True),
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          conv_block_types=('DownSample', 'DownSample', 'UpSample'),
          conv_block_filters=((1, 1), (2, 2), (3, 3)),
          conv_block_kernel_sizes=((2, 3), (4, 5), (6, 7)),
          conv_block_strides=((1, 2), (1, 2), (2, 2)),
      )

  def test_it_produces_prediction_with_correct_shape(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    batch = 2
    image_shape = (8, 8, 3)
    image_set = tf.range(
        batch * image_shape[0] * image_shape[1] * image_shape[2],
        dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))
    num_classes = 10

    model = segmentation_models.convolution_reservoir_alternating_model(
        input_shape=image_shape,
        num_classes=10,
        add_flies=(True, False, True),
        reservoir_weights=reservoirs,
        reservoir_recurrence_degrees=(0, 1, 0),
        conv_block_types=('DownSample', 'DownSample', 'UpSample'),
        conv_block_filters=((1, 2), (2, 3), (3, 4)),
        conv_block_kernel_sizes=((1, 2), (3, 4), (5, 6)),
        conv_block_strides=((1, 2), (1, 2), (2, 2)),
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))


class MultiReservoirsTest(absltest.TestCase):

  def test_it_raises_error_with_incorrectly_chained_reservoir_matrices(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    image_shape = (20, 20, 1)

    with self.assertRaisesRegex(ValueError, 'Reservoir weight matrix'):
      segmentation_models.convolution_multi_reservoirs(
          input_shape=image_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(1, 3, 2),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
      )

  def test_it_raises_error_if_a_reservoir_is_recurrent_but_not_square(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 14)), np.ones((14, 400)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'There is a reservoir that does'):
      segmentation_models.convolution_multi_reservoirs(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(1, 10, 20),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
      )

  def test_it_raises_error_with_unmatched_lengths_of_reservoir_arguments(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, '`reservoir_weights`'):
      segmentation_models.convolution_multi_reservoirs(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True),
          conv_filters=(1, 10, 20),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
      )

  def test_it_raises_error_if_first_reservoir_not_connect_convo_head(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'The last number of convolution'):
      segmentation_models.convolution_multi_reservoirs(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, False),
          conv_filters=(1, 10, 19),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
      )

  def test_it_raises_error_with_unmatched_lengths_of_conv_arguments(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, '`conv_filters`'):
      segmentation_models.convolution_multi_reservoirs(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(5, 20),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(1, 2, 3, 4),
      )

  def test_it_should_produce_prediction_with_correct_shape(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    batch = 2
    image_shape = (24, 24, 3)
    image_set = tf.range(batch * tf.reduce_prod(image_shape), dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))
    num_classes = 10

    model = segmentation_models.convolution_multi_reservoirs(
        input_shape=image_shape,
        reservoir_weights=reservoirs,
        reservoir_recurrence_degrees=(0, 1, 0),
        trainable_reservoir=(True, True, True),
        conv_filters=(5, 10, 20),
        conv_kernel_sizes=(1, 2, 3),
        conv_strides=(2, 1, 2),
        num_classes=num_classes,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], num_classes))


class FlySensorForTwoContrastiveLabelsTest(absltest.TestCase):

  def test_it_raises_error_with_incorrectly_chained_reservoir_matrices(self):
    reservoirs = (np.ones((2, 2)), np.ones((3, 3)), np.ones((4, 4)))
    image_shape = (20, 20, 1)

    with self.assertRaisesRegex(ValueError, 'Reservoir weight matrix'):
      segmentation_models.selective_sensor_for_two_contrastive_labels(
          input_shape=image_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(1, 2, 1),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
          num_sensors_per_channel=2,
      )

  def test_it_raises_error_if_sensor_not_match_first_reservoir(self):
    reservoirs = (np.ones((12, 23)), np.ones((23, 34)), np.ones((34, 56)))
    num_sensors_per_channel = 7
    input_shape = (20, 20, 1)

    with self.assertRaisesRegex(ValueError, 'The number of sensors ='):
      segmentation_models.selective_sensor_for_two_contrastive_labels(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(1, 2, 3),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
          num_sensors_per_channel=num_sensors_per_channel,
      )

  def test_it_raises_error_if_a_reservoir_is_recurrent_but_not_square(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 14)), np.ones((14, 400)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'There is a reservoir that does'):
      segmentation_models.selective_sensor_for_two_contrastive_labels(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(1, 2, 2),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(2, 1, 2),
          num_sensors_per_channel=10,
      )

  def test_it_raises_error_with_unmatched_lengths_of_conv_arguments(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, 'the same number of elements'):
      segmentation_models.selective_sensor_for_two_contrastive_labels(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True, True),
          conv_filters=(5, 10),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(1, 2, 3, 4),
          num_sensors_per_channel=2,
      )

  def test_it_raises_error_with_unmatched_lengths_of_reservoir_arguments(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    input_shape = (20, 20, 1)
    with self.assertRaisesRegex(ValueError, '`reservoir_weights`'):
      segmentation_models.selective_sensor_for_two_contrastive_labels(
          input_shape=input_shape,
          reservoir_weights=reservoirs,
          reservoir_recurrence_degrees=(0, 1, 0),
          trainable_reservoir=(True, True),
          conv_filters=(5, 10, 10),
          conv_kernel_sizes=(1, 2, 3),
          conv_strides=(1, 2, 3),
          num_sensors_per_channel=2,
      )

  def test_it_produces_prediction_with_correct_shape(self):
    reservoirs = (np.ones((20, 13)), np.ones((13, 13)), np.ones((13, 15)))
    batch = 2
    image_shape = (24, 24, 2)
    image_set = tf.range(batch * tf.reduce_prod(image_shape), dtype=tf.float32)
    image_set = tf.reshape(image_set, (batch, *image_shape))

    model = segmentation_models.selective_sensor_for_two_contrastive_labels(
        input_shape=image_shape,
        reservoir_weights=reservoirs,
        reservoir_recurrence_degrees=(0, 7, 0),
        trainable_reservoir=(True, True, True),
        conv_filters=(2, 3, 4),
        conv_kernel_sizes=(3, 4, 5),
        conv_strides=(2, 1, 2),
        num_sensors_per_channel=5,
    )
    result = model(image_set)

    self.assertEqual(result.shape, (batch, *image_shape[:2], 1))


class BuildDeepFlyTest(absltest.TestCase):

  def test_deeplab_return_expected_output_shape_for_mobilenetv2(self):
    input_shape = (32, 64, 3)
    batch = 5
    num_classes = 4
    images = tf.reshape(
        tf.range(batch * tf.math.reduce_prod(input_shape)),
        (batch, *input_shape))

    model = segmentation_models.deeplab_inspired_reservoir_model(
        input_shape=input_shape,
        num_output_channels=num_classes,
        backbone='mobilenetv2',
        pretrained=True,
        final_activation='softmax')
    result = model(images)
    self.assertEqual(result.shape, (batch, *input_shape[:-1], num_classes))

  def test_deeplab_return_expected_output_shape_for_resnet50(self):
    input_shape = (32, 64, 3)
    batch = 5
    num_classes = 4
    images = tf.reshape(
        tf.range(batch * tf.math.reduce_prod(input_shape)),
        (batch, *input_shape))

    model = segmentation_models.deeplab_inspired_reservoir_model(
        input_shape=input_shape,
        num_output_channels=num_classes,
        backbone='resnet50',
        pretrained=False,
        final_activation='softmax')
    result = model(images)
    self.assertEqual(result.shape, (batch, *input_shape[:-1], num_classes))

  def test_deeplab_return_expected_output_shape_for_deeplab_inspired(self):
    input_shape = (32, 64, 3)
    batch = 5
    num_classes = 4
    images = tf.reshape(
        tf.range(batch * tf.math.reduce_prod(input_shape)),
        (batch, *input_shape))

    model = segmentation_models.deeplab_inspired_reservoir_model(
        input_shape=input_shape,
        num_output_channels=num_classes,
        backbone='deeplab_inspired',
        pretrained=False,
        final_activation='softmax')
    result = model(images)
    self.assertEqual(result.shape, (batch, *input_shape[:-1], num_classes))

  def test_deeplab_return_expected_output_shape_for_downsample_standard(self):
    input_shape = (32, 64, 3)
    batch = 5
    num_classes = 4
    images = tf.reshape(
        tf.range(batch * tf.math.reduce_prod(input_shape)),
        (batch, *input_shape))

    model = segmentation_models.deeplab_inspired_reservoir_model(
        input_shape=input_shape,
        num_output_channels=num_classes,
        backbone='downsample_standard',
        pretrained=False,
        final_activation='softmax')
    result = model(images)
    self.assertEqual(result.shape, (batch, *input_shape[:-1], num_classes))


if __name__ == '__main__':
  absltest.main()
