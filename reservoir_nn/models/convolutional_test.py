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

"""Tests for convolutional.py."""

from absl.testing import absltest
import numpy as np
from reservoir_nn.models import convolutional
import tensorflow as tf


class ReservoirInceptionTest(absltest.TestCase):

  def test_reservoir_inception_output_produce_correct_shape(self):
    reservoir = np.ones((42, 42))
    batch = 6
    image_size = (2, 3, 4)
    image_set = tf.range(batch * image_size[0] * image_size[1] * image_size[2])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = convolutional.inception_inspired_reservoir_model(
        image_size, reservoir, num_output_channels=image_size[-1])
    result = model(image_set)
    self.assertEqual(result.shape, (batch, *image_size))

  def test_reservoir_inception_output_produce_correct_shape_in_classification(
      self):
    reservoir = np.ones((42, 42))
    batch = 6
    image_size = (2, 3, 4)
    image_set = tf.range(batch * image_size[0] * image_size[1] * image_size[2])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = convolutional.inception_inspired_reservoir_model(
        image_size,
        reservoir,
        num_output_channels=image_size[-1],
        task='classification')
    result = model(image_set)
    self.assertEqual(result.shape, (batch, image_size[-1]))


class UnetReservoirSandwichTest(absltest.TestCase):

  def test_unet_sandwich_return_expected_output_shape_in_segmentation(self):
    input_shape = (32, 64, 3)
    batch = 2
    images = tf.reshape(
        tf.range(batch * tf.math.reduce_prod(input_shape)),
        (batch, *input_shape))
    reservoir_weight = np.ones((3, 3))
    num_output_channels = 4
    model = convolutional.unet_reservoir_sandwich_model(
        input_shape=input_shape,
        reservoir_weight=reservoir_weight,
        num_output_channels=num_output_channels,
        task='segmentation')
    result = model(images)
    self.assertEqual(result.shape,
                     (batch, *input_shape[:-1], num_output_channels))

  def test_unet_sandwich_return_expected_output_shape_in_3d_segmentation(self):
    if not tf.test.is_gpu_available():
      self.skipTest('This model only works on GPU. '
                    'CPU conv3d kernel does not support this mode.')

    input_shape = (32, 64, 128, 3)
    batch = 2
    images = tf.reshape(
        tf.range(batch * tf.math.reduce_prod(input_shape)),
        (batch, *input_shape))
    reservoir_weight = np.ones((3, 3))
    num_output_channels = 4
    model = convolutional.unet_reservoir_sandwich_model(
        input_shape=input_shape,
        reservoir_weight=reservoir_weight,
        num_output_channels=num_output_channels,
        task='segmentation')
    result = model(images)
    self.assertEqual(result.shape,
                     (batch, *input_shape[:-1], num_output_channels))

  def test_unet_sandwich_return_expected_output_shape_in_classification(self):
    input_shape = (32, 64, 3)
    batch = 2
    images = tf.reshape(
        tf.range(batch * tf.math.reduce_prod(input_shape)),
        (batch, *input_shape))
    reservoir_weight = np.ones((3, 3))
    num_output_channels = 4
    model = convolutional.unet_reservoir_sandwich_model(
        input_shape=input_shape,
        reservoir_weight=reservoir_weight,
        num_output_channels=num_output_channels,
        task='classification')
    result = model(images)
    self.assertEqual(result.shape, (batch, num_output_channels))


if __name__ == '__main__':
  absltest.main()
