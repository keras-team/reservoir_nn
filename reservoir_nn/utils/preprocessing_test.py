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

"""Tests for reservoir_nn.utils.preprocessing."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import attr
from reservoir_nn.utils import preprocessing
import tensorflow as tf


@attr.s
class AddOne(preprocessing.TensorTransform):

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return input_ + 1


@attr.s
class MultiplyTwo(preprocessing.TensorTransform):

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return input_ * 2


class SequentialTest(parameterized.TestCase):

  def test_init_should_take_iterable_but_not_dataset_transform(self):
    dataset_t = preprocessing.Sequential()
    # No error
    preprocessing.Sequential([dataset_t])

    with self.assertRaises(TypeError):
      preprocessing.Sequential(dataset_t)

  def test_repr_should_run_without_error(self):
    pipeline = preprocessing.Sequential([preprocessing.MapLayer([AddOne()])])
    repr(pipeline)

  @parameterized.named_parameters(('add_then_mult', AddOne(), MultiplyTwo(), 6),
                                  ('mult_then_add', MultiplyTwo(), AddOne(), 5))
  def test_call_should_call_transformations_in_order(
      self,
      transformation_a,
      transformation_b,
      expected,
  ):
    dataset = tf.data.Dataset.from_tensor_slices([[2]])
    pipeline = preprocessing.Sequential([
        preprocessing.MapLayer([transformation_a]),
        preprocessing.MapLayer([transformation_b])
    ])
    result = list(pipeline(dataset).as_numpy_iterator())[0]
    self.assertEqual(result[0], expected)


class MapLayerTest(absltest.TestCase):

  def test_init_should_take_iterable_but_not_tensor_transform(self):
    t = preprocessing.Chain()
    # No error
    preprocessing.MapLayer([t])

    with self.assertRaises(TypeError):
      preprocessing.MapLayer(t)

  def test_map_func_should_map_across_dataset(self):
    dataset = tf.data.Dataset.from_tensor_slices(([[2]], [[3]]))
    pipeline = preprocessing.MapLayer([AddOne(), MultiplyTwo()])
    result = list(pipeline(dataset).as_numpy_iterator())[0]
    self.assertEqual(result[0], 3)
    self.assertEqual(result[1], 6)

  def test_map_func_should_error_when_num_input_argument_not_equal_num_transformations(
      self):
    dataset = tf.data.Dataset.from_tensor_slices([[2]])
    pipeline = preprocessing.MapLayer([AddOne(), MultiplyTwo()])
    with self.assertRaises(AttributeError):
      pipeline(dataset)


class ParseDataLayerTest(absltest.TestCase):

  def test_call_will_run_the_parser_with_no_error(self):
    # Create a dataset that has two entries
    dataset = tf.data.Dataset.from_tensor_slices([2, 3])
    parser = lambda x: (x, x * 2)
    pipeline = preprocessing.ParseDataLayer(parser)
    result = list(pipeline(dataset).as_numpy_iterator())
    tf.debugging.assert_equal(result[0], (2, 4))
    tf.debugging.assert_equal(result[1], (3, 6))

  def test_call_will_run_the_parser_with_no_error_with_flat_map(self):
    # Create a dataset that has two entries
    dataset = tf.data.Dataset.from_tensor_slices([2, 3])
    parser = lambda x: tf.data.Dataset.from_tensor_slices([[x, x * 2]])
    # Set deterministic to True to maintain ordering
    pipeline = preprocessing.ParseDataLayer(
        parser, flat_map=True, deterministic=True)
    result = list(pipeline(dataset).as_numpy_iterator())
    tf.debugging.assert_equal(result[0], (2, 4))
    tf.debugging.assert_equal(result[1], (3, 6))

  def test_repr_should_return_a_string(self):
    pipeline = preprocessing.ParseDataLayer(lambda x: x)
    result = repr(pipeline)
    self.assertRegexMatch(result, [r'^ParseDataLayer\(.+\)$'])


class BatchLayerTest(absltest.TestCase):

  @mock.patch.object(tf.data, 'Dataset', autospec=True, instance=True)
  def test_it_will_call_dataset_batch(self, mock_dataset):
    preprocessing.BatchLayer(1)(mock_dataset)
    mock_dataset.batch.assert_called_once_with(1)


class CacheLayerTest(absltest.TestCase):

  @mock.patch.object(tf.data, 'Dataset', autospec=True, instance=True)
  def test_it_will_call_dataset_cache(self, mock_dataset):
    preprocessing.CacheLayer()(mock_dataset)
    mock_dataset.cache.assert_called_once()


class IdentityTest(absltest.TestCase):

  def test_call_should_return_input_unaltered(self):
    input_ = tf.constant([-1, 0, 42])
    transformation = preprocessing.Identity()
    result = transformation(input_)
    tf.debugging.assert_equal(result, input_)


class ChainTest(parameterized.TestCase):

  def test_repr_should_run_without_error(self):
    transformation = preprocessing.Chain([AddOne(), MultiplyTwo()])
    repr(transformation)

  @parameterized.named_parameters(
      ('add_then_mult', AddOne(), MultiplyTwo(), tf.constant([0, 2, 86])),
      ('mult_then_add', MultiplyTwo(), AddOne(), tf.constant([-1, 1, 85])))
  def test_call_should_call_transformations_in_order(
      self,
      transformation_a,
      transformation_b,
      expected,
  ):
    input_ = tf.constant([-1, 0, 42])
    transformation = preprocessing.Chain([transformation_a, transformation_b])
    result = transformation(input_)
    tf.debugging.assert_equal(result, expected)


class LambdaTest(absltest.TestCase):

  def test_call_should_apply_the_provided_function(self):
    plus_two_func = lambda x: x + 2.0

    input_ = tf.constant([[42.0], [-1.0]])
    transformation = preprocessing.Lambda(plus_two_func)
    result = transformation(input_)
    tf.debugging.assert_equal(result, input_ + 2.0)

  def test_repr_should_run_without_errors(self):
    transformation = preprocessing.Lambda(lambda x: x + 2.0)
    transformation.__repr__()


class DemeanTensorTest(parameterized.TestCase):

  def test_demean_tensor_should_produce_zero_mean(self):
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    transformation = preprocessing.DemeanTensor()
    result = transformation(input_)
    tf.debugging.assert_equal(result, input_ - 1.0)
    tf.debugging.assert_equal(tf.reduce_mean(result), 0.0)

  @parameterized.named_parameters(('pos_mean_out', 0.5), ('neg_mean_out', -0.5))
  def test_demean_tensor_should_produce_mean_equal_to_meanval(self, mean_val):
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    transformation = preprocessing.DemeanTensor(mean_val=mean_val)
    result = transformation(input_)
    tf.debugging.assert_equal(result, input_ - 1.0 + mean_val)
    tf.debugging.assert_equal(tf.reduce_mean(result), mean_val)

  def test_demean_tensor_should_avoid_zeros(self):
    padding = tf.zeros([2, 2])  # Zero padding.
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    input_ = tf.concat([padding, input_], 0)
    transformation = preprocessing.DemeanTensor(avoid_zeros=True)
    result = transformation(input_)
    tf.debugging.assert_equal(result[:2, :], padding)
    tf.debugging.assert_equal(result[2:, :], input_[2:, :] - 1.0)
    tf.debugging.assert_equal(tf.reduce_mean(result), 0.0)

  @parameterized.named_parameters(('pos_mean_out', 0.5), ('neg_mean_out', -0.5))
  def test_demean_tensor_should_produce_padding_equal_to_meanval_when_avoidzeros_true(
      self, mean_val):
    padding = tf.zeros([2, 2])  # Zero padding.
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    input_ = tf.concat([padding, input_], 0)
    transformation = preprocessing.DemeanTensor(
        mean_val=mean_val, avoid_zeros=True)
    result = transformation(input_)
    tf.debugging.assert_equal(result[:2, :], padding + mean_val)
    tf.debugging.assert_equal(result[2:, :], input_[2:, :] - 1.0 + mean_val)
    tf.debugging.assert_equal(tf.reduce_mean(result), mean_val)


class NormalizeTensorTest(parameterized.TestCase):

  def test_normalize_tensor_should_normalize_between_minus_1_and_1(self):
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    transformation = preprocessing.NormalizeTensor()
    result = transformation(input_)
    tf.debugging.assert_greater_equal(result, -1.0)
    tf.debugging.assert_less_equal(result, 1.0)

  def test_normalize_tensor_should_normalize_between_minus_maxval_and_maxval(
      self):
    max_val = 0.5
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    transformation = preprocessing.NormalizeTensor(max_val=max_val)
    result = transformation(input_)
    tf.debugging.assert_greater_equal(result, -max_val)
    tf.debugging.assert_less_equal(result, max_val)

  def test_normalize_tensor_should_normalize_between_0_and_1_when_zeromin_true(
      self):
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    transformation = preprocessing.NormalizeTensor(zero_min=True)
    result = transformation(input_)
    tf.debugging.assert_greater_equal(result, 0.0)
    tf.debugging.assert_less_equal(result, 1.0)

  def test_normalize_tensor_should_normalize_between_0_and_maxval_when_zeromin_true(
      self):
    max_val = 0.5
    input_ = tf.constant([[3.0, -2.0], [-1.0, 4.0]])  # Mean of exactly 1.0.
    transformation = preprocessing.NormalizeTensor(
        max_val=max_val, zero_min=True)
    result = transformation(input_)
    tf.debugging.assert_greater_equal(result, 0.0)
    tf.debugging.assert_less_equal(result, max_val)

  def test_normalize_tensor_return_zero_when_input_is_zero(self):
    max_val = 1
    input_ = tf.constant([[0.0, 0.0], [0.0, 0.0]])
    transformation = preprocessing.NormalizeTensor(max_val=max_val)
    result = transformation(input_)
    tf.debugging.check_numerics(result, 'Result contains NaN')
    tf.debugging.assert_equal(input_, result)


class ReshapeTest(absltest.TestCase):

  def test_it_should_call_reshape_should_produce_the_correct_shape(self):
    shape = (42, 2, 3)
    input_ = tf.range(tf.reduce_prod(shape))
    transformation = preprocessing.Reshape(shape)
    result = transformation(input_)
    self.assertNotEqual(input_.shape, shape)
    self.assertEqual(result.shape, shape)


if __name__ == '__main__':
  absltest.main()
