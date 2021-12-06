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

"""Tests for weight_transforms."""
from absl.testing import absltest
from absl.testing import parameterized

import attr
import ml_collections
import numpy as np

from reservoir_nn.utils import weight_properties
from reservoir_nn.utils import weight_transforms


@attr.s
class AddOne(weight_transforms.WeightTransformation):

  def apply_transform(self, input_: np.ndarray) -> np.ndarray:
    return input_ + 1


@attr.s
class MultiplyTwo(weight_transforms.WeightTransformation):

  def apply_transform(self, input_: np.ndarray) -> np.ndarray:
    return input_ * 2


class WeightTransformationTest(parameterized.TestCase):

  def test_apply_single_arg_single_input_works(self):
    inputs = np.arange(2)
    trans = AddOne()
    [r] = trans.batch_apply(inputs)
    np.testing.assert_array_equal(r, [1, 2])

  def test_apply_multiple_arg_single_input_works(self):
    inputs = np.arange(2)
    trans = AddOne()
    [r1, r2] = trans.batch_apply(inputs, inputs)
    np.testing.assert_array_equal(r1, [1, 2])
    np.testing.assert_array_equal(r2, [1, 2])


class ChainTest(parameterized.TestCase):

  def test_repr_should_run_without_error(self):
    transformation = weight_transforms.Chain([AddOne(), MultiplyTwo()])
    repr(transformation)

  @parameterized.named_parameters(
      ('add_then_mult', AddOne(), MultiplyTwo(), np.array([0, 2, 86])),
      ('mult_then_add', MultiplyTwo(), AddOne(), np.array([-1, 1, 85])))
  def test_call_should_call_transformations_in_order(
      self,
      transformation_a,
      transformation_b,
      expected,
  ):
    input_ = np.array([-1, 0, 42])
    transformation = weight_transforms.Chain(
        [transformation_a, transformation_b])
    result = transformation.apply_transform(input_)
    np.testing.assert_equal(result, expected)


class FlipSignEntireRowsTest(parameterized.TestCase):

  @parameterized.named_parameters(('zero', 0), ('all', 1), ('half', 0.5),
                                  ('point_three', 0.3))
  def test_it_results_in_correct_flipping_proportion(self, flip_proportion):
    """Tests weight_transforms.flip_sign_entire_rows.

    Number of negative rows should match the flipping proportion.

    Args:
      flip_proportion: proportion of elements to be flipped to negative
    """
    num_neurons = 101
    arr = np.random.uniform(size=(num_neurons, num_neurons))
    result = weight_transforms.FlipSignEntireRows(
        flip_proportion).apply_transform(arr)
    num_negative_rows = len(np.nonzero(np.any(result < 0, axis=1))[0])
    self.assertEqual(int(num_neurons * flip_proportion), num_negative_rows)

  def test_it_indeed_flips_entire_rows(self):
    """Tests weight_transforms.flip_sign_entire_rows.

    It should indeed flips entire rows.
    """
    flip_proportion = 0.5
    size = (10, 10)
    arr = np.random.uniform(size=size)
    result = weight_transforms.FlipSignEntireRows(
        flip_proportion).apply_transform(arr)
    for i in range(10):
      row = result[i]
      num_positives = np.sum(row > 0)
      self.assertIn(num_positives, [0, 10])


class FlipSignEntireRowsFuncTest(parameterized.TestCase):

  @parameterized.named_parameters(('zero', 0), ('all', 1), ('half', 0.5),
                                  ('point_three', 0.3))
  def test_it_results_in_correct_flipping_proportion(self, flip_proportion):
    """Tests weight_transforms.flip_sign_entire_rows.

    Number of negative elements should match the flipping proportion.

    Args:
      flip_proportion: proportion of elements to be flipped to negative
    """
    num_neurons = 101
    arr = np.random.uniform(size=(num_neurons, num_neurons))
    result = weight_transforms.flip_sign_entire_rows(arr, flip_proportion)
    num_negative_rows = len(np.nonzero(np.any(result < 0, axis=1))[0])
    self.assertEqual(int(num_neurons * flip_proportion), num_negative_rows)

  def test_it_indeed_flips_entire_rows(self):
    """Tests weight_transforms.flip_sign_entire_rows.

    It should indeed flips entire rows.
    """
    flip_proportion = 0.5
    size = (10, 10)
    arr = np.random.uniform(size=size)
    result = weight_transforms.flip_sign_entire_rows(arr, flip_proportion)
    for i in range(10):
      row = result[i]
      num_positives = np.sum(row > 0)
      self.assertIn(num_positives, [0, 10])


class ScaleWeightByDistanceTest(parameterized.TestCase):

  @parameterized.parameters(
      ('linear',
       np.array([[1, 0.2, 0.27272727272727], [0.333333333333, 5, 0.46153846153],
                 [0.5, 0.53333333333, 9]])),
      ('quadratic',
       np.array([[1, 0.02, 0.02479338842], [0.02777777777, 5, 0.03550295857],
                 [0.03571428571, 0.03555555555, 9]])),
      ('exponential',
       np.array([[0.81873075307, 0.27067056647, 0.33240947508],
                 [0.36287181315, 4.09365376539, 0.44564146928],
                 [0.42567043837, 0.39829654694, 7.3685767777]])))
  def test_it_returns_correct_output(
      self,
      scaling_function,
      expected_matrix,
  ):
    """Tests weight_transforms.scale_weight_by_distance.

    It should result in corrected scaled weight matrix. Note that the diagonal
    of distances are converted to 1.

    Args:
      scaling_function: Function to scale weights with distances.
      expected_matrix: Expected weight matrix after being scaled.
    """
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    distances = np.array([[0, 10, 11], [12, 0, 13], [14, 15, 0]])
    box_size = 10
    transformation = weight_transforms.ScaleWeightByDistance(
        scaling_function, box_size)
    result = transformation.apply_transform(weights, distances)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=7)


class RandomizeNonZeroWeightTest(parameterized.TestCase):
  """Tests weight_transforms.RandomizeNonZeroWeight."""

  def test_it_does_not_change_input(self):
    arr = np.random.uniform(size=(100, 100), low=-10, high=10)
    arr_copy = arr.copy()
    transformation = weight_transforms.RandomizeNonZeroWeight()
    transformation.apply_transform(arr)
    np.testing.assert_equal(arr, arr_copy)

  def test_it_returns_the_same_shape_sparcity(self):
    transformation = weight_transforms.RandomizeNonZeroWeight()
    arr = np.array([[0, 2, 3, 0, 5, 6, 0, 8, 9, 10],
                    [0, -2, -3, -4, -5, -6, 0, -8, -9, 0],
                    [10, 2, 3, 0, 0, 6, 7, 8, 9, 0],
                    [0, 2, 0, 0, 5, 16, 0, 28, 9, 0],
                    [0, -2, -3, -4, -5, -6, -7, -8, -9, 0],
                    [0, -2, -3, -4, -51, 0, -7, 0, -9, -10],
                    [0, 0, 3, 4, 5, 6, 7, 8, 9, 0],
                    [0, 0, 23, 4, 15, 6, 0, 8, 9, 0],
                    [1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
                    [10, -2, -3, 0, -5, -6, 0, 0, -9, 0]])

    new_arr = transformation.apply_transform(arr)

    # same shape:
    self.assertEqual(arr.shape[0], new_arr.shape[0])
    self.assertEqual(arr.shape[1], new_arr.shape[1])

    # same sparsity (or same number of zero elements)
    arr_num_zeros = np.sum((arr == 0))
    new_arr_num_zeros = np.sum((new_arr == 0))
    self.assertEqual(arr_num_zeros, new_arr_num_zeros)


class RandomizeNonZeroWeightKdeTest(parameterized.TestCase):
  """Tests weight_transforms.RandomizeNonZeroWeightKde."""

  @parameterized.named_parameters(('no non-zero', np.array([0, 0])),
                                  ('one non-zero', np.array([0, 1, 0])))
  def test_it_raises_error_with_just_one_non_zero_element(self, arr):
    with self.assertRaisesRegex(ValueError,
                                'Expecting matrix of at least 2 non-zeros'):
      weight_transforms.RandomizeNonZeroWeightKde().apply_transform(arr)

  @parameterized.named_parameters(('one', np.array([1, 1])),
                                  ('two', np.array([2, 2, 0])))
  def test_it_raises_error_with_same_value_for_non_zero_elements(self, arr):
    with self.assertRaisesRegex(ValueError, 'Expecting different non-zeros'):
      weight_transforms.RandomizeNonZeroWeightKde().apply_transform(arr)

  def test_it_does_not_change_input(self):
    transformation = weight_transforms.RandomizeNonZeroWeightKde()
    arr = np.random.uniform(size=(100, 100), low=-10, high=10)
    arr_copy = arr.copy()
    transformation.apply_transform(arr)
    np.testing.assert_equal(arr, arr_copy)

  def test_it_returns_the_same_shape_sparcity(self):
    transformation = weight_transforms.RandomizeNonZeroWeightKde()
    arr = np.array([[0, 2, 3, 4, 5, 6, 0, 8, 9, 10],
                    [0, -2, -3, -4, -5, -6, 0, -8, -9, 0],
                    [0, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [0, 2, 0, 4, 5, 16, 7, 28, 9, 0],
                    [0, -2, -3, -4, -5, -6, -7, -8, -9, 0],
                    [0, -2, -3, -4, -51, 0, -7, 0, -9, -10],
                    [0, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [0, 2, 23, 4, 15, 6, 0, 8, 9, 0],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                    [10, -2, -3, 0, -5, -6, -7, 0, -9, 0]])

    new_arr = transformation.apply_transform(arr)

    # same shape:
    self.assertEqual(arr.shape[0], new_arr.shape[0])
    self.assertEqual(arr.shape[1], new_arr.shape[1])

    # same sparsity (or same number of zero elements)
    arr_num_zeros = np.sum((arr == 0))
    new_arr_num_zeros = np.sum((new_arr == 0))
    self.assertEqual(arr_num_zeros, new_arr_num_zeros)


class CutOffSmallWeightsInRowsTest(parameterized.TestCase):
  """Tests weight_transforms.CutOffSmallWeightsInRows."""

  @parameterized.parameters((100, 100), (100, 101), (100, 123))
  def test_it_does_not_change_input(self, arr_size, non_zeros_per_row_limit):
    """Tests weight_transforms.CutOffSmallWeightsInRows.

    If non_zeros_per_row_limit is larger than or equal to the number of columns
    of the weight matrix, this function should not change weights.

    Args:
      arr_size: number of rows/columns of the weight matrix
      non_zeros_per_row_limit: limit of non-zeros to be retained per row of the
        weight matrix
    """
    transformation = weight_transforms.CutOffSmallWeightsInRows(
        non_zeros_per_row_limit)
    arr = np.empty(shape=(arr_size, arr_size))
    for i in range(arr_size):
      arr[i] = np.arange(i, i + arr_size)

    arr_copy = arr.copy()
    arr_transformed = transformation.apply_transform(arr)
    np.testing.assert_equal(arr_transformed, arr_copy)

  @parameterized.parameters((123, 12), (234, 23), (456, 34))
  def test_num_non_zeros_must_be_within_limit(self, arr_size,
                                              non_zeros_per_row_limit):
    """Tests weight_transforms.CutOffSmallWeightsInRows.

    The transformed matrix must have number of non-zeros per row not exceeding
    non_zeros_per_row_limit.

    Args:
      arr_size: number of rows/columns of the weight matrix
      non_zeros_per_row_limit: limit of non-zeros to be retained per row of the
        weight matrix.
    """
    transformation = weight_transforms.CutOffSmallWeightsInRows(
        non_zeros_per_row_limit)
    arr = np.empty(shape=(arr_size, arr_size))
    for i in range(arr_size):
      arr[i] = np.arange(i, i + arr_size)

    arr_transformed = transformation.apply_transform(arr)

    for i in range(arr_size):
      num_non_zeros = np.sum((arr_transformed[i] > 0))
      self.assertLessEqual(num_non_zeros, non_zeros_per_row_limit)


class WeightTransformsTest(parameterized.TestCase):

  def test_make_sparse_output_has_target_sparsity(self):
    """Test output has zero_weight_proportion as intended."""

    input_weights = np.random.rand(100, 100)
    target_sparsity = 0.1
    num_elements = np.size(input_weights)
    target_zeros = np.rint(target_sparsity * num_elements)
    output_weights = weight_transforms.make_sparse(input_weights,
                                                   target_sparsity)
    num_zeros_out = np.sum(output_weights == 0)
    self.assertEqual(num_zeros_out, target_zeros)

  def test_scale_weight_by_distance_raises_error_with_wrong_scaling_function_input(
      self):
    """Tests weight_transforms.scale_weight_by_distance.

    It should raise ValueError if the scaling_distance input is not one of:
    'linear', 'quadratic', 'exponential'
    """
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    distances = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    scaling_function = 'some_function'
    with self.assertRaises(ValueError):
      weight_transforms.scale_weight_by_distance(weights, distances,
                                                 scaling_function)

  @parameterized.parameters(
      ('linear',
       np.array([[1, 0.2, 0.27272727272727], [0.333333333333, 5, 0.46153846153],
                 [0.5, 0.53333333333, 9]])),
      ('quadratic',
       np.array([[1, 0.02, 0.02479338842], [0.02777777777, 5, 0.03550295857],
                 [0.03571428571, 0.03555555555, 9]])),
      ('exponential',
       np.array([[0.81873075307, 0.27067056647, 0.33240947508],
                 [0.36287181315, 4.09365376539, 0.44564146928],
                 [0.42567043837, 0.39829654694, 7.3685767777]])))
  def test_scale_weight_by_distance_returns_correct_output(
      self, scaling_function, expected_matrix):
    """Tests weight_transforms.scale_weight_by_distance.

    It should result in corrected scaled weight matrix. Note that the diagonal
    of distances are converted to 1.

    Args:
      scaling_function: function to scale weights with distances
      expected_matrix: expected weight matrix after being scaled
    """
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    distances = np.array([[0, 10, 11], [12, 0, 13], [14, 15, 0]])
    box_size = 10
    result = weight_transforms.scale_weight_by_distance(weights, distances,
                                                        scaling_function,
                                                        box_size)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=7)

  def test_transform_weight_returns_expected_array(self):
    weights = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    distances = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    params = ml_collections.ConfigDict()
    params.box_size = 1

    # simply normalize weights to (0, 1]:
    params.distance_scaling_function = 'none'
    params.num_cutoff = 10
    params.random_weights = False
    params.inhibitory_neurons_proportion = 0.0
    expected = np.array([[0, 0.125, 0.25], [0.375, 0.5, 0.625],
                         [0.75, 0.875, 1]])
    result = weight_transforms.transform_weight(weights, distances, params)
    np.testing.assert_equal(result, expected)

    # test distance_scaling_function = 'linear':
    params.distance_scaling_function = 'linear'
    expected = np.array([[0, 0.5, 0.6666666667], [0.75, 4, 0.83333333333],
                         [0.85714285714, 0.875, 8]]) / 8  # max = 8
    result = weight_transforms.transform_weight(weights, distances, params)
    np.testing.assert_almost_equal(result, expected, decimal=7)

    # test flipping proportion = 0.4, meaning only one row is flipped:
    params.inhibitory_neurons_proportion = 0.4
    result = weight_transforms.transform_weight(weights, distances, params)
    num_negative_rows = len(np.nonzero(np.any(result < 0, axis=1))[0])
    self.assertEqual(num_negative_rows, 1)


class ScaleToZeroOneTest(absltest.TestCase):

  def test_scale_to_zero_one_returns_expected_result(self):
    arr = np.array([[1, -2], [-3, 4], [0, 5]])
    arr_copy = arr
    expected = np.array([[0.5, 0.125], [0, 0.875], [0.375, 1]])  # x = (x+3)/8
    result = weight_transforms.ScaleToZeroOne().apply_transform(arr)
    np.testing.assert_equal(result, expected)
    # transformation should not change input:
    np.testing.assert_equal(arr, arr_copy)


class SetUpWeightTransformationTest(absltest.TestCase):

  def test_setup_weight_transformation_returns_expected_list(self):
    params = ml_collections.ConfigDict()
    params.box_size = 1
    params.distance_scaling_function = 'linear'
    params.num_cutoff = 10
    params.random_weights = True
    params.kde_random_weights = False
    params.inhibitory_neurons_proportion = 0.2
    transform_list = weight_transforms.setup_weight_transformation(params)
    self.assertIsInstance(transform_list[0],
                          weight_transforms.ScaleWeightByDistance)
    self.assertIsInstance(transform_list[1],
                          weight_transforms.CutOffSmallWeightsInRows)
    self.assertIsInstance(transform_list[2],
                          weight_transforms.RandomizeNonZeroWeight)
    self.assertIsInstance(transform_list[3], weight_transforms.ScaleToZeroOne)
    self.assertIsInstance(transform_list[4],
                          weight_transforms.FlipSignEntireRows)


class ScaleSpectralRadiusTest(absltest.TestCase):

  def test_it_should_produce_matrix_of_correct_spectral_radius(self):
    # matrix: spectral_radius=6
    matrix = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
    expected = 4.2
    # Use the expected as the parameter to ScaleSpectralRadius
    scaled = weight_transforms.ScaleSpectralRadius(
        spectral_radius=expected).apply_transform(matrix)
    result = weight_properties.get_spectral_radius(scaled)

    self.assertAlmostEqual(result, expected)


class GetSubMatrixTest(parameterized.TestCase):

  @parameterized.parameters((123, 123), (321, 322), (234, 432))
  def test_it_returns_original_matrix(self, num_rows, num_neurons):
    """The case where num_rows of original matrix <= num_neurons."""
    matrix = np.arange(num_rows * num_rows).reshape(num_rows, num_rows)
    result = weight_transforms.GetSubMatrix(num_neurons).apply_transform(matrix)
    np.testing.assert_equal(result, matrix)

  @parameterized.parameters((123, 12), (456, 45), (789, 78))
  def test_it_returns_expected_result(self, num_rows, num_neurons):
    """The case where num_rows of original matrix is larger than num_neurons."""
    matrix = np.arange(num_rows * num_rows).reshape(num_rows, num_rows)
    result = weight_transforms.GetSubMatrix(num_neurons).apply_transform(matrix)
    np.testing.assert_equal(result, matrix[:num_neurons, :num_neurons])


class SerializeWeightMatricesTest(absltest.TestCase):

  def test_it_returns_correctly_chained_matrices(self):
    matrices = [np.ones((7, 8)), np.ones((3, 4)), np.ones((5, 6))]
    matrices = weight_transforms.chain_weight_matrices(matrices)
    self.assertEqual(matrices[0].shape, (7, 3))
    self.assertEqual(matrices[1].shape, (3, 4))
    self.assertEqual(matrices[2].shape, (4, 6))


class ResizeNumRowsTest(parameterized.TestCase):

  @parameterized.parameters((12, 12), (34, 33))
  def test_it_returns_a_trimmed_matrix(self, num_rows, target_num_rows):
    """The case where num_rows of the original matrix >= the target."""
    matrix = np.arange(num_rows * 2).reshape(num_rows, 2)
    result = weight_transforms.ResizeNumRows(target_num_rows).apply_transform(
        matrix)
    np.testing.assert_equal(result, matrix[:target_num_rows, :])

  @parameterized.parameters((50, 60), (70, 170))
  def test_it_returns_correct_num_rows(self, num_rows, target_num_rows):
    """The case where num_rows of the original matrix < the target."""
    matrix = np.arange(num_rows * 2).reshape(num_rows, 2)
    result = weight_transforms.ResizeNumRows(target_num_rows).apply_transform(
        matrix)
    np.testing.assert_equal(result.shape[0], target_num_rows)


class ResizeNumColumnsTest(parameterized.TestCase):

  @parameterized.parameters((12, 12), (34, 33))
  def test_it_returns_a_trimmed_matrix(self, num_columns, target_num_columns):
    """The case where num_rows of the original matrix >= the target."""
    matrix = np.arange(2 * num_columns).reshape(2, num_columns)
    result = weight_transforms.ResizeNumColumns(
        target_num_columns).apply_transform(matrix)
    np.testing.assert_equal(result, matrix[:, :target_num_columns])

  @parameterized.parameters((50, 60), (70, 170))
  def test_it_returns_correct_num_cols(self, num_columns, target_num_columns):
    """The case where num_columns of the original matrix < the target."""
    matrix = np.arange(2 * num_columns).reshape(2, num_columns)
    result = weight_transforms.ResizeNumColumns(
        target_num_columns).apply_transform(matrix)
    np.testing.assert_equal(result.shape[1], target_num_columns)


class ResizeWeightMatricesTest(absltest.TestCase):

  def test_it_raises_error_on_unmatched_numbers_of_elements(self):
    shapes = ((2, 2), (3, 3))
    reservoirs = [
        np.arange(shape[0] * shape[1]).reshape(shape) for shape in shapes
    ]
    nums_neurons = (2, 3, 4)
    with self.assertRaisesRegex(ValueError, 'reservoirs_num_neurons` has'):
      weight_transforms.resize_weight_matrices(reservoirs, nums_neurons)

  def test_it_returns_expected_result(self):
    shapes = ((3, 4)), ((5, 6))
    reservoirs = [
        np.arange(shape[0] * shape[1]).reshape(shape) for shape in shapes
    ]
    nums_neurons = (7, 8)
    new_reservoirs = weight_transforms.resize_weight_matrices(
        reservoirs, nums_neurons)
    self.assertEqual(new_reservoirs[0].shape, (7, 7))
    self.assertEqual(new_reservoirs[1].shape, (8, 8))


if __name__ == '__main__':
  absltest.main()
