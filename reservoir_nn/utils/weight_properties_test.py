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

"""Tests for weight_properties."""

from absl.testing import absltest
from absl.testing import parameterized

import networkx as nx
import numpy as np

from reservoir_nn.utils import weight_properties


class NetworkStatisticsTest(absltest.TestCase):

  def test_it_computes_the_right_statistics_value(self):
    weight = nx.to_numpy_array(nx.house_graph())
    stats = weight_properties.get_network_statistics(weight)
    stats_values = np.array(list(stats.values()))
    expected = np.array([
        2.1, 0.3333333333333333, 0.3333333333333333, 0.8, 1.1111111111111112, 0,
        2, -0.3333333333333348, 0.6, 0.4404866569492268, 0.72,
        0.13333333333333333, 0.0, 0.52, 2.481194304092014
    ])
    # small worldness metrics are not very stable, so we are not testing them.
    np.testing.assert_allclose(stats_values[:4].round(decimals=2),
                               expected[:4].round(decimals=2))
    np.testing.assert_allclose(stats_values[6:].round(decimals=2),
                               expected[6:].round(decimals=2))


class WeightPropertiesTest(absltest.TestCase):

  def test_get_inhibitory_proportion_of_total_strength_with_all_inhibitory_weight(
      self):
    """Tests weight_properties.get_inhibitory_proportion_of_total_strength.

    If the weight matrix is all inhibitory, this function raises ValueError.
    """
    size = (10, 10)
    arr = np.random.uniform(size=size, high=-1)
    with self.assertRaises(ValueError):
      weight_properties.get_inhibitory_proportion_of_total_strength(arr)

  def test_get_inhibitory_proportion_of_total_strength_does_not_change_input(
      self):
    """Tests weight_properties.get_inhibitory_proportion_of_total_strength.

    This function should not change the input.
    """
    size = (10, 10)
    arr = np.random.uniform(size=size, low=-1, high=1)
    arr_copy = arr.copy()
    weight_properties.get_inhibitory_proportion_of_total_strength(arr)
    self.assertTrue((arr == arr_copy).all())

  def test_get_inhibitory_neuron_indices_does_not_change_input(self):
    """Tests weight_properties.get_inhibitory_neuron_indices.

    This function should not change the input.
    """
    num_neurons = 1000
    num_inhibits = 321

    arr = np.random.uniform(size=(num_neurons, num_neurons))

    inhibit_indices = np.random.choice(
        range(num_neurons), num_inhibits, replace=False)

    arr[inhibit_indices] *= -1

    arr_copy = arr.copy()
    weight_properties.get_inhibitory_neuron_indices(arr)
    np.testing.assert_equal(arr, arr_copy)

  def test_get_inhibitory_neuron_indices_number_of_inhibitory_neurons(self):
    """Tests weight_properties.get_inhibitory_neuron_indices.

    Test if this function outputs the correct number of inhibitory neurons.
    """
    num_neurons = 1000
    num_inhibits = 123

    arr = np.random.uniform(size=(num_neurons, num_neurons))

    inhibit_indices = np.random.choice(
        range(num_neurons), num_inhibits, replace=False)

    arr[inhibit_indices] *= -1

    inhibitory_neuron_indices = weight_properties.get_inhibitory_neuron_indices(
        arr)
    self.assertEqual(num_inhibits, inhibitory_neuron_indices.size)

  def test_get_inhibitory_neuron_indices_rows_have_elements_of_both_signs(self):
    """Tests weight_properties.get_inhibitory_neuron_indices.

    If there is a row that has both positive and negative elements,
    this function should raise a ValueError.
    """
    arr = np.array([[0, 2, 3, 4, 5], [0, 2, 0, -4, 5], [0, -2, 3, 4, 5],
                    [-1, 2, 3, 4, 0], [0, -2, 3, 0, -5]])

    with self.assertRaises(ValueError):
      weight_properties.get_inhibitory_neuron_indices(arr)


class GetSparsityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('a_zero', np.array([0]), 0, 1),
      ('some_zeros_1d', np.array([0, 0, 0]), 0, 1),
      ('some_zeros_2d', np.array([[0, 0, 0], [0, 0, 0]]), 0, 1),
      ('a_one', np.array([1]), 0, 0),
      ('some_ones_1d', np.array([1, 0, 1, 0]), 0, 0.5),
      ('all_ones_2d', np.array([[1, 1, 1], [1, 1, 1]]), 0, 0),
      ('general_2d', np.array([
          [0.1, 0.4, 0],
          [1, 2, -0.5],
          [-2, 0, 0],
      ]), 0, 1 / 3),
      ('general_2d_2', np.array([[0, 0], [0, 0.001]]), 0, 0.75),
      ('general_2d_with_thres', np.array([[1.1, -0.1], [-0.5, 0.2]]), 0.3, 0.5),
  )
  def test_it_should_calculate_the_correct_sparsity(self, matrix, thres,
                                                    sparsity):
    result = weight_properties.get_sparsity(matrix, thres)
    self.assertAlmostEqual(result, sparsity)


class GetSpectralRadiusTest(parameterized.TestCase):

  def test_it_should_calculate_the_correct_spectral_radius(self):
    matrix = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
    result = weight_properties.get_spectral_radius(matrix)
    expected = 6
    self.assertAlmostEqual(result, expected)

  @parameterized.named_parameters(
      ('1d_input', np.array([1, 2, 3])),
      ('non_square_matrix', np.array([[1, 2, 3], [4, 5, 6]])))
  def test_it_should_raise_error_on_invalid_shapes(self, matrix):
    with self.assertRaises(ValueError):
      weight_properties.get_spectral_radius(matrix)


if __name__ == '__main__':
  absltest.main()
