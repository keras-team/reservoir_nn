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

"""Module for computing and printing useful properties of the weight matrix."""
import networkx as nx
import numpy as np


def print_network_statistics(weight, plot_weight=False):
  """print out useful statistics and plots for a connectome.

  Args:
    weight: the neuron-connection weight matrix, in a square numpy array format
    plot_weight: whether to plot the weight distribution. False by default.
  """
  # compute network statistics
  net_stats = get_network_statistics(weight)
  for k, v in net_stats.items():
    print(k + ': ', v)

  # plot some weight statistics
  if plot_weight:
    import matplotlib.pyplot as plt  # pylint:disable=g-import-not-at-top
    print('The histogram of the weights: ')
    plt.hist(weight.flatten())


def get_network_statistics(weight):
  """compute useful statistics for a connectome.

  Args:
    weight: the neuron-connection weight matrix, in a square numpy array format

  Returns:
    stats: a dictionary that holds useful statistics for the network or matrix

  Raises:
    ValueError: if the matrix is not square or not 2d
  """

  # check weight shape
  if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
    raise ValueError('input matrix must be 2d square, but instead shaped: ' +
                     weight.shape)
  else:
    print('The weight matrix is of shape ' + str(weight.shape))

  stats = {}
  graph = nx.from_numpy_matrix(weight)
  stats['average_node_connectivity'] = nx.average_node_connectivity(graph)
  stats['average_clustering_coefficient'] = nx.average_clustering(graph)
  stats['local_efficiency'] = nx.local_efficiency(graph)
  stats['global_efficiency'] = nx.global_efficiency(graph)
  stats['small_worldness_sigma'] = nx.sigma(graph)  # if too slow, need to tune
  stats['small_worldness_omega'] = nx.omega(graph)  # if too slow, need to tune
  stats['modularity_communities'] = len(
      list(
          nx.algorithms.community.modularity_max.greedy_modularity_communities(
              graph)))
  stats['degree_assortativity'] = nx.degree_assortativity_coefficient(graph)
  stats['average_degree_centrality'] = np.mean(
      list(nx.degree_centrality(graph).values()))
  stats['average_eigenvector_centrality'] = np.mean(
      list(nx.eigenvector_centrality(graph).values()))
  stats['average_closeness_centrality'] = np.mean(
      list(nx.closeness_centrality(graph).values()))
  stats['average_betweenness_centrality'] = np.mean(
      list(nx.betweenness_centrality(graph).values()))
  stats['inhibitory_proportion'] = get_inhibitory_proportion_of_total_strength(
      weight)
  stats['sparsity'] = get_sparsity(weight)
  stats['spectral_radius'] = get_spectral_radius(weight)

  return stats


def get_inhibitory_proportion_of_total_strength(weight):
  """Gets inhibitory weight proportion of the weight matrix.

  It is defined as the sum of the absolute values of the negative elements
  divided by the sum of the absolute values of all elements.

  Args:
    weight: the neuron-connection weight matrix, in a square numpy array format

  Returns:
    inhibitory_weight_proportion: portion of the weight that is inhibitory

  Raises:
    ValueError: if the matrix is all-inhibitory
  """

  # excitatory weights:
  excitatories = weight[weight > 0]

  if excitatories.size == 0:
    raise ValueError('input matrix is a all-inhibitory: {}'.format(weight))

  # inhibitory weights:
  inhibitories = -weight[weight < 0]

  # get proportion of weights that are inhibitory
  inhibitory_weight_proportion = np.sum(inhibitories) / (
      np.sum(inhibitories) + np.sum(excitatories))

  return inhibitory_weight_proportion


def get_inhibitory_neuron_indices(weight):
  """Gets inhibitory neuron indices of the weight matrix.

  A row (corresponding to a neuron) is inhibitory if all elements are less than
  or equal to zero.

  Args:
    weight: the neuron-connection weight matrix, in a square numpy array format

  Returns:
    inhibitory_neuron_indices: list of indices of rows of non-positive elements

  Raises:
    ValueError: if there is a row that has both positive and negative elements
  """

  # list of inhibitory neuron ids:
  inhibitory_neuron_indices = np.nonzero(np.any(weight < 0, axis=1))[0]

  # inhibitory neurons matrix:
  inhibitory_neurons = weight[inhibitory_neuron_indices, :]

  if np.any(inhibitory_neurons > 0):
    raise ValueError(
        'there is a row of the matrix {} that has both positive and negative elements'
        .format(weight))

  return inhibitory_neuron_indices


def get_sparsity(matrix: np.ndarray, thres=0.0) -> float:
  """Calculates the sparsity of the matrix.

  Args:
    matrix: A matrix.
    thres: Values with magnitue below thres are not counted.

  Returns:
    The sparsity of the matrix.
  """
  connection_count = (np.abs(matrix) > thres).sum()
  sparsity = 1 - connection_count / matrix.size
  return sparsity


def get_spectral_radius(square_matrix: np.ndarray) -> float:
  """Calculates the spectral radius of a square matrix.

  Spectral radius is the largest eigenvalue of the matrix.

  Args:
    square_matrix: A square matrix.

  Returns:
    The spectral radius of the matrix.

  Raises:
    ValueError if square_matrix is not a *square* matrix.
  """
  if not (len(square_matrix.shape) == 2 and
          square_matrix.shape[0] == square_matrix.shape[1]):
    raise ValueError(
        f'square_matrix must be a square matrix. square_matrix={square_matrix}')
  eigvals = np.linalg.eigvals(square_matrix)
  return np.max(np.abs(eigvals))
