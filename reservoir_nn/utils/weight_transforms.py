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

"""Module for operating different options on the weight matrix."""
import abc
import textwrap
from typing import List, Optional, Sequence, Tuple, Union

import deprecation
import ml_collections
import numpy as np
from reservoir_nn.utils import weight_properties
from scipy import stats


class WeightTransformation(abc.ABC):
  """Base class for WeightTransformation.

  A WeightTransformation operates on one or more np.ndarray's.
  """

  @abc.abstractmethod
  def apply_transform(self, *args: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  @deprecation.deprecated(
      details='Use WeightTransformation.apply_transform(*args). cl/366874369')
  def __call__(self, *args: np.ndarray) -> np.ndarray:
    return self.apply_transform(*args)

  def batch_apply(
      self, *args: Union[np.ndarray, Tuple[np.ndarray,
                                           ...]]) -> List[np.ndarray]:
    """Applies the transform to each of the args."""
    r = []
    for inputs in args:
      if not isinstance(inputs, tuple):
        inputs = (inputs,)
      r.append(self.apply_transform(*inputs))
    return r


class Chain(WeightTransformation, list):
  """Chains transformations to apply them in sequence."""

  def __init__(self,
               transformations: Optional[Sequence[WeightTransformation]] = None
              ):
    if transformations is None:
      list.__init__(self, [])
    else:
      list.__init__(self, transformations)

  def __repr__(self):
    transformation_str = ',\n'.join([repr(t) for t in self])
    transformation_str = textwrap.indent(transformation_str, '  ')
    return '{}([\n{}\n])'.format(self.__class__.__name__, transformation_str)

  def apply_transform(self, *args: np.ndarray) -> np.ndarray:
    inputs = args
    for i, transformation in enumerate(self):
      try:
        output = transformation.apply_transform(*inputs)
        # output will become inputs for the next transformation:
        inputs = (output,)
      except Exception as e:
        raise Exception(
            f'Error in {transformation}, the transformation at index position '
            f'{i} of this Chain') from e
    return output


class FlipSignEntireRows(WeightTransformation):
  """Flips signs of a number of rows of the matrix to negative.

  Attributes:
    flip_proportion: proportion of the number of rows to be flipped
  """

  def __init__(self, flip_proportion: float):
    self.flip_proportion = flip_proportion

  def apply_transform(self, matrix: np.ndarray, seed: int = -1) -> np.ndarray:
    # number of rows
    num_rows = matrix.shape[0]

    # number of rows to flip sign
    num_flips = int(num_rows * self.flip_proportion)

    # randomly pick ids of rows to flip sign
    if seed > -1:
      np.random.seed(seed)
    ids_row_flip = np.random.choice(num_rows, num_flips, replace=False)

    # flip sign of the selected rows
    matrix = np.copy(matrix)
    matrix[ids_row_flip] *= -1

    return matrix


class ScaleWeightByDistance(WeightTransformation):
  """Scales connection weights of neurons with their distances.

  Attributes:
    scaling_function: either 'linear', 'quadratic', or 'exponential'.
    box_size: length of one side of the cubic box, which is the region that
      contains the neurons. Only used for scaling_function = 'exponential'.

  Returns:
    weights scaled with distances.

  Raises:
    ValueError: if wrong input for scaling_function is provided.
  """

  def __init__(self, scaling_function, box_size=None):
    valid_functions = {'linear', 'quadratic', 'exponential'}
    if scaling_function not in valid_functions:
      raise ValueError(
          f'Wrong input for scale_weight_by_distance in scaling_function: '
          f'{scaling_function}. Must be one of {valid_functions}.')
    elif (scaling_function == 'exponential') and (box_size is None):
      raise ValueError(
          f'Missing argument for scale_weight_by_distance: scaling_function is '
          f'{scaling_function}, but box_size is {box_size}')

    self.scaling_function = scaling_function
    self.box_size = box_size

  def apply_transform(self, weights: np.ndarray,
                      distances: np.ndarray) -> np.ndarray:
    """Runs the step.

    Args:
      weights: weight matrix, each element is the connection weight between two
        neurons (currently representing segments).
      distances: distance matrix of the same size as weights, each element is
        the distance between the two connected neurons.

    Returns:
      Scaled weight matrix.
    """
    size = weights.shape[0]

    # pad diagonal of distances with 1.0 since they are zero:
    for n in range(size):
      distances[n][n] = 1.0

    if self.scaling_function == 'linear':
      weights = weights / distances
    elif self.scaling_function == 'quadratic':
      weights = weights / distances**2
    else:  # self.scaling_function == 'exponential'
      weights = weights * np.exp(-2 * distances / self.box_size)

    return weights


class RandomizeNonZeroWeight(WeightTransformation):
  """Starts with a weight matrix and makes it random uniformly.

  Based on the sparsity of the weight matrix, replace it with another one
  with the same sparsity, but the non-zero values are chosen randomly
  and placed at random locations.
  """

  def __init__(self):
    pass

  def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
    """Executes the transformation.

    Args:
      matrix: a square numpy array.

    Returns:
      A random numpy array of the same size and sparsity with those of input.
    """
    # calculate how many values of weight matrix are zero
    num_zero = np.sum((matrix == 0))

    # initialize a 1D random matrix
    random_weight = np.random.uniform(size=matrix.size)

    # select randomly indices of values that will be made zero
    ids_zero = np.random.choice(range(matrix.size), num_zero, replace=False)

    # make these values zero
    random_weight[ids_zero] = 0

    return random_weight.reshape(matrix.shape)


class RandomizeNonZeroWeightKde(WeightTransformation):
  """Randomizes the weight matrix using kernel-density estimate.

  This function produces a random weight matrix of the same size and sparsity
  with those of the original matrix. The non-zero elements are sampled from
  the probability density function of the original non-zero elements.

  Attributes:
    rng: The random state to use for generating arrays of random numbers.
      Default is the global np.random module.
  """

  def __init__(self, rng: np.random.RandomState = np.random):
    self.rng = rng

  def apply_transform(self, matrix: np.ndarray, seed: int = -1) -> np.ndarray:
    """Executes the transformation.

    Args:
      matrix: a square numpy array.
      seed: Optional seed for np.random.seed()

    Returns:
      A random numpy array of the same sparsity and size with those of input.
    """
    if seed > -1:
      np.random.seed(seed)

    # indices of the non-zero:
    id_non_zeros = (matrix != 0)

    # get non-zero elements
    weight_non_zero = matrix[id_non_zeros].ravel()

    # number of non-zero elements:
    num_non_zeros = weight_non_zero.size

    # There must be at least 2 non-zero elements:
    if num_non_zeros < 2:
      raise ValueError(
          f'Expecting matrix of at least 2 non-zeros, but got {num_non_zeros}.')

    # Non-zero elements must not the same:
    if not np.sum(weight_non_zero != weight_non_zero[0]):
      raise ValueError(
          f'Expecting different non-zeros, but got only {weight_non_zero[0]}.')

    # calculate the probability density function:
    density = stats.gaussian_kde(weight_non_zero)

    # get new non-zero weights:
    weight_non_zero = density.resample(num_non_zeros).ravel()

    # initiate a new random weight matrix:
    random_weight = np.zeros(matrix.size)

    # select randomly indices of values that will be made non-zero
    id_non_zeros = self.rng.choice(
        range(matrix.size), num_non_zeros, replace=False)

    # assign non-zero weights from a sample of the density function:
    random_weight[id_non_zeros] = weight_non_zero

    return random_weight.reshape(matrix.shape)


class CutOffSmallWeightsInRows(WeightTransformation):
  """Cuts off smallest weights of the weight matrix.

  So that number of non-zeros per row does not exceed non_zeros_per_row_limit.

  Attributes:
    non_zeros_per_row_limit: limit of non-zeros to be retained per row.
  """

  def __init__(self, non_zeros_per_row_limit: int):
    self.non_zeros_per_row_limit = non_zeros_per_row_limit

  def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
    """Executes the transformation.

    Args:
      matrix: the weight matrix, a 2d square numpy array.

    Returns:
      matrix_transformed: the transformed weight matrix.
    """
    num_row, num_col = matrix.shape

    # number of smallest elements to be removed per row:
    num_remove_per_row = num_col - self.non_zeros_per_row_limit

    matrix_transformed = matrix.copy()

    if num_remove_per_row > 0:

      for i in range(num_row):
        weight_row = matrix_transformed[i]

        # get indices of num_remove_per_row smallest elements of this row
        small_weight_indices = weight_row.argsort()[:num_remove_per_row]

        # and set these values zero:
        weight_row[small_weight_indices] = 0.0

    return matrix_transformed


class ScaleToZeroOne(WeightTransformation):
  """Scales the weight matrix to the [0, 1] range."""

  def __init__(self):
    pass

  def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
    matrix = matrix - matrix.min()
    return matrix / matrix.max()


class ScaleSpectralRadius(WeightTransformation):
  """Scales the weight matrix to specified spectral radius."""

  def __init__(self, spectral_radius: float = 1.0):
    self.spectral_radius = spectral_radius

  def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
    matrix_spectral_radius = weight_properties.get_spectral_radius(matrix)
    return matrix / matrix_spectral_radius * self.spectral_radius


class GetSubMatrix(WeightTransformation):
  """Returns a sub-matrix composed of `num_neurons` first rows/columns.

  If the original matrix already has number of rows/columns less than or equal
  to the input `num_neurons`, the original matrix is returned.

  Attributes:
    num_neurons: Number of first rows/columns to be retained.
  """

  def __init__(self, num_neurons: int):
    self.num_neurons = num_neurons

  def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
    # numbers of rows and columns in the original matrix
    num_rows, num_columns = matrix.shape

    if num_rows != num_columns:
      raise ValueError(f'A square matrix is expected, but input has number '
                       f'of rows = {num_rows}, which is different from '
                       f'number of columns = {num_columns}')

    if num_rows > self.num_neurons:
      return matrix[:self.num_neurons, :self.num_neurons]

    return matrix


class ResizeNumRows(WeightTransformation):
  """Resizes the weight matrix to the target number of rows.

  If the original matrix already has number of rows larger than or equal to
  to the target number of rows, it is trimmed. Otherwise, new rows are added
  that are built to maintain sparsity and KDE distribution of the original.

  Attributes:
    target_num_rows: The target number of rows.
    rng: The random state to use for generating arrays of random numbers.
      Default is the global np.random module.
  """

  def __init__(self,
               target_num_rows: int,
               rng: np.random.RandomState = np.random):
    if target_num_rows < 1:
      raise ValueError(
          f'Expecting `target_num_rows` > 0, but getting {target_num_rows}')
    self.target_num_rows = target_num_rows
    self.rng = rng

  def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
    original_num_rows = matrix.shape[0]

    # If the original matrix is larger than or equal to the target
    if original_num_rows >= self.target_num_rows:
      return matrix[:self.target_num_rows, :]

    # Otherwise:
    # Number of new rows to be added to the matrix
    num_additional_rows = self.target_num_rows - original_num_rows

    # If the additional is larger than or equal to the original
    num_new_blocks = num_additional_rows // original_num_rows
    new_matrix = [matrix]
    for _ in range(num_new_blocks):
      # The addition is of the same sparsity and KDE distribution
      random_kde = RandomizeNonZeroWeightKde(self.rng).apply_transform(matrix)
      new_matrix.append(random_kde)

    # Add the remainders
    num_remainder_rows = num_additional_rows % original_num_rows
    if num_remainder_rows:
      random_kde = RandomizeNonZeroWeightKde(self.rng).apply_transform(
          matrix[:num_remainder_rows, :])
      new_matrix.append(random_kde)

    return np.vstack(new_matrix)


class ResizeNumColumns(WeightTransformation):
  """Resizes the weight matrix to the target number of columns.

  If the original matrix already has number of columns larger than or equal to
  to the target number of columns, it is trimmed. Otherwise, new columns are
  added while maintaining sparsity and KDE distribution of the original matrix.

  Attributes:
    target_num_columns: The target number of columns.
    rng: The random state to use for generating arrays of random numbers.
      Default is the global np.random module.
  """

  def __init__(self,
               target_num_columns: int,
               rng: np.random.RandomState = np.random):
    if target_num_columns < 1:
      raise ValueError(
          f'Expecting target_num_columns > 0, but getting {target_num_columns}')
    self.target_num_columns = target_num_columns
    self.rng = rng

  def apply_transform(self, matrix: np.ndarray) -> np.ndarray:
    original_num_columns = matrix.shape[1]

    # If the original matrix is larger than or equal to the target
    if original_num_columns >= self.target_num_columns:
      return matrix[:, :self.target_num_columns]

    # Otherwise:
    # Number of new columns to be added to the matrix
    num_additional_columns = self.target_num_columns - original_num_columns

    # If the additional is larger than or equal to the original
    num_new_blocks = num_additional_columns // original_num_columns
    new_matrix = [matrix]
    for _ in range(num_new_blocks):
      # The addition is of the same sparsity and KDE distribution
      random_kde = RandomizeNonZeroWeightKde(self.rng).apply_transform(matrix)
      new_matrix.append(random_kde)

    # Add the remainders
    num_remainder_columns = num_additional_columns % original_num_columns
    if num_remainder_columns:
      random_kde = RandomizeNonZeroWeightKde(self.rng).apply_transform(
          matrix[:, :num_remainder_columns])
      new_matrix.append(random_kde)

    return np.hstack(new_matrix)


def flip_sign_entire_rows(matrix: np.ndarray,
                          flip_proportion: float) -> np.ndarray:
  """Flips signs of a number of rows of the matrix to negative.

  Args:
    matrix: the matrix to be transformed in numpy array
    flip_proportion: proportion of the number of rows to be flipped

  Returns:
    The weight matrix with signs randomly flipped in some rows.
  """

  return FlipSignEntireRows(flip_proportion).apply_transform(matrix)


def randomize_non_zero_weights_kde(matrix: np.ndarray) -> np.ndarray:
  """Randomizes the weight matrix using kernel-density estimate.

  This function produces a random weight matrix of the same size and sparsity
  with those of the original matrix. The non-zero elements are sampled from
  the probability density function of the original non-zero elements.

  Args:
    matrix: a square numpy array.

  Returns:
    A random numpy array of the same sparsity and size with those of input.
  """
  return RandomizeNonZeroWeightKde().apply_transform(matrix)


def randomize_non_zero_weights(matrix: np.ndarray) -> np.ndarray:
  """Starts with a weight matrix and makes it random uniformly.

  Based on the sparsity of the weight matrix, replace it with another one
  with the same sparsity, but the non-zero values are chosen randomly
  and placed at random locations.

  Args:
    matrix: A square matrix.

  Returns:
    A random numpy array of the same size and sparsity with those of input.
  """
  return RandomizeNonZeroWeight().apply_transform(matrix)


def shuffle_weights(matrix: np.ndarray) -> np.ndarray:
  """Shuffles the weights in the weight matrix.

  Args:
    matrix: the weight matrix.

  Returns:
    matrix_shuffled: A shuffled matrix.
  """
  nrows, ncols = matrix.shape
  matrix_shuffled = np.reshape(matrix, (nrows * ncols))
  np.random.shuffle(matrix_shuffled)
  matrix_shuffled = np.reshape(matrix_shuffled, (nrows, ncols))
  return matrix_shuffled


def assign_random_signs(matrix: np.ndarray,
                        inhibitory_proportion: float) -> np.ndarray:
  """Assigns plus or minus signs randomly to the weight matrix.

  given the proportion of connections that should be inhibitory.

  Args:
    matrix: The weight matrix.
    inhibitory_proportion: A [0, 1] number, proportion of inhibitory
      connections.

  Returns:
    The resulting matrix.
  """
  # Make all the connections positive
  matrix = np.abs(matrix)

  # Generate random matrix
  random_matrix = np.random.random(matrix.shape)

  # Select the portion randomly to reverse the sign
  inhibitory_mask = random_matrix < inhibitory_proportion
  matrix[inhibitory_mask] = -matrix[inhibitory_mask]

  return matrix


def scale_weight_by_distance(weights,
                             distances,
                             scaling_function,
                             box_size=None):
  """Scales connection weights of neurons with their distances.

  Args:
    weights: weight matrix, each element is the connection weight between two
      neurons (currently representing segments).
    distances: distance matrix of the same size as weights, each element is the
      distance between the two connected neurons.
    scaling_function: either 'linear', 'quadratic', or 'exponential'.
    box_size: length of one side of the cubic box, which is the region that
      contains the neurons. Only used for scaling_function = 'exponential'.

  Returns:
    weights scaled with distances.
  """
  return ScaleWeightByDistance(scaling_function,
                               box_size).apply_transform(weights, distances)


def make_sparse(weight_array: np.ndarray,
                zero_weight_proportion: float) -> np.ndarray:
  """Sets an arbitrary percentage (pct) of weights in an input matrix to 0.

  Args:
    weight_array: The weight array
    zero_weight_proportion:  A [0,1] number, proportion of weights that should
      be set to 0

  Returns:
    The resulting array.
  """
  num_zeros_initial = np.sum(weight_array == 0)
  if num_zeros_initial >= (weight_array.size * zero_weight_proportion):
    print('This matrix is already sparser than requested')
    return weight_array

  idx = np.random.choice(
      np.arange(weight_array.size),
      replace=False,
      size=int((weight_array.size * zero_weight_proportion) -
               num_zeros_initial))
  weight_array[np.unravel_index(idx, weight_array.shape)] = 0
  return weight_array


def cutoff_small_weights_in_rows(matrix: np.ndarray,
                                 non_zeros_per_row_limit: int) -> np.ndarray:
  """Cuts off smallest weights of the weight matrix.

  So that number of non-zeros per row does not exceed non_zeros_per_row_limit.

  Args:
    matrix: the weight matrix, a 2d square numpy array.
    non_zeros_per_row_limit: limit of non-zeros to be retained per row.

  Returns:
    matrix_transformed: the transformed weight matrix.
  """
  return CutOffSmallWeightsInRows(non_zeros_per_row_limit).apply_transform(
      matrix)


def transform_weight(weight_matrix: np.ndarray, distance_matrix: np.ndarray,
                     params: ml_collections.ConfigDict) -> np.ndarray:
  """Transforms the weight matrix with the following step.

  1. Scale the connection weights with the distances between connected neurons.
  2. Only keep largest weights on each row.
  3. If signaled, a random weight matrix is generated to replace the original
     while retaining the size and sparsity of the original.
  4. Scale the weight matrix to [0, 1] assuming sparsity > 0.
  5. Convert a proportion of neurons to inhibitory (all elements in row made 0).

  Parameters to provide:
    distance_scaling_function - name of method to scale weight by distance.
      Values are 'none', 'linear', 'quadratic', 'exponential'.
    box_size - length of one side of the cubic brain region from which
      the weight and distance matrices were extracted.
    num_cutoff - number of non zeros to keep on each row of the weight matrix,
      the rest are made zero. Make num_cutoff arbitrarily large to keep all.
    random_weights - boolean, whether to replace the original weight matrix
      with a random matrix of the same size and sparsity with the original.
    kde_random_weights - boolean. If random_weights is True, this parameter
      decides whether the random weights are generated with the same
      distribution of those in the original weight matrix.
    inhibitory_neurons_proportion - proportion of rows are made negative.

  Args:
    weight_matrix: 2D square numpy array.
    distance_matrix: 2D square numpy array of the same shape of weight_matrix.
    params: ConfigDict of parameters used for transformation, listed above.

  Returns:
    The transformed weight matrix.
  """

  transformation = setup_weight_transformation(params)

  if params.distance_scaling_function != 'none':
    return transformation.apply_transform(weight_matrix, distance_matrix)

  return transformation.apply_transform(weight_matrix)


def setup_weight_transformation(params: ml_collections.ConfigDict) -> Chain:
  """Sets up the chain of transformations to transform the weight matrix.

  The chain might include:
  1. Scale the connection weights with the distances between connected neurons.
  2. Only keep largest weights on each row.
  3. If signaled, a random weight matrix is generated to replace the original
     while retaining the size and sparsity of the original.
  4. Scale the weight matrix to [0, 1] assuming sparsity > 0.
  5. Convert a proportion of neurons to inhibitory (all elements in row made 0).

  Args:
    params: ConfigDict of parameters used for transformation, including:
      distance_scaling_function - name of method to scale weight by distance.
      Values are 'none', 'linear', 'quadratic', 'exponential'. box_size - length
      of one side of the cubic brain region from which the weight and distance
      matrices were extracted. num_cutoff - number of non zeros to keep on each
      row of the weight matrix, the rest are made zero. Make num_cutoff
      arbitrarily large to keep all. random_weights - boolean, whether to
      replace the original weight matrix with a random matrix of the same size
      and sparsity with the original. kde_random_weights - boolean. If
      random_weights is True, this parameter decides whether the random weights
      are generated with the same distribution of those in the original weight
      matrix. inhibitory_neurons_proportion - proportion of rows are made
      negative.

  Returns:
    The chained transformations.
  """
  transformation = Chain()

  # scale connection weight by distance between connected neurons:
  if params.distance_scaling_function != 'none':
    transformation.append(
        ScaleWeightByDistance(params.distance_scaling_function,
                              params.box_size))

  # only keep largest weights on each row:
  transformation.append(CutOffSmallWeightsInRows(params.num_cutoff))

  # if random weights should be used
  if params.random_weights:
    # if kernel-density estimate is used to randomize non-zero elements:
    if params.kde_random_weights:
      transformation.append(RandomizeNonZeroWeightKde())
    else:
      transformation.append(RandomizeNonZeroWeight())

  # scale weights to [0, 1] range:
  transformation.append(ScaleToZeroOne())

  # convert neurons to inhibitory:
  if params.inhibitory_neurons_proportion > 0:
    transformation.append(
        FlipSignEntireRows(params.inhibitory_neurons_proportion))

  return transformation


def chain_weight_matrices(matrices: List[np.ndarray]) -> List[np.ndarray]:
  """Chains the matrices in the list.

  The purpose is to trim the matrices so their multiplication can be done with
  their order in the chain.

  Suppose we have two matrices in the list, matrix A represents the connections
  between two sets of neurons Set_1 and Set_2 (the two sets might fully or
  partially overlap, or separate completely), and matrix B represents the
  connections between Set_3 and Set_4. Ideally, the realistic representation of
  the actual brain is to find the common subset of Set_2 and Set_3 and trim both
  A and B according to this subset. For simplicity, we assume either Set_2 is a
  subset of Set_3 or vice versa depending on their sizes.

  In practice, we often have matrices from separate regions and the
  post-synaptic neurons of the preceding regions might not overlap with the
  pre-synaptic neurons of the succeeding regions. In this case, we don't have a
  good way to tell which neurons are important to keep. For simplicity, we keep
  neurons that appear first and discard those that appear last in the matrix.

  Therefore, for each pair of adjacent matrices, the number of columns of the
  preceding  matrix is compared to the number of rows of the succeeding matrix,
  and the larger is trimmed down to be equal to the smaller, and in this
  process, the neurons that are trimmed appear last in the original matrix.

  Args:
    matrices: The list of matrices to be chained.

  Returns:
    The list of chained matrices.
  """
  num_matrices = len(matrices)
  if num_matrices < 2:
    return matrices

  new_matrices = []
  for i in range(num_matrices - 1):
    # compare num_columns of the preceding to num_rows of the succeeding:
    smaller_num_neurons = min(matrices[i].shape[1], matrices[i + 1].shape[0])
    # append the new matrix at position i to the new list:
    new_matrices.append(matrices[i][:, :smaller_num_neurons])
    # update the matrix at position i+1 in the old list:
    matrices[i + 1] = matrices[i + 1][:smaller_num_neurons, :]
  # append the last matrix to the new list:
  new_matrices.append(matrices[num_matrices - 1])
  return new_matrices


def resize_weight_matrices(
    reservoir_weights: List[np.ndarray],
    reservoirs_num_neurons: Tuple[int, ...],
) -> List[np.ndarray]:
  """Resizes the weight matrices to the target numbers of neurons.

  Args:
    reservoir_weights: The weight matrices to be transformed.
    reservoirs_num_neurons: The target numbers of neurons of the reservoirs. The
      number of elements must be the same as that of `reservoir_weights`.

  Returns:
    The transformed weight matrices.
  """
  # Check length matching of `reservoir_weights` and `reservoirs_num_neurons`
  if len(reservoir_weights) != len(reservoirs_num_neurons):
    raise ValueError(
        f'`reservoirs_num_neurons` has {len(reservoirs_num_neurons)} elements '
        f'but `reservoir_weights` has {len(reservoir_weights)} elements.')

  for i, num_neurons in enumerate(reservoirs_num_neurons):
    resize_rows = ResizeNumRows(num_neurons)
    reservoir_weights[i] = resize_rows.apply_transform(reservoir_weights[i])
    resize_columns = ResizeNumColumns(num_neurons)
    reservoir_weights[i] = resize_columns.apply_transform(reservoir_weights[i])

  return reservoir_weights
