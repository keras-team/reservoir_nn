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

"""Keras-like layers for custom reservoirs."""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from reservoir_nn.keras import rewiring
from reservoir_nn.local_learning import learning
from reservoir_nn.typing import types
from scipy import sparse
import tensorflow as tf


def _get_coo_indices_and_values(weight: Union[np.ndarray, sparse.spmatrix],
                                matrix_power: int,
                                order: str) -> Tuple[np.ndarray, np.ndarray]:
  """Returns the coordinate format (COO) representation of weight**matrix_power.

  See Returns for the definition of COO representation.

  Args:
    weight: the 2-D array to be sparsified.
    matrix_power: number of integer powers to take. Only defined for >=1.
    order: 'F' or 'C'. 'F' for fortran ordering, sort sparse_indices by the last
      dimension. 'C' for C ordering, sort sparse_indices by the first dimension.
      In 2-d, 'F' is column major ordering and 'C' is row major ordering.
      tf.sparse.sparse_dense_matmul requires the sparse_indices to be ordered.

  Returns:
    sparse_indices: integer array of shape (M, 2), where M is the number of
      non-zero elements in `weight` and n is the number of dimensions of the
      array. Here n == 2 and each element is (row_index, col_index)
      of the non-zero elements.
    sparse_values: matrix value at sparse_indices.
  """
  coo = sparse.coo_matrix(weight)

  product = coo
  for _ in range(matrix_power - 1):
    product = product.dot(coo)

  product = product.tocoo()

  # Build indices with shape (2, M).
  indices = np.array([product.row, product.col], dtype=np.intp)

  if order == 'F':  # column_major
    # Lexsort sort by last index (columns) first.
    arg = np.lexsort(indices)
  elif order == 'C':  # row_major
    # Reverse the keys, such that we sort by rows first.
    arg = np.lexsort(indices[::-1])

  # Build sorted arrays via permutation.
  sparse_indices = indices.T[arg]
  sparse_values = product.data[arg]

  return sparse_indices, sparse_values


class DenseReservoir(tf.keras.layers.Layer):
  """Dense reservoir layer.

  Usage:
    ```
    weight = some_get_weight_function()
    model = tf.keras.Sequential([
      ...,
      DenseReservoir(weight),
      ...
    ])
    ```

  Attributes:
    weight: Fixed weight matrix.
    activation: The activation function, for example tf.nn.relu.
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_regularizer: Regularizer function applied to the kernel.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    trainable_reservoir: Whether the reservoir is trainable.
    name: Optional name for the layer.
    recurrence_degree: number of recurrent steps inside the layer. This
      corresponds to taking the (n+1)-th power of the `weight`, which only works
      if the reservoir is square.
    activation_within_recurrence: Whether to include activation and bias between
      recurrence steps.
    kernel_local_learning: local learning rules to apply to kernel, options
      include ('none', 'hebbian', 'oja', 'contrastive_hebbian')
    kernel_local_learning_params: parameters for the local learning rules.
      (e.g., {'eta':0.1, 'gamma':0.01, 'expected':y_true, 'max_iter':10}).
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      weight: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
      activation: Optional[types.Activation] = None,
      use_bias: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      bias_initializer: types.Initializer = 'zeros',
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      trainable_reservoir: bool = False,
      name: Optional[str] = None,
      reservoir_shape: Optional[Tuple[int, ...]] = None,
      recurrence_degree: int = 0,
      activation_within_recurrence: bool = False,
      kernel_local_learning: str = 'none',
      kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      recurrent_kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      **kwargs: Any,
  ):
    """Initializes the dense layer."""
    super().__init__(
        activity_regularizer=activity_regularizer, name=name, **kwargs)

    self.kernel = None
    self.bias = None
    self.use_bias = use_bias
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.activation = tf.keras.activations.get(activation)
    self.reservoir_shape = reservoir_shape
    self.recurrence_degree = recurrence_degree
    self.activation_within_recurrence = activation_within_recurrence
    self.trainable_reservoir = trainable_reservoir
    self.kernel_local_learning = kernel_local_learning
    self.kernel_local_learning_params = kernel_local_learning_params or {}

    # When initialized from model checkpoints `weight` will be None. Kernel will
    #   be restored by Keras.
    if weight is not None:

      # compatibility with SparseReservoir
      if isinstance(weight, sparse.spmatrix):
        weight = weight.toarray()

      # Only allow numpy arrays for the initial reservoir weight.
      if not isinstance(weight, np.ndarray):
        raise TypeError(
            f'Only accept ndarray or spmatrix objects. Got {type(weight)}')

      self.reservoir_shape = weight.shape
      self.kernel = self.add_weight(
          name='kernel',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir,
      )

      if use_bias:
        self.bias = self.add_weight(
            'bias',
            shape=[self.units],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=tf.float32,
            trainable=True)

      if self.kernel_local_learning != 'none' or recurrence_degree > 0:
        if len(self.reservoir_shape
              ) != 2 or self.reservoir_shape[0] != self.reservoir_shape[1]:
          raise ValueError(
              f'In a recurrent layer the `weight` must be a square matrix. '
              f'Got shape {self.reservoir_shape}.')

  @property
  def units(self):
    return self.reservoir_shape[-1]

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    if self.kernel is None:
      raise RuntimeError('The kernel of the DenseReservoir layer is not '
                         'initialized. Call DenseReservoir with a weight '
                         'matrix to initialize it.')
    matrices = [self.kernel] * (self.recurrence_degree + 1)

    rank = inputs.shape.rank
    if rank < 2:
      raise ValueError(f'Input tensor must be at least 2D, but rank={rank}.')

    outputs = inputs
    for i, kernel in enumerate(matrices):
      outputs = tf.tensordot(outputs, kernel, [[rank - 1], [0]])

      if self.activation_within_recurrence or i == len(matrices) - 1:
        if self.bias is not None:
          outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
          outputs = self.activation(outputs)

    self.kernel = learning.LocalLearning(
        name='kernel_local_learning_layer',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=outputs,
            weight=self.kernel)

    return outputs

  def get_config(self):
    config = super().get_config()
    config.update({
        'weight':
            None,
        'activation':
            tf.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'reservoir_shape':
            self.reservoir_shape,
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        'bias_constraint':
            tf.keras.constraints.serialize(self.bias_constraint),
        'trainable_reservoir':
            self.trainable_reservoir,
        'recurrence_degree':
            self.recurrence_degree,
        'activation_within_recurrence':
            self.activation_within_recurrence,
        'kernel_local_learning':
            self.kernel_local_learning,
    })
    return config


class SparseReservoir(rewiring.SparseMutableInterface, tf.keras.layers.Layer):
  """Sparse reservoir layer.

  Usage: ```
    weight = some_get_weight_function()
    model = tf.keras.Sequential([
      ...,
      SparseReservoir(weight),
      ...
    ])
    ```

  Attributes:
    weight: The weight matrix.
    activation: The activation function, for example tf.nn.relu.
    use_bias: Boolean. Whether the layer uses a bias vector.
    kernel_regularizer: Regularizer function applied to the kernel.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    trainable_reservoir: Whether the reservoir is trainable.
    name: Optional name for the layer.
    recurrence_degree: number of recurrent steps inside the layer. This
      corresponds to taking the (n+1)-th power of the `weight`, which only works
      if the reservoir is square.
    activation_within_recurrence: Whether to include activation and bias between
      recurrence steps.
    kernel_local_learning: local learning rules to apply to kernel, options
      include ('none', 'hebbian', 'oja', 'contrastive_hebbian')
    kernel_local_learning_params: parameters for the local learning rules.
      (e.g., {'eta':0.1, 'gamma':0.01, 'expected':y_true, 'max_iter':10}).
    recurrent_kernel_local_learning_params: placeholder for learning parameters
      for recurrence.
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      weight: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
      activation: Optional[types.Activation] = None,
      use_bias: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      bias_initializer: types.Initializer = 'zeros',
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      trainable_reservoir: bool = False,
      name: Optional[str] = None,
      reservoir_shape: Optional[Tuple[int, ...]] = None,
      recurrence_degree: int = 0,
      activation_within_recurrence: bool = False,
      kernel_local_learning: str = 'none',
      kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      recurrent_kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      **kwargs: Any,
  ):
    """Initializes the sparse layer."""
    super().__init__(
        activity_regularizer=activity_regularizer, name=name, **kwargs)

    self.kernel = None
    self.bias = None
    self.use_bias = use_bias
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.activation = tf.keras.activations.get(activation)
    self.reservoir_shape = reservoir_shape
    self.recurrence_degree = recurrence_degree
    self.activation_within_recurrence = activation_within_recurrence
    self.trainable_reservoir = trainable_reservoir
    self.kernel_local_learning = kernel_local_learning
    self.kernel_local_learning_params = kernel_local_learning_params or {}

    # When initialized from model checkpoints `weight` will be None. Kernel will
    #   be restored by Keras.
    if weight is not None:
      # Backward compatible to ndarrays for the initial reservoir weight.
      if isinstance(weight, np.ndarray):
        weight = sparse.coo_matrix(weight)

      if not isinstance(weight, sparse.spmatrix):
        raise TypeError(
            f'Only accept ndarray or spmatrix objects. Got {type(weight)}')

      self.reservoir_shape = weight.shape

      # optimization
      if self._enable_expanded_weight_optimization():
        sparse_indices, sparse_values = _get_coo_indices_and_values(
            weight, self.recurrence_degree + 1, order='F')
      else:
        sparse_indices, sparse_values = _get_coo_indices_and_values(
            weight, 1, order='F')

      self.kernel = self.add_weight(
          name='kernel',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir,
      )
      # Indices are non-trainable, but we save them as weight to simplify
      # checkpoints and restoring.
      self.sparse_indices = self.add_weight(
          name='indices',
          shape=sparse_indices.shape,
          dtype=tf.int64,
          initializer=lambda shape, dtype: sparse_indices,
          trainable=False)

      if use_bias:
        self.bias = self.add_weight(
            'bias',
            shape=[self.units],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=tf.float32,
            trainable=True)

    if self.kernel_local_learning != 'none' or recurrence_degree > 0:
      if len(self.reservoir_shape
            ) != 2 or self.reservoir_shape[0] != self.reservoir_shape[1]:
        raise ValueError(
            f'In a recurrent layer the `weight` must be a square matrix. '
            f'Got shape {self.reservoir_shape}.')

  def _enable_expanded_weight_optimization(self):
    return (not self.trainable_reservoir and self.recurrence_degree > 0 and
            not self.activation_within_recurrence and
            len(self.reservoir_shape) == 2 and
            self.reservoir_shape[0] == self.reservoir_shape[1])

  @property
  def units(self):
    return self.reservoir_shape[-1]

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    if self.kernel is None:
      raise RuntimeError('The kernel of the SparseReservoir layer is not '
                         'initialized. Call SparseReservoir with a weight '
                         'matrix to initialize it.')

    sp_b = tf.SparseTensor(self.sparse_indices, self.kernel,
                           self.reservoir_shape)

    if self._enable_expanded_weight_optimization():
      matrices = [sp_b]
    else:
      matrices = [sp_b] * (self.recurrence_degree + 1)

    rank = inputs.shape.rank
    if rank < 2:
      raise ValueError(f'Input tensor must be at least 2D, but rank={rank}.')

    outputs = inputs

    # Pack the batch dimension because sparse_dense_matmul only allows 2D
    # matrices.
    outputs = tf.reshape(outputs, (-1, inputs.shape[-1]))

    for i, kernel in enumerate(matrices):

      outputs = tf.sparse.sparse_dense_matmul(
          outputs,
          kernel,  # Prefers kernel to be in column-major order.
          adjoint_a=False,
          adjoint_b=False)

      if self.activation_within_recurrence or i == len(matrices) - 1:
        if self.bias is not None:
          outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
          outputs = self.activation(outputs)

    # Unpack the batch dimension
    outputs = tf.reshape(
        outputs,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])

    self.kernel = learning.LocalLearning(
        name='kernel_local_learning_layer',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=outputs,
            weight=self.kernel)

    return outputs

  def get_config(self):
    config = super().get_config()
    config.update({
        'weight':
            None,
        'activation':
            tf.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'reservoir_shape':
            self.reservoir_shape,
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        'bias_constraint':
            tf.keras.constraints.serialize(self.bias_constraint),
        'trainable_reservoir':
            self.trainable_reservoir,
        'recurrence_degree':
            self.recurrence_degree,
        'activation_within_recurrence':
            self.activation_within_recurrence,
        'kernel_local_learning':
            self.kernel_local_learning,
    })
    return config

  def get_reservoir_shape(self):
    return self.reservoir_shape

  def get_sparse_tensors(self):
    sparse_indices = self.sparse_indices.value()
    sparse_values = self.kernel.value()
    sparse_ages = tf.zeros(tf.shape(sparse_values), tf.int64)
    return sparse_indices, sparse_values, sparse_ages

  def assign_sparse_tensors(self, sparse_indices: tf.Tensor,
                            sparse_values: tf.Tensor, sparse_ages: tf.Tensor):
    del sparse_ages
    self.sparse_indices.assign(sparse_indices)
    self.kernel.assign(sparse_values)


class Conv2DReservoir(tf.keras.layers.Layer):
  """Conv2D reservoir layer.

  Usage:
    ```
    weight = some_get_weight_function()
    model = tf.keras.Sequential([
      ...,
      Conv2DReservoir(weight),
      ...
    ])
    ```

  Attributes:
    weight: Weight matrix.
    filters: A tuple of 2 integers specifying the number of input and output
      filters in the convolution.
    kernel_size: A tuple of 2 integers specifying the height and width of the 2D
      convolution window.
    activation: The activation function, for example tf.nn.relu.
    use_bias: Boolean, whether the layer uses a bias vector.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    trainable_reservoir: Whether the reservoir is trainable.
    name: Optional name for the layer.
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      weight: Optional[np.ndarray] = None,
      filters: Tuple[int, int] = (1, 1),
      kernel_size: Tuple[int, int] = (3, 3),
      activation: Optional[types.Activation] = None,
      use_bias: bool = False,
      bias_initializer: types.Initializer = 'zeros',
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      trainable_reservoir: bool = False,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Initializes the conv2D layer."""
    super().__init__(
        activity_regularizer=activity_regularizer, name=name, **kwargs)

    self.kernel = None
    self.bias = None
    self.use_bias = use_bias
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.activation = tf.keras.activations.get(activation)

    # Compute matrix size
    num_weights = kernel_size[0] * kernel_size[1] * filters[0] * filters[1]
    reservoir_size = int(np.sqrt(num_weights))

    # Reshape weight
    weight = weight[:reservoir_size, :reservoir_size]
    weight = np.reshape(
        weight, (kernel_size[0], kernel_size[1], filters[0], filters[1]))

    # When initialized from model checkpoints `weight` will be None. Kernel will
    #   be restored by Keras.
    if weight is not None:
      self.kernel = tf.Variable(
          weight,
          dtype=tf.float32,
          trainable=trainable_reservoir,
          name='kernel')
      if use_bias:
        self.bias = self.add_weight(
            'bias',
            shape=[self.units],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=tf.float32,
            trainable=True)

  @property
  def units(self):
    num_units = self.kernel.shape
    print('num_units', num_units)
    return num_units[0] * num_units[1] * num_units[3]

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls the conv2D layer.

    Args:
      inputs: The input data with shape `[batch, height, width, color]`.

    Returns:
      The output of the 2D convolution.

    Raises:
      RuntimeError: Reservoir not intitialized.
    """
    if self.kernel is None:
      raise RuntimeError('The kernel of the Conv2DReservoir layer is not '
                         'initialized. Call Conv2DReservoir with a weight '
                         'matrix to initialize it.')

    rank = inputs.shape.rank
    if rank == 4 or rank is None:
      outputs = tf.keras.backend.conv2d(inputs, self.kernel)
    elif rank < 4:
      raise ValueError(f'Input tensor must be at least 4D, but rank={rank}.')

    if self.bias is not None:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def get_config(self):
    config = super().get_config()
    config.update({
        'weight':
            None,
        'activation':
            tf.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        'bias_constraint':
            tf.keras.constraints.serialize(self.bias_constraint),
    })
    return config


def _get_partial_input(indices: List[int]) -> tf.keras.layers.Lambda:
  return tf.keras.layers.Lambda(lambda x: tf.gather(x, indices, axis=-1))


def reservoir_kernel_initializer(weight: np.ndarray):
  return lambda shape, dtype: tf.cast(tf.ensure_shape(weight, shape), dtype)


class PerNeuronSparseReservoir(tf.keras.layers.Layer):
  """Sparsely-connected reservoir layer, one neuron per layer."""

  def __init__(
      self,
      weight: np.ndarray,
      activation: types.Activation = tf.keras.activations.relu,
      use_bias: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Builds a sparsely-connected reservoir layer.

    For each neuron of the reservoir, all the channels that go to the neuron are
    identified (weight is non-zero). Then a layer is built to be composed of
    these channels. The output layer is a concatenation of these individual
    layers. Note, here a "channel" has the meaning of a "path" that connects one
    neuron to another.

    Args:
      weight: The connection weight between neurons of the reservoir.
      activation: Activation function used for the individual reservoir layers.
      use_bias: Whether the individual reservoir layers use a bias vector.
      kernel_regularizer: Regularizer function that is applied to the kernels of
        the individual reservoir layers.
      name: The name of the layer.
      **kwargs: Other keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    num_neurons = weight.shape[0]
    self._partial_input_layers = []
    self._single_output_layers = []
    all_row_indices = np.arange(num_neurons)
    for i, weight_column in enumerate(weight.T):
      non_zero_row_indices = all_row_indices[weight_column != 0]
      non_zero_weight = np.expand_dims(
          weight_column[weight_column != 0], axis=1)

      self._partial_input_layers.append(
          _get_partial_input(non_zero_row_indices))
      self._single_output_layers.append(
          tf.keras.layers.Dense(
              1,
              activation=activation,
              use_bias=use_bias,
              kernel_initializer=reservoir_kernel_initializer(non_zero_weight),
              kernel_regularizer=kernel_regularizer,
              name=f'neuron_{i}',
          ))
    self._concatenate_layer = tf.keras.layers.Concatenate()

  def call(self, inputs):
    all_outputs = []
    for partial_layer, output_layer in zip(self._partial_input_layers,
                                           self._single_output_layers):
      x = partial_layer(inputs)
      x = output_layer(x)
      all_outputs.append(x)
    return self._concatenate_layer(all_outputs)


def _get_single_input_channel(i: int) -> tf.keras.layers.Lambda:
  return tf.keras.layers.Lambda(lambda x: x[..., i:i + 1])


class SelectiveSensor(tf.keras.layers.Layer):
  """Sensor-like layer for connecting Input to Reservoir."""

  def __init__(
      self,
      num_sensors_per_channel: int,
      kernel_initializer: types.Initializer = 'glorot_normal',
      activation: types.Activation = tf.keras.activations.relu,
      use_bias: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Builds a sensor-like layer.

    Biologically, a sensor is a neuron that takes input from the environment and
    send it to the downstream processing units. Each sensor might not take the
    whole input but only a partial input. The whole input is taken by the full
    set of sensors.

    This assumes that the original input has the channel dimension along the
    last axis with shape of (..., num_channels)

    Each input channel is connected to a subset of sensory neurons and different
    input channels are connected to different non-overlapping subsets. For each
    input channel, a fully connected layer is built to be composed of all the
    paths that connect every pixel of the channel to every neuron of the
    corresponding subset of sensors. The output layer is a concatenation of
    these individual layers.

    Args:
      num_sensors_per_channel: The number of sensory neurons that are connected
        to one channel.
      kernel_initializer: The initializer function for the kernels.
      activation: Activation function used for the individual reservoir layers.
      use_bias: Whether the individual reservoir layers use a bias vector.
      kernel_regularizer: Regularizer function that is applied to the kernels of
        the individual reservoir layers.
      name: The name of the layer.
      **kwargs: Other keyword arguments.
    """
    super().__init__(name=name, **kwargs)

    self.num_sensors_per_channel = num_sensors_per_channel
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

  def build(self, input_shape):
    self._input_shape = input_shape[1:-1]
    self._num_sensors = input_shape[-1] * self.num_sensors_per_channel
    self.kernel = self.add_weight(
        name='selective_sensor_kernel',
        dtype=tf.float32,
        regularizer=self.kernel_regularizer,
        initializer=self.kernel_initializer,
        # Each row of the kernel is the weights for the sensor neurons
        # assignment for that channel.
        shape=(input_shape[-1], self.num_sensors_per_channel),
    )
    self._reshape_output = tf.keras.layers.Reshape(
        (*self._input_shape, self._num_sensors))

  def call(self, inputs):
    outputs = tf.einsum('...i,ij->...ij', inputs, self.kernel)
    return self._reshape_output(outputs)

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_sensors_per_channel':
            self.num_sensors_per_channel,
        'activation':
            tf.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
        'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
    })
    return config


class SparseSensor(tf.keras.layers.Layer):
  """SparseReservoir Sensor-like layer for connecting Input to the Reservoir.

  First, the input is reshaped into a shape of (indices, num_input_channels):
  [[idx_0_channel_0, idx_0_channel_1,...]
   [idx_1_channel_0, idx_1_channel_1,...]
   .....................................
   [idx_N_channel_0, idx_N_channel_1....]]

  Then it is passed to a SparseReservoir. In this scheme, each channel is sent
  to a subset of neurons and the topology of the connectivity is defined by the
  weight matrix of a reservoir. Note, the input weight matrix must have the
  number of rows the same as `num_input_channels` and the number of columns the
  same as `num_sensors`.
  """

  def __init__(
      self,
      num_input_channels: int,
      num_sensors: int,
      weight: np.ndarray,
      activation: types.Activation = tf.keras.activations.relu,
      use_bias: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    """Builds a SparseReservoir sensor-like layer.

    Args:
      num_input_channels: The number of channels in the input data.
      num_sensors: The number of sensory neurons.
      weight: The weight matrix from a reservoir.
      activation: Activation function used for the SparseReservoir layer.
      use_bias: Whether the individual reservoir layers use a bias vector.
      kernel_regularizer: Regularizer function for the SparseReservoir kernel.
      name: The name of the layer.
      **kwargs: Other keyword arguments.
    """
    super().__init__(name=name, **kwargs)
    self.num_input_channels = num_input_channels
    self.num_sensors = num_sensors
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.weight_shape = weight.shape[0]

    self._reshape = tf.keras.layers.Reshape((-1, num_input_channels))

    # Chec if the weight matrix has shape of (num_input_channels, num_sensors)
    if weight.shape[0] != num_input_channels or weight.shape[1] != num_sensors:
      raise ValueError(
          f'The `weight` matrix is expected to have a shape of '
          f'({num_input_channels}, {num_sensors}), but got {weight.shape}.')

    self._sparse_reservoir = SparseReservoir(
        weight=weight,
        activation=activation,
        trainable_reservoir=True,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
    )

  def call(self, inputs):
    return self._sparse_reservoir(self._reshape(inputs))

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_input_channels':
            self.num_input_channels,
        'num_sensors':
            self.num_sensors,
        'weight_shape':
            self.weight_shape,
        'activation':
            tf.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
    })
    return config


class RecurrentDenseReservoir(DenseReservoir):
  """RNN reservoir layer.

  Usage:
    ```
    weight = some_get_weight_function()
    model = tf.keras.Sequential([
      ...,
      RecurrentDenseReservoir(weight),
      ...
    ])
    ```

  Attributes:
    weight: The weight matrix.
    activation: The activation function, for example tf.nn.relu.
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_regularizer: Regularizer function applied to the kernel.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    trainable_reservoir: Whether the reservoir is trainable.
    name: Optional name for the layer.
    recurrence_degree: number of recurrent steps inside the layer. This
      corresponds to taking the (n+1)-th power of the `weight`, which only works
      if the reservoir is square.
    activation_within_recurrence: Whether to include activation and bias between
      recurrence steps.
    kernel_local_learning: local learning rules to apply to kernel, options
      include ('none', 'hebbian', 'oja', 'contrastive_hebbian').
    recurrent_kernel_local_learning: local learning rules in recurrence.
    kernel_local_learning_params: parameters for the local learning rules.
      (e.g., {'eta':0.1, 'gamma':0.01, 'expected':y_true, 'max_iter':10}).
    recurrent_kernel_local_learning_params: learning parameters for recurrence.
    keep_memory: whether to keep prior_states during learning or inference.
    max_batch_size: the maximum batch size to keep the cell states.
    state_discount: the discounting factor to regularize prior states.
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      weight: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
      activation: Optional[types.Activation] = None,
      use_bias: bool = True,
      kernel_regularizer: Optional[types.Regularizer] = None,
      bias_initializer: types.Initializer = 'zeros',
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      trainable_reservoir: bool = False,
      name: Optional[str] = None,
      reservoir_shape: Optional[Tuple[int, ...]] = None,
      recurrence_degree: int = 0,
      activation_within_recurrence: bool = True,
      kernel_local_learning: str = 'none',
      recurrent_kernel_local_learning: str = 'none',
      kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      recurrent_kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      keep_memory: bool = False,
      max_batch_size: int = 256,
      state_discount: float = 1.0,
      **kwargs: Any,
  ):
    """Initializes the rnn layer."""
    super().__init__(
        weight=weight,
        activation=activation,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        trainable_reservoir=trainable_reservoir,
        name=name,
        reservoir_shape=reservoir_shape,
        recurrence_degree=recurrence_degree,
        activation_within_recurrence=activation_within_recurrence,
        kernel_local_learning=kernel_local_learning,
        kernel_local_learning_params=kernel_local_learning_params,
        **kwargs)

    self.recurrent_kernel_local_learning = recurrent_kernel_local_learning
    self.recurrent_kernel_local_learning_params = (
        recurrent_kernel_local_learning_params or {})
    self.keep_memory = keep_memory
    self.max_batch_size = max_batch_size
    self.state_discount = state_discount
    if weight is not None:

      # compatibility with SparseReservoir
      if isinstance(weight, sparse.spmatrix):
        weight = weight.toarray()

      # Only allow numpy arrays for the initial reservoir weight.
      if not isinstance(weight, np.ndarray):
        raise TypeError(
            f'Only accept ndarray or spmatrix objects. Got {type(weight)}')

      self.recurrent_kernel = self.add_weight(
          name='recurrent_kernel',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir,
      )
    else:
      self.recurrent_kernel = None

  def build(self, input_shape):

    super().build(input_shape)

    self.cell_states = self.add_weight(
        name='cell_states',
        initializer=tf.keras.initializers.Zeros(),
        shape=(self.max_batch_size, *input_shape[1:]),
        trainable=False,
        dtype=tf.float32)

  def get_config(self):
    config = super().get_config()
    config.update({
        'max_batch_size': self.max_batch_size,
        'state_discount': self.state_discount,
        'recurrent_kernel_local_learning': self.recurrent_kernel_local_learning,
        'keep_memory': self.keep_memory,
    })
    return config

  def call(self,
           inputs: tf.Tensor,
           prior_states: Optional[tf.Tensor] = None) -> tf.Tensor:
    """RNN forward function.

    Args:
      inputs: the input tensor into the model
      prior_states: the previous cell states of the recurrent layer

    Returns:
      outputs: the output tensor from the recurrent layer

    Raises:
      RuntimeError: if the kernel is not initialized
      RuntimeError: if the recurrent kernel is not initialized
      ValueError: if the input tensor has fewer than 2 dimensions
    """
    if self.kernel is None:
      raise RuntimeError('The reservoir of the RNN kernel layer is not '
                         'initialized. Call RNN with a weight matrix to '
                         'initialize it.')
    if self.recurrent_kernel is None:
      raise RuntimeError('The reservoir of the RNN recurrent kernel is not '
                         'initialized. Call RNN with a weight matrix to '
                         'initialize it.')

    rank = inputs.shape.rank
    if rank < 2:
      raise ValueError(f'Input tensor must be at least 2D, but rank={rank}.')

    # Use the first batch_size cell_states.
    batch_size = tf.shape(inputs)[0]

    if prior_states is not None:
      cell_states = prior_states
    else:
      # Have to constrain with sigmoid. Otherwise, the gradient might blow up.
      # (The state discounting is still kept here since it's an existing notion
      # in the field that might be useful in the future. No effects as of now.)
      cell_states = tf.math.sigmoid(
          self.state_discount *
          tf.gather(self.cell_states, tf.range(batch_size)))

    input_states = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
    self.kernel = learning.LocalLearning(
        name='kernel_local_learning_layer',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_states,
            weight=self.kernel,
        )

    for i in np.arange(self.recurrence_degree + 1):
      cell_states = tf.tensordot(cell_states, self.recurrent_kernel,
                                 [[rank - 1], [0]])
      outputs = cell_states + input_states

      if self.activation_within_recurrence or i == self.recurrence_degree:
        if self.bias is not None:
          outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
          outputs = self.activation(outputs)
      cell_states = outputs

      self.recurrent_kernel = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=cell_states,
              weight=self.recurrent_kernel,
          )
    if self.keep_memory:
      self.cell_states.batch_scatter_update(
          tf.IndexedSlices(
              cell_states,
              tf.range(batch_size),
              dense_shape=self.cell_states.shape))

    return outputs


class RecurrentSparseReservoir(SparseReservoir):
  """Sparse RNN reservoir layer.

  Usage: ```
    weight = some_get_weight_function()
    model = tf.keras.Sequential([
      ...,
      RecurrentSparseReservoir(weight),
      ...
    ])
    ```

  Attributes:
    weight: The weight matrix.
    activation: The activation function, for example tf.nn.relu.
    use_bias: Boolean. Whether the layer uses a bias vector.
    kernel_regularizer: Regularizer function applied to the kernel.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    trainable_reservoir: Whether the reservoir is trainable.
    name: Optional name for the layer.
    recurrence_degree: number of recurrent steps inside the layer. This
      corresponds to taking the (n+1)-th power of the `weight`, which only works
      if the reservoir is square.
    activation_within_recurrence: Whether to include activation and bias between
      recurrence steps.
    kernel_local_learning: local learning rules to apply to kernel, options
      include ('none', 'hebbian', 'oja', 'contrastive_hebbian').
    recurrent_kernel_local_learning: local learning rules in recurrence.
    kernel_local_learning_params: parameters for the local learning rules.
      (e.g., {'eta':0.1, 'gamma':0.01, 'expected':y_true, 'max_iter':10}).
    recurrent_kernel_local_learning_params: learning parameters for recurrence.
    keep_memory: whether to keep prior_states during learning or inference.
    max_batch_size: the maximum batch size to keep the cell states.
    state_discount: the discounting factor to regularize prior states.
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      weight: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
      activation: Optional[types.Activation] = None,
      use_bias: bool = True,
      kernel_regularizer: Optional[types.Regularizer] = None,
      bias_initializer: types.Initializer = 'zeros',
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      trainable_reservoir: bool = False,
      name: Optional[str] = None,
      reservoir_shape: Optional[Tuple[int, ...]] = None,
      recurrence_degree: int = 0,
      activation_within_recurrence: bool = True,
      kernel_local_learning: str = 'none',
      recurrent_kernel_local_learning: str = 'none',
      kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      recurrent_kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      keep_memory: bool = False,
      max_batch_size: int = 256,
      state_discount: float = 1.0,
      **kwargs: Any,
  ):
    """Initializes the rnn layer."""
    super().__init__(
        weight=weight,
        activation=activation,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        trainable_reservoir=trainable_reservoir,
        name=name,
        reservoir_shape=reservoir_shape,
        recurrence_degree=recurrence_degree,
        activation_within_recurrence=activation_within_recurrence,
        kernel_local_learning=kernel_local_learning,
        kernel_local_learning_params=kernel_local_learning_params,
        **kwargs)

    self.recurrent_kernel_local_learning = recurrent_kernel_local_learning
    self.recurrent_kernel_local_learning_params = (
        recurrent_kernel_local_learning_params or {})
    self.keep_memory = keep_memory
    self.max_batch_size = max_batch_size
    self.state_discount = state_discount

    if weight is not None:
      # Backward compatible to ndarrays for the initial reservoir weight.
      if isinstance(weight, np.ndarray):
        weight = sparse.coo_matrix(weight)

      if not isinstance(weight, sparse.spmatrix):
        raise TypeError(
            f'Only accept ndarray or spmatrix objects. Got {type(weight)}')

      self.reservoir_shape = weight.shape

      sparse_indices, sparse_values = _get_coo_indices_and_values(
          weight, 1, order='F')

      self.kernel = self.add_weight(
          name='kernel',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir,
      )
      self.recurrent_kernel = self.add_weight(
          name='recurrent_kernel',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir,
      )
      # Indices are non-trainable, but we save them as weight to simplify
      # checkpoints and restoring.
      self.sparse_indices = self.add_weight(
          name='indices',
          shape=sparse_indices.shape,
          dtype=tf.int64,
          initializer=lambda shape, dtype: sparse_indices,
          trainable=False)
    else:
      self.recurrent_kernel = None

  def build(self, input_shape):

    super().build(input_shape)

    self.cell_states = self.add_weight(
        name='cell_states',
        initializer=tf.keras.initializers.Zeros(),
        shape=(self.max_batch_size, *input_shape[1:]),
        trainable=False,
        dtype=tf.float32)

  def get_config(self):
    config = super().get_config()
    config.update({
        'max_batch_size': self.max_batch_size,
        'state_discount': self.state_discount,
        'recurrent_kernel_local_learning': self.recurrent_kernel_local_learning,
        'keep_memory': self.keep_memory,
    })
    return config

  def call(self,
           inputs: tf.Tensor,
           prior_states: Optional[tf.Tensor] = None) -> tf.Tensor:
    """RNN forward function.

    Args:
      inputs: the input tensor into the model
      prior_states: the previous cell states of the recurrent layer

    Returns:
      outputs: the output tensor from the recurrent layer

    Raises:
      RuntimeError: if the kernel is not initialized
      RuntimeError: if the recurrent kernel is not initialized
      ValueError: if the input tensor has fewer than 2 dimensions
    """

    if self.kernel is None:
      raise RuntimeError('The kernel of RecurrentSparseReservoir layer is not '
                         'initialized. Call RecurrentSparseReservoir with a '
                         'weight matrix to initialize it.')
    if self.recurrent_kernel is None:
      raise RuntimeError('The recurrent kernel of the RecurrentSparseReservoir '
                         'is not initialized. Call RecurrentSparseReservoir '
                         'with a weight matrix to initialize it.')

    rank = inputs.shape.rank
    if rank < 2:
      raise ValueError(f'Input tensor must be at least 2D, but rank={rank}.')

    sp_k = tf.SparseTensor(self.sparse_indices, self.kernel,
                           self.reservoir_shape)
    sp_r = tf.SparseTensor(self.sparse_indices, self.recurrent_kernel,
                           self.reservoir_shape)

    # Use the first batch_size cell_states.
    batch_size = tf.shape(inputs)[0]

    if prior_states is not None:
      cell_states = prior_states
    else:
      # Have to constrain with sigmoid. Otherwise, the gradient might blow up.
      # (The state discounting is still kept here since it's an existing notion
      # in the field that might be useful in the future. No effects as of now.)
      cell_states = tf.math.sigmoid(
          self.state_discount *
          tf.gather(self.cell_states, tf.range(batch_size)))
    cell_states = tf.reshape(cell_states, (-1, cell_states.shape[-1]))

    # Pack the batch dimension because sparse_dense_matmul only allows 2D
    # matrices.
    inputs_2d = tf.reshape(inputs, (-1, inputs.shape[-1]))
    input_states = tf.sparse.sparse_dense_matmul(
        inputs_2d,
        sp_k,  # Prefers kernel to be in column-major order.
        adjoint_a=False,
        adjoint_b=False)

    input_states_unpacked = tf.reshape(
        input_states,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])

    self.kernel = learning.LocalLearning(
        name='kernel_local_learning_layer',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_states_unpacked,
            weight=self.kernel,
        )

    for i in np.arange(self.recurrence_degree + 1):

      cell_states = tf.sparse.sparse_dense_matmul(
          cell_states,
          sp_r,  # in column-major order.
          adjoint_a=False,
          adjoint_b=False)
      outputs = cell_states + input_states

      if self.activation_within_recurrence or i == self.recurrence_degree:
        if self.bias is not None:
          outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
          outputs = self.activation(outputs)
      cell_states = outputs

      cell_states_unpacked = tf.reshape(
          cell_states, [i if i is not None else -1 for i in inputs.shape[:-1]] +
          [self.units])

      self.recurrent_kernel = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=cell_states_unpacked,
              weight=self.recurrent_kernel,
          )

    # Unpack the batch dimension
    if self.keep_memory:
      cell_states = tf.reshape(
          cell_states, [i if i is not None else -1 for i in inputs.shape[:-1]] +
          [self.units])
      self.cell_states.batch_scatter_update(
          tf.IndexedSlices(
              cell_states,
              tf.range(batch_size),
              dense_shape=self.cell_states.shape))

    outputs = tf.reshape(
        outputs,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])

    return outputs


class RecurrentSparseSensor(SparseSensor):
  """SparseReservoir RNN Sensor-like layer for connecting Input to Reservoir.

  First, the input is reshaped into a shape of (indices, num_input_channels):
  [[idx_0_channel_0, idx_0_channel_1,...]
   [idx_1_channel_0, idx_1_channel_1,...]
   .....................................
   [idx_N_channel_0, idx_N_channel_1....]]

  It is passed to a SparseReservoir RNN. In this scheme, each channel is sent
  to a subset of neurons and the topology of the connectivity is defined by the
  weight matrix of a reservoir. Note, the input weight matrix must have the
  number of rows the same as `num_input_channels` and the number of columns the
  same as `num_sensors`.
  """

  def __init__(self,
               num_input_channels: int,
               num_sensors: int,
               weight: Optional[types.WeightMatrix] = None,
               activation: types.Activation = tf.keras.activations.relu,
               use_bias: bool = False,
               kernel_regularizer: Optional[types.Regularizer] = None,
               bias_initializer: types.Initializer = 'zeros',
               bias_regularizer: Optional[types.Regularizer] = None,
               activity_regularizer: Optional[types.Regularizer] = None,
               bias_constraint: Optional[types.Constraint] = None,
               trainable_reservoir: bool = False,
               name: Optional[str] = None,
               reservoir_shape: Optional[Tuple[int, ...]] = None,
               recurrence_degree: int = 0,
               activation_within_recurrence: bool = False,
               rnnbase: Type[SparseReservoir] = RecurrentSparseReservoir,
               **kwargs: Any):
    """Builds a SparseReservoir sensor-like layer.

    Args:
      num_input_channels: The number of channels in the input data.
      num_sensors: The number of sensory neurons.
      weight: The weight matrix.
      activation: The activation function, for example tf.nn.relu.
      use_bias: Boolean. Whether the layer uses a bias vector.
      kernel_regularizer: Regularizer function applied to the kernel.
      bias_initializer: Initializer for the bias vector.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      bias_constraint: Constraint function applied to the bias vector.
      trainable_reservoir: Whether the reservoir is trainable.
      name: Optional name for the layer.
      reservoir_shape: the shape of the reservoir tensor
      recurrence_degree: number of recurrent steps inside the layer. This
        corresponds to taking the (n+1)-th power of the `weight`, which only
        works if the reservoir is square.
      activation_within_recurrence: Whether to include activation and bias
        between recurrence steps.
      rnnbase: the RNN layer to use in here.
      **kwargs: Other keyword arguments.
    """
    super().__init__(
        num_input_channels=num_input_channels,
        num_sensors=num_sensors,
        weight=weight,
        activation=activation,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        name=name,
        **kwargs)

    self._rnn_base_layer = rnnbase(
        weight=weight,
        activation=activation,
        use_bias=use_bias,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        trainable_reservoir=trainable_reservoir,
        reservoir_shape=reservoir_shape,
        recurrence_degree=recurrence_degree,
        activation_within_recurrence=activation_within_recurrence,
        **kwargs)

  def call(self, inputs, **kwargs):
    return self._rnn_base_layer(self._reshape(inputs), **kwargs)


class LSTMDenseReservoir(RecurrentDenseReservoir):
  """Long Short Term Memory (LSTM) reservoir layer.

  Usage:
    ```
    weight = some_get_weight_function()
    model = tf.keras.Sequential([
      ...,
      LSTMDenseReservoir(weight),
      ...
    ])
    ```

  Attributes:
    weight: The weight matrix.
    activation: The activation function, for example tf.nn.relu.
    use_bias_per_cell: tuple of Boolean, whether the layers use a bias vector.
    kernel_regularizer: Regularizer function applied to the kernel.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    trainable_reservoir_per_cell: Whether the several reservoirs are trainable.
    name: Optional name for the layer.
    recurrence_degree: number of recurrent steps inside the layer. This
      corresponds to taking the (n+1)-th power of the `weight`, which only works
      if the reservoir is square.
    activation_within_recurrence: Whether to include activation and bias between
      recurrence steps.
    kernel_local_learning: local learning rules to apply to kernel, options
      include ('none', 'hebbian', 'oja', 'contrastive_hebbian').
    recurrent_kernel_local_learning: local learning rules in recurrence.
    kernel_local_learning_params: parameters for the local learning rules.
      (e.g., {'eta':0.1, 'gamma':0.01, 'expected':y_true, 'max_iter':10}).
    recurrent_kernel_local_learning_params: learning parameters for recurrence.
    keep_memory: whether to keep prior_states during learning or inference.
    max_batch_size: the maximum batch size to keep the cell states.
    state_discount: the discounting factor to regularize prior states.
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      weight: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
      activation: Optional[types.Activation] = None,
      use_bias_per_cell: Tuple[bool, ...] = (True, True, True, True),
      trainable_reservoir_per_cell: Tuple[bool,
                                          ...] = (True, False, False, False),
      bias_initializer: types.Initializer = 'zeros',
      recurrence_degree: int = 0,
      activation_within_recurrence: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      name: Optional[str] = None,
      reservoir_shape: Optional[Tuple[int, ...]] = None,
      kernel_local_learning: str = 'none',
      recurrent_kernel_local_learning: str = 'none',
      kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      recurrent_kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      keep_memory: bool = False,
      max_batch_size: int = 256,
      state_discount: float = 1.0,
      **kwargs: Any,
  ):
    """Initializes the rnn layer."""
    super().__init__(
        weight=weight,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        name=name,
        reservoir_shape=reservoir_shape,
        recurrence_degree=recurrence_degree,
        activation_within_recurrence=activation_within_recurrence,
        kernel_local_learning=kernel_local_learning,
        recurrent_kernel_local_learning=recurrent_kernel_local_learning,
        kernel_local_learning_params=kernel_local_learning_params,
        recurrent_kernel_local_learning_params=recurrent_kernel_local_learning_params,
        keep_memory=keep_memory,
        max_batch_size=max_batch_size,
        state_discount=state_discount,
        **kwargs)

    self.use_bias_per_cell = use_bias_per_cell
    self.trainable_reservoir_per_cell = trainable_reservoir_per_cell

    if weight is not None:

      # compatibility with SparseReservoir
      if isinstance(weight, sparse.spmatrix):
        weight = weight.toarray()

      # Only allow numpy arrays for the initial reservoir weight.
      if not isinstance(weight, np.ndarray):
        raise TypeError(
            f'Only accept ndarray or spmatrix objects. Got {type(weight)}')

      self.reservoir_shape = weight.shape

      # there are four sets of parameters: f stands for "forget", i stands for
      # "input", o stands for "output", and c stands for "cell". They are for
      # gating purposes.
      self.w_f = self.add_weight(
          name='kernel_f',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[0],
      )
      self.w_i = self.add_weight(
          name='kernel_i',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[1],
      )
      self.w_o = self.add_weight(
          name='kernel_o',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[2],
      )
      self.w_c = self.add_weight(
          name='kernel_c',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[3],
      )

      self.u_f = self.add_weight(
          name='recurrent_kernel_f',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[0],
      )
      self.u_i = self.add_weight(
          name='recurrent_kernel_i',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[1],
      )
      self.u_o = self.add_weight(
          name='recurrent_kernel_o',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[2],
      )
      self.u_c = self.add_weight(
          name='recurrent_kernel_c',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=reservoir_kernel_initializer(weight),
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[3],
      )

      self.b_f = self.add_weight(
          'bias_f',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
      self.b_i = self.add_weight(
          'bias_i',
          shape=[self.units],
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
      self.b_o = self.add_weight(
          'bias_o',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
      self.b_c = self.add_weight(
          'bias_c',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
    else:
      self.w_f, self.w_i, self.w_o, self.w_c = None, None, None, None
      self.u_f, self.u_i, self.u_o, self.u_c = None, None, None, None
      self.b_f, self.b_i, self.b_o, self.b_c = None, None, None, None

  def build(self, input_shape):

    super().build(input_shape)

    self.hidden_states = self.add_weight(
        name='hidden_states',
        initializer=tf.keras.initializers.Zeros(),
        shape=(self.max_batch_size, *input_shape[1:]),
        trainable=False,
        dtype=tf.float32)
    self.cell_states = self.add_weight(
        name='cell_states',
        initializer=tf.keras.initializers.Zeros(),
        shape=(self.max_batch_size, *input_shape[1:]),
        trainable=False,
        dtype=tf.float32)

  def get_config(self):
    config = super().get_config()
    config.update({
        'trainable_reservoir_per_cell': self.trainable_reservoir_per_cell,
        'use_bias_per_cell': self.use_bias_per_cell,
    })
    return config

  def call(
      self,
      inputs: tf.Tensor,
      prior_states: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.Tensor:
    """LSTM forward function.

    Args:
      inputs: the input tensor into the model
      prior_states: the previous states of the recurrent layer

    Returns:
      outputs: the output tensor from the recurrent layer

    Raises:
      RuntimeError: if the kernel is not initialized
      RuntimeError: if the recurrent kernel is not initialized
      ValueError: if the input tensor has fewer than 2 dimensions
    """
    if (self.w_f is None or self.w_i is None or self.w_o is None or
        self.w_c is None):
      raise RuntimeError('The reservoir of the LSTM kernel layer is not '
                         'initialized. Call LSTM with a weight matrix to '
                         'initialize it.')
    if (self.u_f is None or self.u_i is None or self.u_o is None or
        self.u_c is None):
      raise RuntimeError('The reservoir of the LSTM recurrent kernel is not '
                         'initialized. Call LSTM with a weight matrix to '
                         'initialize it.')

    rank = inputs.shape.rank
    if rank < 2:
      raise ValueError(f'Input tensor must be at least 2D, but rank={rank}.')

    # Use the first batch_size cell_states.
    batch_size = tf.shape(inputs)[0]

    if prior_states is not None:
      hidden_states, cell_states = prior_states[0], prior_states[1]
    else:
      hidden_states = self.state_discount * tf.gather(self.hidden_states,
                                                      tf.range(batch_size))
      cell_states = self.state_discount * tf.gather(self.cell_states,
                                                    tf.range(batch_size))

    input_part_f = tf.tensordot(inputs, self.w_f, [[rank - 1], [0]])
    input_part_i = tf.tensordot(inputs, self.w_i, [[rank - 1], [0]])
    input_part_o = tf.tensordot(inputs, self.w_o, [[rank - 1], [0]])
    input_part_c = tf.tensordot(inputs, self.w_c, [[rank - 1], [0]])

    self.w_f = learning.LocalLearning(
        name='kernel_local_learning_layer_f',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_f,
            weight=self.w_f,
        )

    self.w_i = learning.LocalLearning(
        name='kernel_local_learning_layer_i',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_i,
            weight=self.w_i,
        )

    self.w_o = learning.LocalLearning(
        name='kernel_local_learning_layer_o',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_o,
            weight=self.w_o,
        )

    self.w_c = learning.LocalLearning(
        name='kernel_local_learning_layer_c',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_c,
            weight=self.w_c,
        )

    for _ in np.arange(self.recurrence_degree + 1):
      if hidden_states is not None:
        recurrent_part_f = tf.tensordot(hidden_states, self.u_f,
                                        [[rank - 1], [0]])
        recurrent_part_i = tf.tensordot(hidden_states, self.u_i,
                                        [[rank - 1], [0]])
        recurrent_part_o = tf.tensordot(hidden_states, self.u_o,
                                        [[rank - 1], [0]])
        recurrent_part_c = tf.tensordot(hidden_states, self.u_c,
                                        [[rank - 1], [0]])

        f_part = input_part_f + recurrent_part_f
        i_part = input_part_i + recurrent_part_i
        o_part = input_part_o + recurrent_part_o
        c_part = input_part_c + recurrent_part_c
      else:
        f_part = input_part_f
        i_part = input_part_i
        o_part = input_part_o
        c_part = input_part_c

      if self.use_bias_per_cell is not None:
        if self.use_bias_per_cell[0]:
          f_part = tf.nn.bias_add(f_part, self.b_f)
        if self.use_bias_per_cell[1]:
          i_part = tf.nn.bias_add(i_part, self.b_i)
        if self.use_bias_per_cell[2]:
          o_part = tf.nn.bias_add(o_part, self.b_o)
        if self.use_bias_per_cell[3]:
          c_part = tf.nn.bias_add(c_part, self.b_c)

      f_t = tf.keras.activations.sigmoid(f_part)
      i_t = tf.keras.activations.sigmoid(i_part)
      o_t = tf.keras.activations.sigmoid(o_part)
      c_t = tf.keras.activations.tanh(c_part)

      c_t = tf.math.multiply(f_t, cell_states) + tf.math.multiply(i_t, c_t)
      h_t = tf.math.multiply(o_t, tf.keras.activations.tanh(c_t))
      cell_states, hidden_states = c_t, h_t

      self.u_f = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_f',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states,
              weight=self.u_f,
          )

      self.u_i = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_i',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states,
              weight=self.u_i,
          )

      self.u_o = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_o',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states,
              weight=self.u_o,
          )

      self.u_c = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_c',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states,
              weight=self.u_c,
          )

    if self.keep_memory:
      self.hidden_states.batch_scatter_update(
          tf.IndexedSlices(
              hidden_states,
              tf.range(batch_size),
              dense_shape=self.hidden_states.shape))
      self.cell_states.batch_scatter_update(
          tf.IndexedSlices(
              cell_states,
              tf.range(batch_size),
              dense_shape=self.cell_states.shape))

    outputs = hidden_states
    return outputs


class LSTMSparseReservoir(RecurrentSparseReservoir):
  """Sparse Long Short Term Memory (LSTM) reservoir layer.

  Usage:
    ```
    weight = some_get_weight_function()
    model = tf.keras.Sequential([
      ...,
      LSTMSparseReservoir(weight),
      ...
    ])
    ```

  Attributes:
    weight: The weight matrix.
    activation: The activation function, for example tf.nn.relu.
    use_bias_per_cell: tuple of Boolean, whether the layers use a bias vector.
    kernel_regularizer: Regularizer function applied to the kernel.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    trainable_reservoir_per_cell: Whether the several reservoirs are trainable.
    name: Optional name for the layer.
    recurrence_degree: number of recurrent steps inside the layer. This
      corresponds to taking the (n+1)-th power of the `weight`, which only works
      if the reservoir is square.
    activation_within_recurrence: Whether to include activation and bias between
      recurrence steps.
    kernel_local_learning: local learning rules to apply to kernel, options
      include ('none', 'hebbian', 'oja', 'contrastive_hebbian').
    recurrent_kernel_local_learning: local learning rules in recurrence.
    kernel_local_learning_params: parameters for the local learning rules.
      (e.g., {'eta':0.1, 'gamma':0.01, 'expected':y_true, 'max_iter':10}).
    recurrent_kernel_local_learning_params: learning parameters for recurrence.
    keep_memory: whether to keep prior_states during learning or inference.
    max_batch_size: the maximum batch size to keep the cell states.
    state_discount: the discounting factor to regularize prior states.
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      weight: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
      activation: Optional[types.Activation] = None,
      use_bias_per_cell: Tuple[bool, ...] = (True, True, True, True),
      trainable_reservoir_per_cell: Tuple[bool,
                                          ...] = (True, False, False, False),
      bias_initializer: types.Initializer = 'zeros',
      recurrence_degree: int = 0,
      activation_within_recurrence: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      name: Optional[str] = None,
      reservoir_shape: Optional[Tuple[int, ...]] = None,
      kernel_local_learning: str = 'none',
      recurrent_kernel_local_learning: str = 'none',
      kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      recurrent_kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      keep_memory: bool = False,
      max_batch_size: int = 256,
      state_discount: float = 1.0,
      **kwargs: Any,
  ):
    """Initializes the rnn layer."""
    super().__init__(
        weight=weight,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        name=name,
        reservoir_shape=reservoir_shape,
        recurrence_degree=recurrence_degree,
        activation_within_recurrence=activation_within_recurrence,
        kernel_local_learning=kernel_local_learning,
        recurrent_kernel_local_learning=recurrent_kernel_local_learning,
        kernel_local_learning_params=kernel_local_learning_params,
        recurrent_kernel_local_learning_params=recurrent_kernel_local_learning_params,
        keep_memory=keep_memory,
        max_batch_size=max_batch_size,
        state_discount=state_discount,
        **kwargs)

    self.use_bias_per_cell = use_bias_per_cell
    self.trainable_reservoir_per_cell = trainable_reservoir_per_cell

    if weight is not None:

      # Backward compatible to ndarrays for the initial reservoir weight.
      if isinstance(weight, np.ndarray):
        weight = sparse.coo_matrix(weight)

      if not isinstance(weight, sparse.spmatrix):
        raise TypeError(
            f'Only accept ndarray or spmatrix objects. Got {type(weight)}')

      self.reservoir_shape = weight.shape

      sparse_indices, sparse_values = _get_coo_indices_and_values(
          weight, 1, order='F')

      # there are four sets of parameters: f stands for "forget", i stands for
      # "input", o stands for "output", and c stands for "cell". They are for
      # gating purposes.
      self.w_f = self.add_weight(
          name='kernel_f',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[0],
      )
      self.w_i = self.add_weight(
          name='kernel_i',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[1],
      )
      self.w_o = self.add_weight(
          name='kernel_o',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[2],
      )
      self.w_c = self.add_weight(
          name='kernel_c',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[3],
      )

      self.u_f = self.add_weight(
          name='recurrent_kernel_f',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[0],
      )
      self.u_i = self.add_weight(
          name='recurrent_kernel_i',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[1],
      )
      self.u_o = self.add_weight(
          name='recurrent_kernel_o',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[2],
      )
      self.u_c = self.add_weight(
          name='recurrent_kernel_c',
          shape=sparse_values.shape,
          dtype=tf.float32,
          initializer=lambda shape, dtype: sparse_values,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir_per_cell[3],
      )

      self.b_f = self.add_weight(
          'bias_f',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
      self.b_i = self.add_weight(
          'bias_i',
          shape=[self.units],
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
      self.b_o = self.add_weight(
          'bias_o',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
      self.b_c = self.add_weight(
          'bias_c',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)
      # Indices are non-trainable, but we save them as weight to simplify
      # checkpoints and restoring.
      self.sparse_indices = self.add_weight(
          name='indices',
          shape=sparse_indices.shape,
          dtype=tf.int64,
          initializer=lambda shape, dtype: sparse_indices,
          trainable=False)
    else:
      self.w_f, self.w_i, self.w_o, self.w_c = None, None, None, None
      self.u_f, self.u_i, self.u_o, self.u_c = None, None, None, None
      self.b_f, self.b_i, self.b_o, self.b_c = None, None, None, None

  def build(self, input_shape):

    super().build(input_shape)

    self.hidden_states = self.add_weight(
        name='hidden_states',
        initializer=tf.keras.initializers.Zeros(),
        shape=(self.max_batch_size, *input_shape[1:]),
        trainable=False,
        dtype=tf.float32)
    self.cell_states = self.add_weight(
        name='cell_states',
        initializer=tf.keras.initializers.Zeros(),
        shape=(self.max_batch_size, *input_shape[1:]),
        trainable=False,
        dtype=tf.float32)

  def get_config(self):
    config = super().get_config()
    config.update({
        'trainable_reservoir_per_cell': self.trainable_reservoir_per_cell,
        'use_bias_per_cell': self.use_bias_per_cell,
    })
    return config

  def call(
      self,
      inputs: tf.Tensor,
      prior_states: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.Tensor:
    """LSTM forward function.

    Args:
      inputs: the input tensor into the model
      prior_states: the previous states of the recurrent layer

    Returns:
      outputs: the output tensor from the recurrent layer

    Raises:
      RuntimeError: if the kernel is not initialized
      RuntimeError: if the recurrent kernel is not initialized
      ValueError: if the input tensor has fewer than 2 dimensions
    """
    if (self.w_f is None or self.w_i is None or self.w_o is None or
        self.w_c is None):
      raise RuntimeError('The reservoir of the LSTM kernel layer is not '
                         'initialized. Call LSTM with a weight matrix to '
                         'initialize it.')
    if (self.u_f is None or self.u_i is None or self.u_o is None or
        self.u_c is None):
      raise RuntimeError('The reservoir of the LSTM recurrent kernel is not '
                         'initialized. Call LSTM with a weight matrix to '
                         'initialize it.')

    # Use the first batch_size cell_states.
    batch_size = tf.shape(inputs)[0]

    if prior_states is not None:
      hidden_states, cell_states = prior_states[0], prior_states[1]
    else:
      hidden_states = self.state_discount * tf.gather(self.hidden_states,
                                                      tf.range(batch_size))
      cell_states = self.state_discount * tf.gather(self.cell_states,
                                                    tf.range(batch_size))
    hidden_states = tf.reshape(hidden_states, (-1, hidden_states.shape[-1]))
    cell_states = tf.reshape(cell_states, (-1, cell_states.shape[-1]))

    rank = inputs.shape.rank
    if rank < 2:
      raise ValueError(f'Input tensor must be at least 2D, but rank={rank}.')

    sp_w_f = tf.SparseTensor(self.sparse_indices, self.w_f,
                             self.reservoir_shape)
    sp_w_i = tf.SparseTensor(self.sparse_indices, self.w_i,
                             self.reservoir_shape)
    sp_w_o = tf.SparseTensor(self.sparse_indices, self.w_o,
                             self.reservoir_shape)
    sp_w_c = tf.SparseTensor(self.sparse_indices, self.w_c,
                             self.reservoir_shape)
    sp_u_f = tf.SparseTensor(self.sparse_indices, self.u_f,
                             self.reservoir_shape)
    sp_u_i = tf.SparseTensor(self.sparse_indices, self.u_i,
                             self.reservoir_shape)
    sp_u_o = tf.SparseTensor(self.sparse_indices, self.u_o,
                             self.reservoir_shape)
    sp_u_c = tf.SparseTensor(self.sparse_indices, self.u_c,
                             self.reservoir_shape)

    # Pack the batch dimension because sparse_dense_matmul only allows 2D
    # matrices.
    inputs_2d = tf.reshape(inputs, (-1, inputs.shape[-1]))

    input_part_f = tf.sparse.sparse_dense_matmul(
        inputs_2d,
        sp_w_f,  # Prefers kernel to be in column-major order.
        adjoint_a=False,
        adjoint_b=False)
    input_part_i = tf.sparse.sparse_dense_matmul(
        inputs_2d,
        sp_w_i,  # Prefers kernel to be in column-major order.
        adjoint_a=False,
        adjoint_b=False)
    input_part_o = tf.sparse.sparse_dense_matmul(
        inputs_2d,
        sp_w_o,  # Prefers kernel to be in column-major order.
        adjoint_a=False,
        adjoint_b=False)
    input_part_c = tf.sparse.sparse_dense_matmul(
        inputs_2d,
        sp_w_c,  # Prefers kernel to be in column-major order.
        adjoint_a=False,
        adjoint_b=False)

    input_part_f_unpacked = tf.reshape(
        input_part_f,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])
    input_part_i_unpacked = tf.reshape(
        input_part_i,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])
    input_part_o_unpacked = tf.reshape(
        input_part_o,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])
    input_part_c_unpacked = tf.reshape(
        input_part_c,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])

    self.w_f = learning.LocalLearning(
        name='kernel_local_learning_layer_f',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_f_unpacked,
            weight=self.w_f,
        )

    self.w_i = learning.LocalLearning(
        name='kernel_local_learning_layer_i',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_i_unpacked,
            weight=self.w_i,
        )

    self.w_o = learning.LocalLearning(
        name='kernel_local_learning_layer_o',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_o_unpacked,
            weight=self.w_o,
        )

    self.w_c = learning.LocalLearning(
        name='kernel_local_learning_layer_c',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_part_c_unpacked,
            weight=self.w_c,
        )

    for _ in np.arange(self.recurrence_degree + 1):
      if hidden_states is not None:
        recurrent_part_f = tf.sparse.sparse_dense_matmul(
            hidden_states,
            sp_u_f,  # in column-major order.
            adjoint_a=False,
            adjoint_b=False)
        recurrent_part_i = tf.sparse.sparse_dense_matmul(
            hidden_states,
            sp_u_i,  # in column-major order.
            adjoint_a=False,
            adjoint_b=False)
        recurrent_part_o = tf.sparse.sparse_dense_matmul(
            hidden_states,
            sp_u_o,  # in column-major order.
            adjoint_a=False,
            adjoint_b=False)
        recurrent_part_c = tf.sparse.sparse_dense_matmul(
            hidden_states,
            sp_u_c,  # in column-major order.
            adjoint_a=False,
            adjoint_b=False)
        f_part = input_part_f + recurrent_part_f
        i_part = input_part_i + recurrent_part_i
        o_part = input_part_o + recurrent_part_o
        c_part = input_part_c + recurrent_part_c
      else:
        f_part = input_part_f
        i_part = input_part_i
        o_part = input_part_o
        c_part = input_part_c

      if self.use_bias_per_cell is not None:
        if self.use_bias_per_cell[0]:
          f_part = tf.nn.bias_add(f_part, self.b_f)
        if self.use_bias_per_cell[1]:
          i_part = tf.nn.bias_add(i_part, self.b_i)
        if self.use_bias_per_cell[2]:
          o_part = tf.nn.bias_add(o_part, self.b_o)
        if self.use_bias_per_cell[3]:
          c_part = tf.nn.bias_add(c_part, self.b_c)

      f_t = tf.keras.activations.sigmoid(f_part)
      i_t = tf.keras.activations.sigmoid(i_part)
      o_t = tf.keras.activations.sigmoid(o_part)
      c_t = tf.keras.activations.tanh(c_part)
      c_t = tf.math.multiply(f_t, cell_states) + tf.math.multiply(i_t, c_t)
      h_t = tf.math.multiply(o_t, tf.keras.activations.tanh(c_t))
      cell_states, hidden_states = c_t, h_t

      hidden_states_unpacked = tf.reshape(
          hidden_states,
          [i if i is not None else -1 for i in inputs.shape[:-1]] +
          [self.units])

      self.u_f = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_f',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states_unpacked,
              weight=self.u_f,
          )

      self.u_i = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_i',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states_unpacked,
              weight=self.u_i,
          )

      self.u_o = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_o',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states_unpacked,
              weight=self.u_o,
          )

      self.u_c = learning.LocalLearning(
          name='recurrent_kernel_local_learning_layer_c',
          learning_rule=self.recurrent_kernel_local_learning)(
              learning_params=self.recurrent_kernel_local_learning_params,
              activation=hidden_states_unpacked,
              weight=self.u_c,
          )

    outputs = hidden_states

    # Unpack the batch dimension
    if self.keep_memory:
      cell_states = tf.reshape(
          cell_states, [i if i is not None else -1 for i in inputs.shape[:-1]] +
          [self.units])
      hidden_states = tf.reshape(
          hidden_states,
          [i if i is not None else -1 for i in inputs.shape[:-1]] +
          [self.units])
      self.hidden_states.batch_scatter_update(
          tf.IndexedSlices(
              hidden_states,
              tf.range(batch_size),
              dense_shape=self.hidden_states.shape))
      self.cell_states.batch_scatter_update(
          tf.IndexedSlices(
              cell_states,
              tf.range(batch_size),
              dense_shape=self.cell_states.shape))

    outputs = tf.reshape(
        outputs,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])

    return outputs


class DenseReservoirRecurrentCell(tf.keras.layers.SimpleRNNCell):
  """Cell class to use in RNN unfolding in time series data.

  Usage:
    ```
    weight = some_get_weight_function()
    model = tf.keras.layers.RNN(layers.DenseReservoirRecurrentCell(10))
    ```

  Attributes:
    weight: the reservoir weights.
    trainable_reservoir: Whether the reservoir is trainable. Default True.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation is
        applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0. ```
    kernel_local_learning: local learning rules to use (e.g. hebbian, oja, etc.)
    recurrent_kernel_local_learning: local learning rules in recurrence.
    local_learning_params: the parameters for the local learning rules.
  """

  def __init__(
      self,
      weight: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
      trainable_reservoir: bool = True,
      activation: str = 'tanh',
      use_bias: bool = True,
      kernel_initializer: str = 'glorot_uniform',
      recurrent_initializer: str = 'orthogonal',
      bias_initializer: str = 'zeros',
      kernel_regularizer: Optional[types.Regularizer] = None,
      recurrent_regularizer: Optional[types.Regularizer] = None,
      bias_regularizer: Optional[types.Regularizer] = None,
      kernel_constraint: Optional[types.Regularizer] = None,
      recurrent_constraint: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Regularizer] = None,
      dropout: Optional[float] = 0.,
      recurrent_dropout: Optional[float] = 0.,
      kernel_local_learning: Optional[str] = 'none',
      recurrent_kernel_local_learning: Optional[str] = 'none',
      kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      recurrent_kernel_local_learning_params: Optional[Dict[str, Any]] = None,
      **kwargs):
    self.units = weight.shape[0]  # pytype: disable=attribute-error
    super().__init__(
        units=self.units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        **kwargs)
    self.weight = weight
    self.trainable_reservoir = trainable_reservoir

    if weight is not None:

      # compatibility with SparseReservoir
      if isinstance(weight, sparse.spmatrix):
        weight = weight.toarray()

      # Only allow numpy arrays for the initial reservoir weight.
      if not isinstance(weight, np.ndarray):
        raise TypeError(
            f'Only accept ndarray or spmatrix objects. Got {type(weight)}')

      self.reservoir_kernel_initializer = reservoir_kernel_initializer(weight)
      self.kernel = self.add_weight(
          name='kernel',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=self.reservoir_kernel_initializer,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir,
      )
      self.recurrent_kernel = self.add_weight(
          name='recurrent_kernel',
          shape=weight.shape,
          dtype=tf.float32,
          initializer=self.reservoir_kernel_initializer,
          regularizer=self.kernel_regularizer,
          trainable=trainable_reservoir,
      )
    else:
      self.kernel = None
      self.recurrent_kernel = None
    self.kernel_local_learning = kernel_local_learning
    self.recurrent_kernel_local_learning = recurrent_kernel_local_learning
    self.kernel_local_learning_params = kernel_local_learning_params or {}
    self.recurrent_kernel_local_learning_params = (
        recurrent_kernel_local_learning_params or {})

  def call(self, inputs: tf.Tensor,
           states: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """RNN forward function.

    Args:
      inputs: A 2D tensor, with shape of `[batch, feature]`.
      states: the previous cell states of the recurrent layer

    Returns:
      outputs: the output tensor from the recurrent layer

    Raises:
      RuntimeError: if the kernel is not initialized
      RuntimeError: if the recurrent kernel is not initialized
    """
    if self.kernel is None:
      raise RuntimeError('The reservoir of the RNN kernel layer is not '
                         'initialized. Call RNN with a weight matrix to '
                         'initialize it.')
    if self.recurrent_kernel is None:
      raise RuntimeError('The reservoir of the RNN recurrent kernel is not '
                         'initialized. Call RNN with a weight matrix to '
                         'initialize it.')

    cell_states = states[0]

    input_states = inputs @ self.kernel
    self.kernel = learning.LocalLearning(
        name='kernel_local_learning_layer',
        learning_rule=self.kernel_local_learning)(
            learning_params=self.kernel_local_learning_params,
            activation=input_states,
            weight=self.kernel)

    cell_states = cell_states @ self.recurrent_kernel
    cell_states += input_states

    if self.bias is not None:
      cell_states = tf.nn.bias_add(cell_states, self.bias)
    if self.activation is not None:
      cell_states = self.activation(cell_states)

    self.recurrent_kernel = learning.LocalLearning(
        name='recurrent_kernel_local_learning_layer',
        learning_rule=self.recurrent_kernel_local_learning)(
            learning_params=self.recurrent_kernel_local_learning_params,
            activation=cell_states,
            weight=self.recurrent_kernel)

    return cell_states, [cell_states]
