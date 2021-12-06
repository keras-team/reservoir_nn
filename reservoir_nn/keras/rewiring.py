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

"""Neural Net Rewiring algorithms.

Includes the following.

Sparse Evolutionary Training (SET).

The model is a mashup of two evolutionary training frameworks with some
local twists:

a. (SET) Sparse evolutionary training:
https://www.nature.com/articles/s41467-018-04316-3

b. (TD) Targeted Dropout: https://arxiv.org/abs/1905.13678

Algorithm:

1. Select the k-least (fraction_mutation_candidates) significant connections
based on the absolute value of the weights and a probability (TD).

2. Stochastically drop least significant connections per training batch (TD),
before back propagation starts (SET). Unlike TD, the probability in SET
favors least significiant connections with a negative softmax.
The rate of mutations per mutation step is fixed.

3. New connections are replenished to ensure the sparsity of the model is
exactly conserved (SET).

The mutation rate can be controlled with a global mutation schedule. For
example at the later epochs of the training when the weight update slows down,
we can compensate by reducing the mutation rate with the training.


Usage:

1. Create a model with some SparseEvolutionLayer layers.
2. Train the model with a MutationCallback callback.

"""

import abc
import dataclasses
import functools
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from reservoir_nn.typing import types
from scipy import sparse
import tensorflow as tf


class RoundRobinConnection():
  """Creates a roundrobin sparse reservoir."""

  def __init__(self, num_connections: int):
    self.num_connections = num_connections

  def __call__(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if shape[0] * shape[-1] < self.num_connections:
      raise ValueError(f'Cannot build layer because the target'
                       f'({self.num_connections}) is larger than max number'
                       f'of possible connections between input units '
                       f'({shape[0]}) and output units ({shape[1]})')

    initial_indices = np.zeros(shape=(self.num_connections, 2), dtype='int64')
    initial_indices[...] = np.arange(self.num_connections)[:, None]
    initial_indices[:, 0] %= shape[0]
    initial_indices[:, 1] %= shape[1]

    initial_values = np.zeros(self.num_connections)

    return initial_indices, initial_values


class CooMatrixConnection():
  """Creates a sparse reservoir for the input."""

  def __init__(self, weight_matrix: types.WeightMatrix):
    self.coo = sparse.coo_matrix(weight_matrix)

  def __call__(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if self.coo.shape != shape:
      raise ValueError(f'Reservoir has a shape of {self.coo.shape}, '
                       f'but the layer expects {shape}')

    initial_indices = np.array([self.coo.row, self.coo.col], dtype='int64').T
    initial_values = self.coo.data

    return initial_indices, initial_values


def _get_reservoir_initializer(
    reservoir_initializer: Any) -> types.ReservoirInitializer:
  if isinstance(reservoir_initializer, int):  # num_connections
    return RoundRobinConnection(reservoir_initializer)
  elif isinstance(reservoir_initializer, (np.ndarray, sparse.spmatrix)):
    return CooMatrixConnection(reservoir_initializer)
  return reservoir_initializer


class SparseMutableInterface(metaclass=abc.ABCMeta):
  """Interface to implement to allow a class become mutable."""

  @abc.abstractmethod
  def get_reservoir_shape(self):
    """Returns the shape of the reservoir."""

    raise NotImplementedError

  @abc.abstractmethod
  def get_sparse_tensors(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns the sparse evolution tensors.

    Returns:
      sparse_indices, sparse_values, sparse_ages. Tensors describing
        the current state of the sparsity.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def assign_sparse_tensors(self, sparse_indices: tf.Tensor,
                            sparse_values: tf.Tensor, sparse_ages: tf.Tensor):
    """Returns the sparse evolution tensors.

    Args:
      sparse_indices: see below
      sparse_values: see below
      sparse_ages: Tensors describing the current state of the sparsity.
    """
    raise NotImplementedError


class AdaptiveSparseReservoir(SparseMutableInterface, tf.keras.layers.Layer):
  """An example layer with trainable connections via SparseMutableInterface.

  Attributes:
    units: size of the output.
    reservoir_initializer: Initializer for the reservoir. If int, generate an
      empty reservoir with as many connections of zero weights. If an ndarray or
      sparse.spmatrix, create a reservoir with the given connection. If a
      callable, calls the function with the shape of the adjacency matrix and
      expects the function to return the indices and values in tensorflow's
      SparseTensor format.
    activation: The activation function, for example tf.nn.relu.
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_regularizer: Regularizer function applied to the sparse weights.
    bias_initializer: Initializer for the bias vector.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation").
    bias_constraint: Constraint function applied to the bias vector.
    name: Optional name for the layer.
    **kwargs: Other keyword arguments.
  """

  def __init__(
      self,
      units: int,
      reservoir_initializer: Union[int, types.WeightMatrix,
                                   types.ReservoirInitializer],
      activation: Optional[types.Activation] = None,
      use_bias: bool = False,
      kernel_regularizer: Optional[types.Regularizer] = None,
      bias_initializer: types.Initializer = 'zeros',
      bias_regularizer: Optional[types.Regularizer] = None,
      activity_regularizer: Optional[types.Regularizer] = None,
      bias_constraint: Optional[types.Constraint] = None,
      name: Optional[str] = None,
      **kwargs: Any,
  ):
    super().__init__(
        activity_regularizer=activity_regularizer, name=name, **kwargs)

    self.reservoir_initializer = _get_reservoir_initializer(
        reservoir_initializer)

    self.units = units
    self.bias = None
    self.use_bias = use_bias
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.activation = tf.keras.activations.get(activation)

  def build(self, input_shape: Tuple[int, ...]):
    reservoir_shape = (input_shape[-1], self.units)

    self.reservoir_shape = reservoir_shape

    initial_indices, initial_values = self.reservoir_initializer(
        reservoir_shape)

    self._num_connections = len(initial_indices)

    self.sparse_indices = self.add_weight(
        name='indices',
        shape=tf.shape(initial_indices),
        dtype=tf.int64,
        initializer=lambda shape, dtype: tf.cast(initial_indices, dtype),
        trainable=False,
    )

    self.sparse_values = self.add_weight(
        name='kernel',
        shape=tf.shape(initial_values),
        dtype=tf.float32,
        initializer=lambda shape, dtype: tf.cast(initial_values, dtype),
        regularizer=self.kernel_regularizer,
        trainable=True,
    )

    self.sparse_ages = self.add_weight(
        name='ages',
        shape=tf.shape(initial_values),
        dtype=tf.int64,
        initializer=tf.zeros,
        trainable=False,
    )

    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=tf.float32,
          trainable=True)

    super().build(input_shape)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:

    # NOTE(feyu): if I define the sparse tensor in build, we get a gradient
    # not computed error.
    sparse_matrix = tf.SparseTensor(self.sparse_indices, self.sparse_values,
                                    self.reservoir_shape)
    # pack the batch dimension
    outputs = tf.reshape(inputs, (-1, inputs.shape[-1]))

    outputs = tf.sparse.sparse_dense_matmul(
        outputs, sparse_matrix, adjoint_a=False, adjoint_b=False)

    # unpack the batch dimension
    outputs = tf.reshape(
        outputs,
        [i if i is not None else -1 for i in inputs.shape[:-1]] + [self.units])

    if self.bias is not None:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs

  def get_coo_weight_matrix(self) -> sparse.spmatrix:
    """Returns the weight matrix as a scipy sparse matrix."""
    return sparse.coo_matrix(
        (self.sparse_values,
         (self.sparse_indices[:, 0], self.sparse_indices[:, 1])),
        shape=self.reservoir_shape)

  def get_coo_age_matrix(self) -> sparse.spmatrix:
    """Returns the connection age matrix as a scipy sparse matrix."""
    return sparse.coo_matrix(
        (self.sparse_ages,
         (self.sparse_indices[:, 0], self.sparse_indices[:, 1])),
        shape=self.reservoir_shape)

  def get_reservoir_shape(self):
    return self.reservoir_shape

  def get_sparse_tensors(self):
    sparse_indices = self.sparse_indices.value()
    sparse_values = self.sparse_values.value()
    sparse_ages = self.sparse_ages.value()
    return sparse_indices, sparse_values, sparse_ages

  def assign_sparse_tensors(self, sparse_indices, sparse_values, sparse_ages):
    self.sparse_indices.assign(sparse_indices)
    self.sparse_values.assign(sparse_values)
    self.sparse_ages.assign(sparse_ages)

  def get_config(self):
    config = super().get_config()
    config.update({
        'units':
            self.units,
        # upon restart create indices and values with the correct shape
        'reservoir_initializer':
            self._num_connections,
        'activation':
            tf.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
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
    })
    return config


@dataclasses.dataclass(frozen=True)
class GlobalPolicy:
  """Global mutation policy for modulating mutation rate."""
  scale_candidate_fraction: float = 1.0
  scale_candidate_mutation_rate: float = 1.0


@dataclasses.dataclass(frozen=True)
class MutationPolicy:
  """Mutation policy."""
  candidate_fraction: float = 0.1
  candidate_mutation_rate: float = 0.5
  minimum_candidate_age: float = 0
  uniform_probability: bool = False

  def apply_global_policy(self, gpolicy: GlobalPolicy) -> 'MutationPolicy':
    return MutationPolicy(
        candidate_fraction=self.candidate_fraction *
        gpolicy.scale_candidate_fraction,
        candidate_mutation_rate=self.candidate_mutation_rate *
        gpolicy.scale_candidate_mutation_rate,
        minimum_candidate_age=self.minimum_candidate_age,
        uniform_probability=self.uniform_probability)

  def mutation_step(self,
                    layer: SparseMutableInterface,
                    rng: Optional[np.random.RandomState] = None):
    """Mutates the reservoir of the sparse layer for one step.

    This function modifies the weights in-place. Least significant connections
    are dropped, and new connections are uniformly sampled to replenish the
    reservoir, such that the total number of connections is conserved.

    The ages of preserved connections are incremented by 1. The ages of
    new connections are set to zero. The weight of new connections are also
    set to zero.

    Args:
      layer: Layer to mutate. The layer is mutated in place.
      rng: If given, the random number generator to use.
    """
    rng = rng or np.random

    if not isinstance(layer, SparseMutableInterface):
      raise ValueError(
          f'Layer of type {type(layer)} does not support sparse evolution. Implement SparseMutableInterface.'
      )

    sparse_indices, sparse_values, sparse_ages = layer.get_sparse_tensors()

    shape = layer.get_reservoir_shape()

    mutation_prob = self.compute_mutation_probability(
        sparse_values=sparse_values, sparse_ages=sparse_ages)

    npfunc = functools.partial(
        _mutate_sparse_reservoir,
        mutation_rate=self.candidate_mutation_rate * self.candidate_fraction,
        rng=rng,
        shape=shape)

    sparse_indices, sparse_values, sparse_ages = tf.numpy_function(
        npfunc, [sparse_indices, sparse_values, sparse_ages, mutation_prob],
        [tf.int64, tf.float32])

    layer.assign_sparse_tensors(
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        sparse_ages=sparse_ages)

  def compute_mutation_probability(self, sparse_values: tf.Tensor,
                                   sparse_ages: tf.Tensor) -> tf.Tensor:
    """Returns the mutation probability as a tf Tensor.


    Args:
      sparse_values: weights of the sparse connections.
      sparse_ages: ages of the sparse connections. Only used when
        minimum_candidate_age is set.
    """
    a = tf.abs(sparse_values)

    if self.minimum_candidate_age > 0:
      # shuffle young connections to the end of ranking.
      where = tf.where(sparse_ages < self.minimum_candidate_age)
      a = tf.tensor_scatter_nd_add(a, where,
                                   tf.repeat(tf.reduce_max(a), tf.size(where)))

    arg = tf.argsort(a)
    # This is a standard recipe for getting the rank of data.
    rank = tf.argsort(arg)

    num_mutation_candidates = self.candidate_fraction * tf.size(
        sparse_values, tf.float32)
    # Select the least sigificant connections
    mask = tf.cast(rank, tf.float32) < num_mutation_candidates

    # Enable simple model in the TD paper.
    if self.uniform_probability:
      return tf.cast(mask, tf.float32) / num_mutation_candidates

    adrop = tf.boolean_mask(a, mask)
    # normalize to sum(w) == 1
    w = tf.nn.softmax(-adrop)
    pdrop = tf.tensor_scatter_nd_update(
        tf.zeros(a.shape, tf.float32), tf.where(mask), w)
    return pdrop


def _mutate_sparse_reservoir(
    sparse_indices: np.ndarray,
    sparse_values: np.ndarray,
    sparse_ages: np.ndarray,
    mutation_prob: np.ndarray,
    mutation_rate: float,
    rng: np.random.RandomState,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Mutates the sparse reservoir.

  The mutation replaces connections with new connections. The probability
  for a connection to mutate is given in mutation_prob.

  Args:
    sparse_indices: Array of shape (N, 2), indices of the sparse connections.
    sparse_values: Array of shape (N,), values of the sparse connections.
    sparse_ages: Array of shape (N,), ages of the sparse connections.
    mutation_prob: Array of shape (N,), probability of mutation occurring on the
      connection.
    mutation_rate: rate of mutation. Expected number of mutation is
      mutation_rate * num_connections.
    rng: Random generator to use.
    shape: shape of the full connection reservoir.

  Returns:
    sparse_indices, sparse_values: the mutated sparse reservoir, in the row
    major order to ease efficiency of tf.SparseTensor.
  """
  # NOTE(feyu): It is hard to write 'set' operations in tf.
  # if this is too slow, consider rewrite with Cython or C++/Clif.

  num_connections = len(sparse_indices)

  if sparse_indices.shape != (num_connections, 2):
    raise ValueError('unexpected shape of sparse_indices')
  if num_connections != len(sparse_values):
    raise ValueError('unexpected length of sparse_values')
  if num_connections != len(sparse_ages):
    raise ValueError('unexpected length of sparse_ages')
  if num_connections != len(mutation_prob):
    raise ValueError('unexpectd length of mutation_prob')

  # Sample the number of mutations for this mutate step from a Poisson.
  # We cap the result. Should have used a truncated poisson distribution.
  # This approximation is valid when num_mutations is large (~ > 100).
  num_mutations = rng.poisson(num_connections * mutation_rate)
  num_mutation_sites = (mutation_prob > 0).sum()
  num_mutations = min(num_mutations, num_mutation_sites)

  # draw mutation sites with the mutation probability.
  mask = np.ones(num_connections, 'bool')

  if num_mutations > 0:
    sites = rng.choice(
        num_connections, size=num_mutations, replace=False, p=mutation_prob)
    mask[sites] = False

  sparse_values = sparse_values[mask]
  sparse_indices = sparse_indices[mask]
  sparse_ages = sparse_ages[mask]
  live_integers = np.ravel_multi_index(sparse_indices.T, shape)

  unique_live_integers = set(live_integers)

  new_integers = set([])

  while len(new_integers) < num_mutations:
    candidates = rng.choice(
        shape[0] * shape[1],
        size=num_mutations - len(new_integers),
        replace=False)
    new_integers.update(candidates)
    new_integers.difference_update(unique_live_integers)

  new_indices = np.array(
      np.unravel_index(np.array(list(new_integers), dtype='int64'), shape)).T

  # Appends the new connections.
  sparse_indices = np.append(sparse_indices, new_indices, axis=0)
  sparse_values = np.append(sparse_values,
                            np.zeros(num_mutations, dtype=sparse_values.dtype))
  sparse_ages = np.append(sparse_ages + 1,
                          np.zeros(num_mutations, dtype=sparse_ages.dtype))

  # row major
  arg = np.lexsort(sparse_indices.T[::-1])

  return sparse_indices[arg], sparse_values[arg], sparse_ages[arg]


class MutationCallback(tf.keras.callbacks.Callback):
  """A keras callback to mutate layers with SparseMutableInterface.

  Use with model.fit(..., callbacks=[MutationCallback(...)]).
  """

  def __init__(self,
               policy: Union[MutationPolicy, Dict[str, MutationPolicy]],
               mutation_schedule: Optional[Callable[[int],
                                                    GlobalPolicy]] = None,
               rng: Optional[np.random.RandomState] = None,
               verbose: int = 0):
    """Constructor of MutationCallback.

    Args:
      policy: the mutation policy for all layers or a dict of policy per layer,
        keyed by the layer object or layer name.
      mutation_schedule: A function that returns global mutation policy, which
        modulates the policy per step.
      rng: The np.random.RandomState to use for the mutation.
      verbose: 0, quiet. 1, update messages.
    """
    self.rng = rng
    self.mutation_schedule = mutation_schedule
    self.policy = policy
    self.verbose = verbose
    super().__init__()

  def _get_policy(self, layer):
    if isinstance(self.policy, MutationPolicy):
      return self.policy
    if layer.name in self.policy:
      return self.policy[layer.name]
    return self.policy[layer]

  def on_epoch_begin(self, epoch, logs):
    if self.mutation_schedule:
      self._global_policy = self.mutation_schedule(epoch)
    else:
      self._global_policy = GlobalPolicy()

    if self.verbose > 0:
      logging.info('Epoch %05d: '
                   'global policy set to %s', epoch + 1, self._global_policy)

  def on_train_batch_begin(self, batch, logs):
    model = self.model

    with tf.control_dependencies(None):
      for layer in model.layers:
        if not layer.trainable:
          continue

        if not isinstance(layer, SparseMutableInterface):
          continue

        policy = self._get_policy(layer).apply_global_policy(
            self._global_policy)

        if self.verbose > 1:
          logging.info('Mutation step of layer %s with %s', layer.name, policy)

        policy.mutation_step(layer, rng=self.rng)
