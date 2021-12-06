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

"""Tests for reservoir_nn.keras.layers."""
import os
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reservoir_nn.keras import layers
from scipy import sparse
import tensorflow as tf

FLAGS = flags.FLAGS

StandardLayerParameters = parameterized.named_parameters(
    ('SparseReservoir', layers.SparseReservoir),
    ('DenseReservoir', layers.DenseReservoir),
)

AllLayerParameters = parameterized.named_parameters(
    ('SparseReservoir', layers.SparseReservoir),
    ('DenseReservoir', layers.DenseReservoir),
    ('RecurrentDenseReservoir', layers.RecurrentDenseReservoir),
    ('RecurrentSparseReservoir', layers.RecurrentSparseReservoir),
    ('LSTMDenseReservoir', layers.LSTMDenseReservoir),
    ('LSTMSparseReservoir', layers.LSTMSparseReservoir),
)

# unused for now, but leave room for future testing
RNNLayerParameters = parameterized.named_parameters(
    ('RecurrentDenseReservoir', layers.RecurrentDenseReservoir),
    ('RecurrentSparseReservoir', layers.RecurrentSparseReservoir),
    ('LSTMDenseReservoir', layers.LSTMDenseReservoir),
    ('LSTMSparseReservoir', layers.LSTMSparseReservoir),
)


class ReservoirLayerTest(parameterized.TestCase):

  @AllLayerParameters
  def test_it_raises_type_error_for_weights(self, layer_class):
    with self.assertRaisesRegex(
        TypeError, 'Only accept ndarray or spmatrix objects. Got .*'):
      layer_class(object())

  @StandardLayerParameters
  def test_initialize_with_sparse_matrix_works(self, layer_class):
    weights = np.arange(42 * 42).reshape((42, 42)).astype(np.float32)
    layer = layer_class(sparse.coo_matrix(weights))
    x = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)
    result = layer(x)
    expected = x @ tf.constant(weights)

    tf.debugging.assert_equal(result, expected)

  @StandardLayerParameters
  def test_it_should_produce_correct_output(self, layer_class):
    weights = np.arange(42 * 10).reshape((42, 10)).astype(np.float32)
    layer = layer_class(weights)
    x = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)
    result = layer(x)
    expected = x @ tf.constant(weights)

    tf.debugging.assert_equal(result, expected)

  @StandardLayerParameters
  def test_it_should_behave_the_same_as_weight_swapped_dense_layer(
      self, layer_class):
    weights = np.arange(42 * 10).reshape((42, 10)).astype(np.float32)
    layer = layer_class(weights)
    dense_layer = tf.keras.layers.Dense(weights.shape[1])
    dense_layer.build((42,))
    dense_layer_weights = dense_layer.get_weights()
    dense_layer_weights[0] = weights
    dense_layer.set_weights(dense_layer_weights)

    x = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)

    tf.debugging.assert_equal(layer(x), dense_layer(x))
    self.assertEqual(layer.units, 10)

  @StandardLayerParameters
  def test_it_should_be_the_same_as_weight_swapped_dense_with_3d_input(
      self, layer_class):
    weights = np.arange(4 * 5).reshape((4, 5)).astype(np.float32)
    layer = layer_class(weights)
    dense_layer = tf.keras.layers.Dense(weights.shape[1])

    input_shape = (2, 3, 4)
    dense_layer.build(input_shape[1:])  # Batch dimension is omitted
    dense_layer_weights = dense_layer.get_weights()
    dense_layer_weights[0] = weights
    dense_layer.set_weights(dense_layer_weights)

    # 3D input (batch, dim1, dim2). It should be flatten to (batch * dim1, dim2)
    #   by the layers internally and reshaped back to (batch, dim1, out_dim)
    x = tf.reshape(
        tf.range(tf.reduce_prod(input_shape), dtype=tf.float32), input_shape)
    result = layer(x)
    tf.debugging.assert_equal(result, dense_layer(x))
    self.assertEqual(layer.units, 5)
    self.assertEqual(result.shape, (2, 3, 5))

  @AllLayerParameters
  def test_it_should_raise_value_error_on_1d_input(self, layer_class):
    weights = np.arange(4 * 5).reshape((4, 5)).astype(np.float32)
    layer = layer_class(weights)

    input_shape = (4)  # 1D input
    x = tf.reshape(tf.range(4, dtype=tf.float32), input_shape)

    with self.assertRaisesRegex(ValueError,
                                r'Input tensor must be at least 2D'):
      layer(x)

  @StandardLayerParameters
  def test_it_should_apply_bias(self, layer_class):
    weights = np.arange(3 * 2).reshape((3, 2)).astype(np.float32)
    layer = layer_class(
        weights,
        use_bias=True,
        bias_initializer='ones',
    )
    x = tf.constant([[-5, -4, 3]], dtype=tf.float32)

    result = layer(x)
    expected = tf.constant([[5, -1]], dtype=tf.float32)

    tf.debugging.assert_equal(result, expected)

  @StandardLayerParameters
  def test_it_should_apply_activation(self, layer_class):
    weights = np.arange(3 * 2).reshape((3, 2)).astype(np.float32)
    layer = layer_class(
        weights,
        activation=tf.nn.relu,
    )
    x = tf.constant([[-5, -4, 3]], dtype=tf.float32)
    result = layer(x)

    # If relu not applied, will be [[4, -2]]
    expected = tf.constant([[4, 0]], dtype=tf.float32)

    tf.debugging.assert_equal(result, expected)

  @StandardLayerParameters
  def test_it_should_apply_activation_within_recurrence(self, layer_class):
    weights = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    layer = layer_class(
        weights,
        use_bias=True,
        bias_initializer='ones',
        activation=tf.nn.relu,
        activation_within_recurrence=True,
        recurrence_degree=3,
    )
    x = tf.constant([[0, 0, 0]], dtype=tf.float32)

    result = layer(x)
    expected = tf.constant([[1810.0, 2335.0, 2860.0]], dtype=tf.float32)

    tf.debugging.assert_equal(result, expected)

  @AllLayerParameters
  def test_get_config_should_be_the_same_before_and_after_reconstruction(
      self, layer_class):
    weights = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    layer = layer_class(
        weights,
        use_bias=True,
        activation=tf.nn.relu,
        bias_initializer='ones',
        bias_regularizer=tf.keras.regularizers.L1(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        bias_constraint=tf.keras.constraints.max_norm(2.),
        recurrence_degree=3,
        trainable_reservoir=True,
        name='test_layer',
    )
    config = layer.get_config()
    reconstructed = layer_class.from_config(config)
    self.assertEqual(reconstructed.get_config(), config)

  @AllLayerParameters
  def test_it_should_reconstruct_with_load_model(self, layer_class):  # pylint:disable=g-doc-args
    """This test makes sure DenseReservoir's get_config works with load_model.

    As well as the weights are properly restored.
    """
    weights = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    layer = layer_class(weights)
    model = tf.keras.Sequential([layer])
    x = tf.constant([[-5, -4, 3]], dtype=tf.float32)
    model.build(x.shape)

    with tempfile.TemporaryDirectory(dir=FLAGS.test_tmpdir) as tmp_dir:
      model_dir = os.path.join(tmp_dir, 'test_model')
      model.save(model_dir)
      with tf.keras.utils.CustomObjectScope({layer_class.__name__: layer_class
                                            }):
        reconstructed_model = tf.keras.models.load_model(model_dir)
    tf.debugging.assert_equal(reconstructed_model(x), model(x))
    self.assertEqual(reconstructed_model.get_config(), model.get_config())

  @AllLayerParameters
  def test_call_should_raise_error_if_not_initialized_with_weights(
      self, layer_class):
    layer = layer_class()
    x = tf.constant([[-5, -4, 3]], dtype=tf.float32)
    with self.assertRaises(RuntimeError):
      layer(x)

  @StandardLayerParameters
  def test_non_trainable_weights_should_not_change_when_trained(
      self, layer_class):
    weights = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    layer = layer_class(weights, trainable_reservoir=False, recurrence_degree=3)
    initial_kernel = layer.kernel.initialized_value().numpy()
    model = tf.keras.Sequential([layer])
    model.compile(loss='categorical_crossentropy')
    input_ = tf.constant([[0.5, 0, -0.5]])
    label = tf.constant([[1, 0, 2]])
    model.fit(input_, label)
    # Weights should not change
    tf.debugging.assert_near(layer.kernel, initial_kernel)

  @StandardLayerParameters
  def test_trainable_weights_should_change_when_trained(self, layer_class):
    weights = np.arange(3 * 3).reshape((3, 3)).astype(np.float32)
    layer = layer_class(weights, trainable_reservoir=True, recurrence_degree=3)
    model = tf.keras.Sequential([layer])
    model.compile(loss='categorical_crossentropy')
    input_ = tf.constant([[0.5, 0, -0.5]])
    label = tf.constant([[1, 0, 2]])
    model.fit(input_, label)
    # Weights should change (not equal)
    self.assertTrue(tf.reduce_any(layer.kernel != weights))

  @AllLayerParameters
  def test_it_can_use_hebbian_learning(self, layer_class):
    weights = np.arange(10 * 10).reshape((10, 10)).astype(np.float32)
    layer = layer_class(
        weights,
        kernel_local_learning='hebbian',
        kernel_local_learning_params={'eta': 0.1})
    x = tf.expand_dims(tf.range(10, dtype=tf.float32), axis=0)
    _ = layer(x)

  @RNNLayerParameters
  def test_rnns_can_use_recurrent_hebbian_learning(self, layer_class):
    weights = np.arange(10 * 10).reshape((10, 10)).astype(np.float32)
    layer = layer_class(
        weights,
        recurrent_kernel_local_learning='hebbian',
        recurrent_kernel_local_learning_params={'eta': 0.1})
    x = tf.expand_dims(tf.range(10, dtype=tf.float32), axis=0)
    _ = layer(x)

  @AllLayerParameters
  def test_it_can_use_oja_learning(self, layer_class):
    weights = np.arange(10 * 10).reshape((10, 10)).astype(np.float32)
    layer = layer_class(
        weights,
        kernel_local_learning='oja',
        kernel_local_learning_params={'eta': 0.1})
    x = tf.expand_dims(tf.range(10, dtype=tf.float32), axis=0)
    _ = layer(x)

  @RNNLayerParameters
  def test_rnns_can_use_recurrent_oja_learning(self, layer_class):
    weights = np.arange(10 * 10).reshape((10, 10)).astype(np.float32)
    layer = layer_class(
        weights,
        recurrent_kernel_local_learning='oja',
        recurrent_kernel_local_learning_params={'eta': 0.1})
    x = tf.expand_dims(tf.range(10, dtype=tf.float32), axis=0)
    _ = layer(x)

  @RNNLayerParameters
  def test_rnns_can_set_keep_memory(self, layer_class):
    weights = np.arange(10 * 10).reshape((10, 10)).astype(np.float32)
    layer = layer_class(weights, keep_memory=True)
    x = tf.expand_dims(tf.range(10, dtype=tf.float32), axis=0)
    _ = layer(x)


class Conv2DReservoirTest(absltest.TestCase):

  def test_it_should_behave_the_same_as_weight_swapped_conv2d_layer(self):
    weights = np.arange(6**2, dtype=np.float32).reshape((6, 6))
    layer = layers.Conv2DReservoir(weights, filters=(1, 4), kernel_size=(3, 3))

    weights = np.reshape(weights, (3, 3, 1, 4))
    conv2d_layer = tf.keras.layers.Conv2D(4, (3, 3))
    conv2d_layer.build((8, 8, 1))
    conv2d_layer_weights = conv2d_layer.get_weights()
    conv2d_layer_weights[0] = weights
    conv2d_layer.set_weights(conv2d_layer_weights)

    x = tf.reshape(tf.range(64, dtype=np.float32), [1, 8, 8, 1])

    tf.debugging.assert_equal(layer(x), conv2d_layer(x))
    self.assertEqual(layer.units, 36)

  def test_it_should_raise_value_error_on_3d_input(self):
    weights = np.arange(6**2, dtype=np.float32).reshape((6, 6))
    layer = layers.Conv2DReservoir(weights)

    x = tf.reshape(tf.range(64, dtype=np.float32), [1, 8, 8])

    with self.assertRaisesRegex(ValueError,
                                r'Input tensor must be at least 4D'):
      layer(x)


class SparseReservoirTest(absltest.TestCase):
  """Unique tests for SparseReservoir."""

  def test_sparse_evolution_interface_works(self):
    weights = np.arange(6**2, dtype=np.float32).reshape((6, 6))
    layer = layers.SparseReservoir(weights)

    shape = layer.get_reservoir_shape()
    self.assertEqual(shape, weights.shape)

    # this shall not fail
    indices, values, ages = layer.get_sparse_tensors()

    # mutation works
    layer.assign_sparse_tensors(indices * 0, values * 0, ages + 1)
    indices, values, _ = layer.get_sparse_tensors()
    np.testing.assert_array_equal(indices, 0)
    np.testing.assert_array_equal(values, 0.)


class PerNeuronSparseReservoirTest(absltest.TestCase):

  def test_it_stores_correct_reservoir_kernels(self):
    # channels with zero weights will not be part of the reservoir layer
    # non-zero weights on column are in pattern [2*i+1, 2*i+2] for easy testing
    reservoir_weight = np.array([[1, 0, 5], [0, 3, 6], [2, 4, 0]])
    model = tf.keras.models.Sequential(name='model')
    model.add(tf.keras.layers.InputLayer((3, 1), name='input_layer'))
    model.add(tf.keras.layers.Flatten(name='flatten_input'))
    model.add(
        layers.PerNeuronSparseReservoir(weight=reservoir_weight, name='sparse'))

    sparse_reservoir = model.get_layer('sparse')
    for i in range(3):
      # non-zero weights on column:
      non_zero_weight = np.array([[2 * i + 1], [2 * i + 2]])
      layer = sparse_reservoir._single_output_layers[i]
      kernel = layer.kernel.numpy()
      np.testing.assert_equal(kernel, non_zero_weight)

  def test_it_should_produce_correct_output(self):
    weights = np.arange(42 * 10).reshape((42, 10)).astype(np.float32)
    sparse_layer = layers.PerNeuronSparseReservoir(weights)
    x = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)
    result = sparse_layer(x)
    expected = x @ tf.constant(weights)

    tf.debugging.assert_equal(result, expected)

  def test_it_should_produce_correct_output_with_some_zero_weights(self):
    weights = np.arange(42 * 10).reshape((42, 10)).astype(np.float32)
    # Set some weights to zero.
    weights[1, 9] = 0
    weights[4, 2] = 0
    weights[23, 8] = 0
    sparse_layer = layers.PerNeuronSparseReservoir(weights)
    x = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)
    result = sparse_layer(x)
    expected = x @ tf.constant(weights)

    tf.debugging.assert_equal(result, expected)

  def test_it_should_behave_the_same_as_dense_reservoir_of_the_same_weight(
      self):
    weights = np.arange(42 * 10).reshape((42, 10)).astype(np.float32)
    sparse_layer = layers.PerNeuronSparseReservoir(weights)
    dense_layer = layers.DenseReservoir(weights)

    x = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)

    tf.debugging.assert_equal(sparse_layer(x), dense_layer(x))

  def test_it_is_the_same_as_dense_reservoir_of_the_same_weight_with_3d_input(
      self):
    weights = np.arange(4 * 5).reshape((4, 5)).astype(np.float32)
    sparse_layer = layers.PerNeuronSparseReservoir(weights)
    dense_layer = layers.DenseReservoir(weights)
    input_shape = (2, 3, 4)
    x = tf.reshape(
        tf.range(tf.reduce_prod(input_shape), dtype=tf.float32), input_shape)

    result = sparse_layer(x)

    tf.debugging.assert_equal(result, dense_layer(x))
    self.assertEqual(result.shape, (2, 3, 5))


class SelectiveSensorTest(absltest.TestCase):

  def test_it_should_produce_correct_output(self):
    num_sensors_per_channel = 2
    x = tf.constant([
        [
            [0., 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ],
        [
            [8, 9],
            [10, 11],
            [12, 13],
            [14, 15],
        ],
    ])
    kernel_initializer = lambda shape, dtype: np.ones(shape)
    # Because weights = ones and there are 2 sensors per channel, the output has
    # 2 copies per channel of the input.
    expected = tf.constant([
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
            [6.0, 6.0, 7.0, 7.0],
        ],
        [
            [8.0, 8.0, 9.0, 9.0],
            [10.0, 10.0, 11.0, 11.0],
            [12.0, 12.0, 13.0, 13.0],
            [14.0, 14.0, 15.0, 15.0],
        ],
    ])
    selective_sensors = layers.SelectiveSensor(
        num_sensors_per_channel=num_sensors_per_channel,
        kernel_initializer=kernel_initializer,
    )
    result = selective_sensors(x)
    tf.debugging.assert_equal(result, expected)


class SparseSensorTest(absltest.TestCase):

  def test_it_should_produce_correct_output(self):
    num_input_channels = 2
    num_sensors = 3
    weight = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    x = tf.constant([[[0.0], [1.0], [2.0], [3.0]]])
    expected = tf.constant([[[3.0, 4.0, 5.0], [9.0, 14.0, 19.0]]])
    sparsensor = layers.SparseSensor(
        num_input_channels=num_input_channels,
        num_sensors=num_sensors,
        weight=weight,
    )
    result = sparsensor(x)
    tf.debugging.assert_equal(result, expected)

  def test_it_should_produce_correct_output_with_recurrence(self):
    num_input_channels = 3
    num_sensors = 3
    weight = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    x = tf.constant([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]])
    expected = tf.constant([[[46203., 59598., 72993.],
                             [118698., 153108., 187518.]]])

    rnn_sparsensor = layers.RecurrentSparseSensor(
        num_input_channels=num_input_channels,
        num_sensors=num_sensors,
        weight=weight,
        recurrence_degree=3,
        trainable_reservoir=True)
    result = rnn_sparsensor(x)
    tf.debugging.assert_equal(result, expected)


class ModuleTest(absltest.TestCase):

  def test_get_coo_indices_and_values_row_major(self):
    weights = np.arange(4).reshape(2, 2)
    sp_ind, sp_val = layers._get_coo_indices_and_values(weights, 1, order='C')
    np.testing.assert_array_equal(sp_ind, [[0, 1], [1, 0], [1, 1]])
    np.testing.assert_array_equal(sp_val, [1, 2, 3])

  def test_get_coo_indices_and_values_column_major(self):
    weights = np.arange(4).reshape(2, 2)
    sp_ind, sp_val = layers._get_coo_indices_and_values(weights, 1, order='F')
    np.testing.assert_array_equal(sp_ind, [[1, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(sp_val, [2, 1, 3])

  def test_get_coo_indices_and_values_power(self):
    weights = np.arange(4).reshape(2, 2)
    sp_ind, sp_val = layers._get_coo_indices_and_values(weights, 2, order='F')
    np.testing.assert_array_equal(sp_ind, [[0, 0], [1, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(sp_val, [2, 6, 3, 11])


class RNNSpecificTest(absltest.TestCase):

  def test_RecurrentDenseReservoir_can_take_prior_states(self):
    weights = np.arange(4 * 4).reshape((4, 4)).astype(np.float32)
    layer = layers.RecurrentDenseReservoir(weights)

    input_shape = (2, 3, 4)
    # 3D input (batch, dim1, dim2). It should be flatten to (batch * dim1, dim2)
    #   by the layers internally and reshaped back to (batch, dim1, out_dim)
    x = tf.reshape(
        tf.range(tf.reduce_prod(input_shape), dtype=tf.float32), input_shape)
    prior_states = tf.zeros((2, 3, 4))
    result = layer(x, prior_states=prior_states)
    expected = tf.constant([[[56., 62., 68., 74.], [152., 174., 196., 218.],
                             [248., 286., 324., 362.]],
                            [[344., 398., 452., 506.], [440., 510., 580., 650.],
                             [536., 622., 708., 794.]]])
    tf.debugging.assert_equal(result, expected)

  def test_RecurrentSparseReservoir_can_take_prior_states(self):
    weights = np.arange(4 * 4).reshape((4, 4)).astype(np.float32)
    layer = layers.RecurrentSparseReservoir(weights)

    input_shape = (2, 3, 4)
    # 3D input (batch, dim1, dim2). It should be flatten to (batch * dim1, dim2)
    #   by the layers internally and reshaped back to (batch, dim1, out_dim)
    x = tf.reshape(
        tf.range(tf.reduce_prod(input_shape), dtype=tf.float32), input_shape)
    prior_states = tf.zeros((2, 3, 4))
    result = layer(x, prior_states=prior_states)
    expected = tf.constant([[[56., 62., 68., 74.], [152., 174., 196., 218.],
                             [248., 286., 324., 362.]],
                            [[344., 398., 452., 506.], [440., 510., 580., 650.],
                             [536., 622., 708., 794.]]])
    tf.debugging.assert_equal(result, expected)

  def test_LSTMDenseReservoir_can_take_prior_states(self):
    weights = np.arange(4 * 4).reshape((4, 4)).astype(np.float32)
    layer = layers.LSTMDenseReservoir(weights)

    input_shape = (2, 3, 4)
    # 3D input (batch, dim1, dim2). It should be flatten to (batch * dim1, dim2)
    #   by the layers internally and reshaped back to (batch, dim1, out_dim)
    x = tf.reshape(
        tf.range(tf.reduce_prod(input_shape), dtype=tf.float32), input_shape)
    prior_states = [tf.zeros((2, 3, 4))] * 2
    result = layer(x, prior_states=prior_states)
    expected = tf.constant([[[0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942]],
                            [[0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942]]])
    tf.debugging.assert_equal(result, expected)

  def test_LSTMSparseReservoir_can_take_prior_states(self):
    weights = np.arange(4 * 4).reshape((4, 4)).astype(np.float32)
    layer = layers.LSTMSparseReservoir(weights)

    input_shape = (2, 3, 4)
    # 3D input (batch, dim1, dim2). It should be flatten to (batch * dim1, dim2)
    #   by the layers internally and reshaped back to (batch, dim1, out_dim)
    x = tf.reshape(
        tf.range(tf.reduce_prod(input_shape), dtype=tf.float32), input_shape)
    prior_states = [tf.zeros((2, 3, 4))] * 2
    result = layer(x, prior_states=prior_states)
    expected = tf.constant([[[0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942]],
                            [[0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942],
                             [0.7615942, 0.7615942, 0.7615942, 0.7615942]]])
    tf.debugging.assert_equal(result, expected)

  def test_RecurrentSparseSensor_should_produce_correct_output(self):
    num_input_channels = 3
    num_sensors = 3
    weight = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    x = tf.constant([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]])
    expected = tf.constant([[[19.5, 24., 28.5], [46.5, 60., 73.5]]])
    rnn_sparsensor = layers.RecurrentSparseSensor(
        num_input_channels=num_input_channels,
        num_sensors=num_sensors,
        weight=weight)
    result = rnn_sparsensor(x)
    tf.debugging.assert_equal(result, expected)


class DenseReservoirRecurrentCellTest(absltest.TestCase):

  def test_densernncell_should_produce_hidden_states_as_outputs(self):
    weight = np.arange(9, dtype=np.float32).reshape((3, 3))
    cell = layers.DenseReservoirRecurrentCell(weight)
    input_shape = (42,)
    cell.build(input_shape)
    inputs = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)
    input_states = [tf.zeros((1, 3))]
    outputs, hidden_states = cell(inputs, input_states)
    tf.debugging.assert_equal(outputs, hidden_states[0])

  def test_hebbian_densernncell_should_produce_hidden_states_as_outputs(self):
    weight = np.arange(9, dtype=np.float32).reshape((3, 3))
    cell = layers.DenseReservoirRecurrentCell(
        weight,
        kernel_local_learning='hebbian',
        recurrent_kernel_local_learning='hebbian')
    input_shape = (42,)
    cell.build(input_shape)
    inputs = tf.expand_dims(tf.range(42, dtype=tf.float32), axis=0)
    input_states = [tf.zeros((1, 3))]
    outputs, hidden_states = cell(inputs, input_states)
    tf.debugging.assert_equal(outputs, hidden_states[0])


if __name__ == '__main__':
  absltest.main()
