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

"""Tests for reservoir_nn.keras.rewiring."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reservoir_nn.keras import rewiring
import tensorflow as tf


class AdaptiveSparseReservoirTest(parameterized.TestCase):

  def test_layer_with_num_connections_works(self):
    x = tf.constant([1.])
    layer = rewiring.AdaptiveSparseReservoir(
        units=10,
        reservoir_initializer=10,
    )
    layer(x)

  def test_layer_with_large_num_connections_fails(self):
    x = tf.constant([1.])
    layer = rewiring.AdaptiveSparseReservoir(
        units=10,
        reservoir_initializer=100,
    )

    with self.assertRaisesRegex(ValueError,
                                "Cannot build layer.*(100).*(1).*(10)"):
      layer(x)

  def test_layer_with_reservoir_works(self):
    initial_reservoir = np.arange(10).reshape(1, 10)

    x = tf.constant([1.])
    layer = rewiring.AdaptiveSparseReservoir(
        units=10,
        reservoir_initializer=initial_reservoir,
    )

    np.testing.assert_array_equal(layer(x), np.arange(10))

  def test_layer_with_misshaped_reservoir_fails(self):
    initial_reservoir = np.arange(10).reshape(2, 5)

    x = tf.constant([1.])
    layer = rewiring.AdaptiveSparseReservoir(
        units=10,
        reservoir_initializer=initial_reservoir,
    )

    with self.assertRaisesRegex(
        ValueError,
        r"Reservoir has a shape of \(2, 5\), but the layer expects \(1, 10\)"):
      layer(x)

  def test_get_coo_weight_matrix_works(self):
    initial_reservoir = np.arange(10).reshape(2, 5)

    x = tf.keras.Input(shape=(2,))
    layer = rewiring.AdaptiveSparseReservoir(
        units=5,
        reservoir_initializer=initial_reservoir,
    )
    layer(x)

    coo = layer.get_coo_weight_matrix().toarray()
    np.testing.assert_array_equal(coo, initial_reservoir)

  def test_get_coo_age_matrix_works(self):
    initial_reservoir = np.arange(10).reshape(2, 5)

    x = tf.keras.Input(shape=(2,))
    layer = rewiring.AdaptiveSparseReservoir(
        units=5,
        reservoir_initializer=initial_reservoir,
    )
    layer(x)

    coo = layer.get_coo_age_matrix().toarray()
    np.testing.assert_array_equal(coo, np.zeros((2, 5)))

    policy = rewiring.MutationPolicy(
        candidate_fraction=0.0,
        candidate_mutation_rate=1.0,
    )
    policy.mutation_step(layer)
    coo = layer.get_coo_age_matrix().toarray()
    np.testing.assert_array_equal(
        coo,
        [[0., 1, 1, 1, 1], [1., 1, 1, 1, 1]],
    )

  def test_apply_global_policy_works(self):
    policy = rewiring.MutationPolicy(
        candidate_fraction=0.5,
        candidate_mutation_rate=0.5,
    )
    gpolicy = rewiring.GlobalPolicy(
        scale_candidate_fraction=0.5, scale_candidate_mutation_rate=0.5)

    policy = policy.apply_global_policy(gpolicy)

    self.assertEqual(
        policy,
        rewiring.MutationPolicy(
            candidate_fraction=0.25, candidate_mutation_rate=0.25))

  def test_compute_mutation_probability_works(self):
    initial_reservoir = np.arange(10).reshape(2, 5)

    x = tf.keras.Input(shape=(2,))
    layer = rewiring.AdaptiveSparseReservoir(
        units=5,
        reservoir_initializer=initial_reservoir,
    )

    layer(x)

    policy = rewiring.MutationPolicy(
        candidate_fraction=1.0,
        candidate_mutation_rate=1.0,
    )

    p = policy.compute_mutation_probability(
        sparse_values=layer.sparse_values.value(),
        sparse_ages=layer.sparse_ages.value(),
    )

    coo = layer.get_coo_weight_matrix().copy()
    coo.data[:] = p

    np.testing.assert_allclose(
        coo.toarray(), [
            [0., 0.7, 0.3, 0, 0],
            [0., 0, 0, 0, 0],
        ], atol=0.1)

  def test_mutation_works(self):
    initial_reservoir = np.arange(10).reshape(2, 5)

    x = tf.keras.Input(shape=(2,))
    layer = rewiring.AdaptiveSparseReservoir(
        units=5,
        reservoir_initializer=initial_reservoir,
    )

    layer(x)
    coo = layer.get_coo_weight_matrix()
    np.testing.assert_allclose(coo.toarray(), [
        [0., 1, 2, 3, 4],
        [5., 6, 7, 8, 9],
    ])
    rng = np.random.RandomState(1234)

    policy = rewiring.MutationPolicy(
        candidate_fraction=0.2,
        candidate_mutation_rate=1.0,
    )

    policy.mutation_step(layer, rng)

    coo = layer.get_coo_weight_matrix()

    # least active connections are replenished with zeros with 100%
    # probability.
    np.testing.assert_allclose(coo.toarray(), [
        [0., 0, 2, 3, 4],
        [5., 6, 7, 8, 9],
    ])


class SparseEvolutionEnd2EndTest(parameterized.TestCase):

  def test_fit_small_model_works(self):
    """Testing fitting a 4x4 sparse reservoir with 4 truth connections."""
    nunits = 4
    inputs = tf.keras.Input(shape=(nunits,))

    policy = rewiring.MutationPolicy(
        candidate_fraction=0.5,
        candidate_mutation_rate=0.1,
    )
    layer = rewiring.AdaptiveSparseReservoir(
        units=nunits,
        reservoir_initializer=2 * nunits,
        # regularizer helps sparsify the redundant connections.
        kernel_regularizer=tf.keras.regularizers.l2(1e-2),
    )
    outputs = layer(inputs)

    rng = np.random.RandomState(1333)

    model = tf.keras.Model(inputs, outputs)

    # Low dimension model prefers SGD:
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.1, nesterov=True),
        loss="mse",
        metrics=["mse"])

    x = rng.uniform(size=(1000000, nunits)) - 0.5
    y = x[:, ::-1].copy()

    truth = np.eye(4)[::-1]

    def mutation_schedule(epoch):
      del epoch
      return rewiring.GlobalPolicy()

    model.fit(
        x,
        y,
        batch_size=int(len(x) / 100),
        epochs=10,
        verbose=False,
        callbacks=rewiring.MutationCallback(
            policy={layer: policy},
            mutation_schedule=mutation_schedule,
            rng=rng,
            verbose=1))

    connection = layer.get_coo_weight_matrix().toarray()

    # Use truth * 0.65 here because L2 biases the fit towards zero.
    np.testing.assert_allclose(connection, truth * 0.65, atol=0.10)

  def test_fit_large_model_works(self):
    """Testing fitting a 100x100 sparse reservoir with 100 truth connections."""
    nunits = 100
    inputs = tf.keras.Input(shape=(nunits,))
    policy = rewiring.MutationPolicy(
        candidate_fraction=0.2,
        candidate_mutation_rate=0.8,
    )
    layer = rewiring.AdaptiveSparseReservoir(
        units=nunits,
        reservoir_initializer=2 * nunits,
    )
    outputs = layer(inputs)

    model = tf.keras.Model(inputs, outputs)

    model.compile(loss="mse", metrics=["mse"])

    x = np.random.uniform(size=(100000, nunits)) - 0.5
    truth = np.eye(nunits)[::-1].copy()

    y = np.einsum("ij,jk->ik", x, truth)

    class Reporter(tf.keras.callbacks.Callback):

      def on_epoch_end(self, epoch, logs):
        connection = layer.get_coo_weight_matrix().toarray()
        ages = layer.get_coo_age_matrix().toarray()
        cross = np.abs(connection) * truth
        print(sorted(zip(*np.nonzero(cross))))
        significant_elements = np.sum(cross > 0.02)
        print(significant_elements)
        print(ages[cross > 0.02])

    model.fit(
        x,
        y,
        batch_size=1600,
        epochs=20,
        verbose=True,
        callbacks=[
            rewiring.MutationCallback(policy=policy, verbose=1),
            Reporter()
        ])

    connection = layer.get_coo_weight_matrix().todense()

    # We shall have some correlation with the truth after training for a while.
    cross = np.abs(connection) * truth
    np.testing.assert_allclose(np.sum(cross > 0.02), 80, atol=20)


if __name__ == "__main__":
  absltest.main()
