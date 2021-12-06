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

"""Functions for Valkyrie learning rules."""

import enum
from typing import Any, Dict, Optional, Union

from reservoir_nn.typing import types
import tensorflow as tf


@enum.unique
class LearningRuleName(enum.Enum):
  NONE = 'none'
  HEBBIAN = 'hebbian'
  OJA = 'oja'
  CONTRASTIVEHEBBIAN = 'contrastive_hebbian'


class LocalLearning(tf.keras.layers.Layer):
  """Layer to perform local learning."""

  def __init__(self, learning_rule: str = 'none', **kwargs):
    super().__init__(**kwargs)

    self.learning_rule = LearningRuleName(learning_rule).value

  def call(
      self,
      learning_params: Dict[str, Any],
      activation: types.TensorLike,
      weight: types.TensorLike,
      expected: Optional[types.TensorLike] = None
  ) -> Union[types.TensorLike, None]:
    """Perform local learning.

    Args:
      learning_params: dictionary containing all local learning parameters.
      activation: activation of the neurons in this layer.
      weight: weight before local learning.
      expected: expected activation of the layer.

    Returns:
      weight: weight after local learning.
    """

    eta = learning_params.get('eta', 0.1)
    gamma = learning_params.get('gamma', 0.1)
    max_iter = learning_params.get('max_iter', 10)

    activation = tf.reshape(activation, [-1, activation.shape[-1]])
    if expected is not None:
      expected = tf.reshape(expected, [-1, expected.shape[-1]])

    if self.learning_rule == LearningRuleName.NONE:
      pass
    elif self.learning_rule == LearningRuleName.HEBBIAN:
      weight = self.hebbian_learning(activation, weight, eta)
    elif self.learning_rule == LearningRuleName.OJA:
      weight = self.oja_learning(activation, weight, eta)
    elif self.learning_rule == LearningRuleName.CONTRASTIVEHEBBIAN:
      weight = self.contrastive_hebbian_learning(activation, weight, expected,
                                                 eta, gamma, max_iter)
    return weight

  def hebbian_learning(self,
                       activation: types.TensorLike,
                       weight: types.TensorLike,
                       eta: float = 0.1) -> types.TensorLike:
    """Hebbian Learning.

    Args:
      activation: the activation of the layer
      weight: the recurrent kernel weight
      eta: learning rate of the local learning

    Returns:
      weight: updated recurrent kernel weight
    """
    weight = weight + eta * (tf.transpose(activation) @ activation)
    weight = weight / tf.norm(weight)
    return weight

  def oja_learning(self,
                   activation: types.TensorLike,
                   weight: types.TensorLike,
                   eta: float = 0.1) -> types.TensorLike:
    """Oja Learning.

    Args:
      activation: the activation of the layer
      weight: the recurrent kernel weight
      eta: learning rate of the local learning

    Returns:
      weight: updated recurrent kernel weight
    """
    weight = weight + eta * (
        tf.transpose(activation) @ activation -
        tf.transpose(activation) @ (activation @ weight))
    return weight

  def contrastive_hebbian_learning(self,
                                   activation: types.TensorLike,
                                   weight: types.TensorLike,
                                   expected: types.TensorLike,
                                   eta: float = 0.1,
                                   gamma: float = 0.1,
                                   max_iter: int = 10) -> types.TensorLike:
    """Contrastive Hebbian Learning.

    Args:
      activation: the activation of the layer
      weight: the recurrent kernel weight
      expected: the expected activation of the layer
      eta: learning rate of the local learning
      gamma: hypothetical feedback rate
      max_iter: maximum iterations for the recurrent steps to convergence

    Returns:
      weight: updated recurrent kernel weight

    Raises:
      ValueError: if expected output is not provided.
      ValueError: if gamma is a negative or zero number.
    """
    if expected is None:
      raise ValueError('Expected output should be provided')
    if gamma <= 0.0:
      raise ValueError('Please provide a positive gamma value')
    free_activation = activation
    clamped_activation = expected
    for _ in range(max_iter):
      free_activation = free_activation @ weight
      clamped_activation = clamped_activation @ weight
    weight = weight + eta * (
        (tf.transpose(free_activation) @ free_activation) -
        (tf.transpose(clamped_activation) @ clamped_activation)) / gamma
    return weight
