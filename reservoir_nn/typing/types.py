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

"""Valkyrie's common unionized types."""

import os
from typing import Callable, List, Tuple, Union

import numpy as np
from scipy import sparse
import tensorflow as tf

# Reservoir weight matrix
WeightMatrix = Union[np.ndarray, sparse.spmatrix]

# Data that can be either tf.Tensor or numpy
TensorLike = Union[tf.Tensor, np.ndarray]

# Activation function applied in a keras layer
Activation = Union[str, Callable[[tf.Tensor], tf.Tensor]]

# Regularizer function applied in a keras layer
Regularizer = Union[str, tf.keras.regularizers.Regularizer]

# Initializer function applied in a keras layer
Initializer = Union[str, tf.keras.initializers.Initializer,
                    Callable[[Tuple[int, ...], tf.dtypes.DType], np.ndarray]]

# Constraint function applied in a keras layer
Constraint = Union[str, tf.keras.constraints.Constraint]

PathLike = Union[str, os.PathLike]

ReservoirInitializer = Callable[[Tuple[int, int]], Tuple[np.ndarray,
                                                         np.ndarray]]

ClassWeight = Union[float, List[float], Tuple[float, ...], np.ndarray]
