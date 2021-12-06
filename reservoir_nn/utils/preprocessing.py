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

"""Building blocks for constructing a preprocessing pipeline.

Designed for tf.Tensor and tf.data.Dataset.

There are two levels of abstraction in a pipeline. A "TensorTransform"
operates on a single tf.Tensor; a "DatasetTransform" operates on
a tf.data.Dataset.

TensorTransforms can be "chained" together to become a TensorTransform of
many steps, and they can be arranged in a "DatasetTransform" where each
TensorTransform is applied on a feature (i.e. an argument of the function
provided to dataset.map) in the dataset.

Use the `Layer` suffix when naming DatasetTransform classes to distinguish
transformations on `Dataset`s from transformations on `Tensor`s.

An example of the pipeline for the keyword spotting data is:
```
Sequential([
  RejectionResampleLayer(
    target_dist=[0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
                 0.09, 0.09, 0.09, 0.09],
    initial_dist=[0.637, 0.0363, 0.0363, 0.0363, 0.0363, 0.0363,
                  0.0363, 0.0363, 0.0363, 0.0363, 0.0363],
    label_index=1, deterministic=True
  ),
  MapLayer([
    Chain([
      RemoveDCOffset(),
      AlignAudio(
        output_sec=1.0, max_shift_sec=0, offset_sec=1.0,
        sample_rate=16000.0, deterministic=True),
      Spectrogram(sample_rate=16000.0, frame_length=512, resample_rate=100),
      MelFeatures(
        sample_rate=16000.0, num_mel_bins=128, lower_edge_hertz=75.0,
        upper_edge_hertz=7500.0),
      ScaleAudio(vmax=1),
      StridedWindow(sequence_length=1, sequence_stride=1, axis=-2),
      FlattenSpectrogram(),
    ]),
    OneHotLabel(num_classes=11)
  ], num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
])
```
"""
# Lint as: python3
import abc
import textwrap
from typing import Callable, Optional, Sequence, Tuple, Union

import attr
from reservoir_nn.typing import types
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DatasetTransform(abc.ABC):
  """Base class for transformations that operate on `tf.data.Dataset`s."""

  def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return self.call(dataset)

  @abc.abstractmethod
  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset


class TensorTransform(abc.ABC):
  """Base class for transformations that operate on `tf.Tensor`s."""

  def __call__(self, input_: types.TensorLike) -> tf.Tensor:
    input_ = tf.convert_to_tensor(input_)
    return self.call(input_)

  @abc.abstractmethod
  def call(self, input_: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError()


class Sequential(DatasetTransform, list):
  """Sequentially combines `DatasetTransform`s into a single DatasetTransform."""

  def __init__(self, layers: Optional[Sequence[DatasetTransform]] = None):
    """Initializes a Sequential layer.

    Args:
      layers: The layers to be combined sequentially.

    Raises:
      TypeError when layers is of type DatasetTransform. In other words, do
        `Sequential([layer])`. Don't do `Sequential(layer)`.
    """
    if isinstance(layers, DatasetTransform):
      raise TypeError('layers must be provided in an iterable.')
    if layers is None:
      list.__init__(self, [])
    else:
      list.__init__(self, layers)

  def __repr__(self):
    layer_str = ',\n'.join([repr(layer) for layer in self])
    layer_str = textwrap.indent(layer_str, '  ')
    return '{}([\n{}\n])'.format(self.__class__.__name__, layer_str)

  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    for layer in self:
      dataset = layer(dataset)
    return dataset


class MapLayer(DatasetTransform, list):
  """Maps a list of transformations on a Dataset in parallel.

  If transformations = [t1, t2, t3], and the dataset has 3 features [d1, d2, d3]
    in each example, then the output example is [t1(d1), t2(d2), t3(d3)].
  """

  def __init__(self,
               transformations: Optional[Sequence[TensorTransform]] = None,
               num_parallel_calls=AUTOTUNE,
               deterministic: Optional[bool] = None):
    """Initializes a MapLayer.

    Args:
      transformations: A list of transformations to map on each of the elements
        in the dataset.
      num_parallel_calls: num_parallel_calls used in `dataset.map()`.
      deterministic: deterministic used in `dataset.map()`.

    Raises:
      TypeError when transformations is of type TensorTransform.
    """
    if isinstance(transformations, TensorTransform):
      raise TypeError('transformations must be provided in an iterable.')
    if transformations is None:
      list.__init__(self, [])
    else:
      list.__init__(self, transformations)
    self.num_parallel_calls = num_parallel_calls
    self.deterministic = deterministic

  def __repr__(self):
    transformation_str = ',\n'.join([repr(t) for t in self])
    transformation_str = textwrap.indent(transformation_str, '  ')
    return '{}([\n{}\n], num_parallel_calls={}, deterministic={})'.format(
        self.__class__.__name__, transformation_str, self.num_parallel_calls,
        self.deterministic)

  def map_func(self, *args):
    if len(args) != self.__len__():
      raise AttributeError(
          f'The number of transformations = {self.__len__()} must be equal to '
          f'the number of features = {len(args)}.\nIn {repr(self)}')
    output = [f(input_) for input_, f in zip(args, self)]
    return output

  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    output = dataset.map(
        self.map_func,
        num_parallel_calls=self.num_parallel_calls,
        deterministic=self.deterministic)
    return output


class ParseDataLayer(DatasetTransform):
  """Parses entries in the dataset with a parser.

  For example, this layer can be used to convert (key, tf.Example) from an
  SSTableDataset to (image, label) pairs.
  """

  # TODO(b/182943036): The value of flat_map depends on the kind of parser
  #  provided. If having keep track of the type of parser and setting the
  #  flat_map argument becomes to much we will need to adjust.
  def __init__(self,
               parser: Callable[..., Union[Tuple[tf.Tensor, ...], tf.Tensor,
                                           tf.data.Dataset]],
               flat_map: bool = False,
               num_parallel_calls=AUTOTUNE,
               deterministic: Optional[bool] = None):
    """Initializes the transform.

    When parser produces tf.data.Dataset of a series of examples, set flat_map
    to true.

    Args:
      parser: Parser function that converts input tensors to output tensors.
      flat_map: Set to True when the parser function outputs a Dataset (of a
        series of examples). Set to False when it outputs a tuple of tensors
        (one example).
      num_parallel_calls: Number of concurrent workers for loading the data.
      deterministic: Whether the ordering of the data is preserved.
    """
    self.parser = parser
    self.flat_map = flat_map
    self.num_parallel_calls = num_parallel_calls
    self.deterministic = deterministic

  def __repr__(self):
    return '{}(parser={}, flat_map={}, num_parallel_calls={}, deterministic={})'.format(
        self.__class__.__name__, self.parser, self.flat_map,
        self.num_parallel_calls, self.deterministic)

  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    if self.flat_map:
      return dataset.interleave(
          self.parser,
          cycle_length=1 if self.deterministic else None,
          num_parallel_calls=self.num_parallel_calls,
          deterministic=self.deterministic)
    return dataset.map(
        self.parser,
        num_parallel_calls=self.num_parallel_calls,
        deterministic=self.deterministic)


@attr.s(auto_attribs=True)
class BatchLayer(DatasetTransform):
  """Batches the dataset."""
  batch_size: int

  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.batch(self.batch_size)


class UnBatchLayer(DatasetTransform):
  """Unbatches the batched dataset."""

  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return tf.data.experimental.unbatch()(dataset)


@attr.s(auto_attribs=True)
class CacheLayer(DatasetTransform):
  """Caches the dataset by calling `dataset.cache()`."""
  filename: str = ''

  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.cache(self.filename)


@attr.s
class RejectionResampleLayer(DatasetTransform):
  """Rebalances the audio dataset to have an even distribution of class labels.

  For efficiency, this step should be executed prior to converting the class
  labels to one hot coding as it requires class labels of type tf.int32/64.

  Attributes:
    target_dist: A list of floats specifying the target label distribtution.
    initial_dist: A list of floats specifying the initial label distribtution.
    label_index: Index of the label in an example, default to 1.
    deterministic: Whether the resampled output should be deterministic.
  """
  target_dist = attr.ib()
  initial_dist = attr.ib(default=None)
  label_index = attr.ib(default=1)
  deterministic = attr.ib(default=None)

  def _get_class_label(self, *args):
    return args[self.label_index]

  def call(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    # Generate dataset resampler transformation function
    resampler = tf.data.experimental.rejection_resample(
        self._get_class_label,
        target_dist=self.target_dist,
        initial_dist=self.initial_dist,
        seed=None if not self.deterministic else 0)

    # Apply resampler
    dataset = dataset.apply(resampler)

    # Drop extra copy of the labels
    output = dataset.map(
        lambda extra_label, features_and_label: features_and_label)

    return output


@attr.s(auto_attribs=True)
class Identity(TensorTransform):
  """A transformation that outputs the input."""

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return input_


class Chain(TensorTransform, list):
  """Chains transformations."""

  def __init__(self,
               transformations: Optional[Sequence[TensorTransform]] = None):
    if transformations is None:
      list.__init__(self, [])
    else:
      list.__init__(self, transformations)

  def __repr__(self):
    transformation_str = ',\n'.join([repr(t) for t in self])
    transformation_str = textwrap.indent(transformation_str, '  ')
    return '{}([\n{}\n])'.format(self.__class__.__name__, transformation_str)

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    output = input_
    for transformation in self:
      output = transformation(output)
    return output


class Lambda(TensorTransform):
  """A transformation that applies a provided function to the input."""

  def __init__(self, function: Callable[[tf.Tensor], tf.Tensor]):
    self.function = function

  def __repr__(self) -> str:
    return '{}({})'.format(self.__class__.__name__, self.function)

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return self.function(input_)


@attr.s(auto_attribs=True)
class OneHotLabel(TensorTransform):
  """Convert to onehot labels in order to use keras recall and precision metrics."""
  num_classes: int

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return tf.one_hot(input_, self.num_classes)


@attr.s(auto_attribs=True)
class DemeanTensor(TensorTransform):
  """Demean input tensor or set to some desired value.

  For audio signals that contain zero padding, zero values can be ignored by
  using the `avoid_zeros` argument.

  Attributes:
    mean_val: A float specifying the desired mean value.
    avoid_zeros: A bool specifying whether to avoid demeaning zero values.
  """
  mean_val: float = None
  avoid_zeros: bool = False

  def call(self, input_: tf.Tensor) -> tf.Tensor:

    # Demean input tensor.
    if self.avoid_zeros:
      condition = input_ != 0
      output = tf.where(condition, input_ - tf.reduce_mean(input_[condition]),
                        input_)
    else:
      output = input_ - tf.reduce_mean(input_)

    if self.mean_val is not None:
      # Set mean to some desired value.
      output += self.mean_val

    return output


@attr.s(auto_attribs=True)
class NormalizeTensor(TensorTransform):
  """Normalize input tensor by some value or between zero and that value.

  Attributes:
    max_val: A float specifying the absolute maximum value of the output.
    zero_min: A bool specifying whether to normalize between zero and max_val.
  """
  max_val: float = None
  zero_min: bool = False

  def call(self, input_: tf.Tensor) -> tf.Tensor:

    if self.zero_min:
      # Make min value zero.
      input_ -= tf.reduce_min(input_)

    # Normalize by absolute max value.
    # The output will be zero if the denominator is zero.
    output = tf.math.divide_no_nan(input_, tf.reduce_max(tf.abs(input_)))

    if self.max_val is not None:
      # Normalize by some desired value.
      output *= self.max_val

    return output


@attr.s(auto_attribs=True)
class Reshape(TensorTransform):
  """Reshapes the input tensor.

  Attributes:
    shape: Target shape.
  """
  shape: Tuple[int, ...]

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return tf.reshape(input_, self.shape)
