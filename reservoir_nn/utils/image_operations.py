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

"""Common image processing operations."""
from typing import Tuple

from absl import logging
import attr
import numpy as np
from reservoir_nn.typing import types
from reservoir_nn.utils import preprocessing
import tensorflow as tf
from tensorflow_addons import image as tfa_image


@attr.s(auto_attribs=True)
class TileImage(preprocessing.TensorTransform):
  """Creates overlapping tiles from an image.

  input_ must be rank 2 `(row, col)`.
  output is rank 3 `(tiles, row, col)` The tiles will be ordered row by row.
    That is, for patches

    A B C
    D E F

    the output will be in the order of (A, B, C, D, E, F).

  Note, padding='VALID' is used so only patches which are fully contained in
  the input image are included.

  Attributes:
    sizes: (size_rows, size_cols) The size of the extracted patches.
    strides: (stride_rows, stride_cols) How far the centers of two consecutive
      patches are in the images. Needs to be > 0.
    rates: This (inversed) rate to sample the input pixels, specifying how far
      two consecutive patch samples are in the input. For example (1, 1) samples
      every pixel. (2, 2) samples every two pixels.
    padding: Whether to use padding='VALID' (default) or 'SAME'. If
      padding='VALID' is used, only patches which are fully contained in the
      input image are included. If padding='SAME' is used, the entire input
      image is included. All patches whose starting point is within the input
      image are included, and all pixels outside the image are set to 0.
  """
  sizes: Tuple[int, int]
  strides: Tuple[int, int]
  rates: Tuple[int, int] = (1, 1)
  padding: str = 'VALID'

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    tf.debugging.assert_rank(input_, 2,
                             'Input to TileImage must be 2D tensors.')
    output = tf.image.extract_patches(
        input_[tf.newaxis, :, :, tf.newaxis],  # Expand to a 4D tensor
        sizes=(1, self.sizes[0], self.sizes[1], 1),
        strides=(1, self.strides[0], self.strides[1], 1),
        rates=(1, self.rates[0], self.rates[1], 1),
        padding=self.padding,
    )
    # Output is (num_img_in_row, num_img_in_col, flatten_img)
    # Reshape to (tiles, row, col)
    output = tf.reshape(output, (-1, self.sizes[0], self.sizes[1]))

    return output


@attr.s(auto_attribs=True)
class TileImage3D(preprocessing.TensorTransform):
  """Creates overlapping tiles from an image.

  input_ must be rank 3 `(row, col, channel)`.
  output is rank 4 `(tiles, row, col, channel)` The tiles will be ordered row by
  row.
    That is, for patches

    A B C
    D E F

    the output will be in the order of (A, B, C, D, E, F).

  Note, padding='VALID' is used so only patches which are fully contained in
  the input image are included.

  Attributes:
    sizes: (size_rows, size_cols, channels) The size of the extracted patches.
    strides: (stride_rows, stride_cols) How far the centers of two consecutive
      patches are in the images. Needs to be > 0.
    rates: This (inversed) rate to sample the input pixels, specifying how far
      two consecutive patch samples are in the input. For example (1, 1) samples
      every pixel. (2, 2) samples every two pixels.
  """
  sizes: Tuple[int, int, int]
  strides: Tuple[int, int]
  rates: Tuple[int, int] = (1, 1)

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    tf.debugging.assert_rank(input_, 3,
                             'Input to TileImage3D must be 3D tensors.')
    output = tf.image.extract_patches(
        input_[tf.newaxis, :, :, :],  # Expand to a 4D tensor
        sizes=(1, self.sizes[0], self.sizes[1], 1),
        strides=(1, self.strides[0], self.strides[1], 1),
        rates=(1, self.rates[0], self.rates[1], 1),
        padding='VALID',
    )

    # Output is (num_img_in_row, num_img_in_col, flatten_img)
    # Reshape to (tiles, row, col, chan)
    output = tf.reshape(output,
                        (-1, self.sizes[0], self.sizes[1], self.sizes[2]))

    return output


@attr.s(auto_attribs=True)
class TileNoOverlap(preprocessing.TensorTransform):
  """Tiles an image without overlapping.

  Attributes:
    input_shape: The shape of the input tensor.
    tile_shape: The (height, width) of the tiling window.
  """
  input_shape: Tuple[int, int]
  tile_shape: Tuple[int, int]

  def call(self, input_: tf.Tensor) -> tf.Tensor:

    # Check shape
    input_ = tf.ensure_shape(input_, (*self.input_shape, None))

    row_max = self.input_shape[0] // self.tile_shape[0] * self.tile_shape[0]
    col_max = self.input_shape[1] // self.tile_shape[1] * self.tile_shape[1]
    out_stack = []

    for row in range(0, row_max, self.tile_shape[0]):

      for col in range(0, col_max, self.tile_shape[1]):
        # get a tile:
        tile = input_[row:row + self.tile_shape[0],
                      col:col + self.tile_shape[1], :]
        out_stack.append(tile)

    return tf.stack(out_stack, axis=0)


@attr.s(auto_attribs=False)
class MixRGBChannels(preprocessing.TensorTransform):
  """Mixes the RGB channels of an image.

  input shape = [row, col, 3]
  output shape = [row, col]

  Attributes:
    red: Contribution from the red channel.
    green: Contribution from the green channel.
    blue: Contribution from the blue channel.
  """
  red = attr.ib(default=0, type=float)
  green = attr.ib(default=0, type=float)
  blue = attr.ib(default=0, type=float)
  _scale_factors = attr.ib(init=False, repr=False)

  def __attrs_post_init__(self):
    self._scale_factors = tf.reshape(
        tf.constant([self.red, self.green, self.blue], dtype=tf.float32),
        (1, 1, 3))

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    scaled = input_ * self._scale_factors
    return tf.reduce_sum(scaled, axis=-1)


@attr.s(auto_attribs=False)
class SelectChannels(preprocessing.TensorTransform):
  """Selects channels of an image.

  input shape = [row, col, 3]
  output shape = [row, col, num_channels]

  Attributes:
    red: Whether to select the red channel.
    green: Whether to select the green channel.
    blue: Whether to select the blue channel.
  """
  red = attr.ib(default=False, type=bool)
  green = attr.ib(default=False, type=bool)
  blue = attr.ib(default=False, type=bool)

  def __attrs_post_init__(self):
    channels = []
    if self.red:
      channels.append(0)
    if self.green:
      channels.append(1)
    if self.blue:
      channels.append(2)
    if not channels:
      raise ValueError(f'Expect at least one channel, but received: '
                       f'red={self.red}, green={self.green}, blue={self.blue}')
    self._channels = channels

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return tf.gather(input_, self._channels, axis=-1)


def tile_image(image: np.ndarray, tile_size: int = 30) -> np.ndarray:
  """Deprecated.

  Creates overlapping tiles from an image.

  Please use `TileImage()` instead.

  The tiles will overlap each other by half of their size, to make sure that
  no "edges" of objects in the original image ever end up only on the edge of
  a tile.

  Args:
    image: 2D numpy array - the image to tile.
    tile_size: the desired tile size.

  Returns:
    out_stack: a stacked np.array with dims (num_tiles, tile_size, tile_size).
  """
  logging.warning('`tile_image` is deprecated. Please use TileImage() instead.')
  sizes = (tile_size, tile_size)
  strides = (tile_size // 2, tile_size // 2)

  return TileImage(sizes, strides)(image).numpy()


def image_to_tiles(image: np.ndarray,
                   tile_size: int,
                   strides: Tuple[int, int],
                   padding=True) -> np.ndarray:
  """Creates overlapping tiles from an image.

  The tiles are created by sliding a window of (tile_size, tile_size) along
  the height and width of the image with stride as the sliding step size.

  Args:
    image: 2D numpy array - the image to tile.
    tile_size: the size of one side of the square tile.
    strides: tuple of step sizes of moving the tiling window along the image's
      height and width, respectively.
    padding: boolean, if True, the image is expanded and padded with zeros on
      the right and bottom edges, so that all tiles are fully inside the image.

  Returns:
    A stacked np.array with shape (num_tiles, tile_size, tile_size).

  Raises:
    ValueError: if tile_size is larger than the smaller dimension of the image.
  """
  height, width = image.shape

  if tile_size > min(height, width):
    raise ValueError('tile_size = {} exceeds the image height = {} '
                     'or width = {}'.format(tile_size, height, width))

  if padding:
    image = pad_image_right_bottom(image, tile_size, strides)
    height, width = image.shape

  num_tiles_per_height = (height - tile_size) // strides[0] + 1
  num_tiles_per_width = (width - tile_size) // strides[1] + 1

  out_stack = []

  for row in range(0, num_tiles_per_height * strides[0], strides[0]):

    for col in range(0, num_tiles_per_width * strides[1], strides[1]):
      # get a tile:
      tile = image[row:row + tile_size, col:col + tile_size]
      out_stack.append(tile)

  return np.stack(out_stack)


# TODO(b/175058778): unify pad_image_right_bottom and PadRightBottom
def pad_image_right_bottom(image: np.ndarray, tile_size: int,
                           strides: Tuple[int, int]) -> np.ndarray:
  """Pads the right and bottom edges of the image with zeros.

  Padding is done if afer tiling the image, there are still leftovers that
  are not fit in tiles.

  Args:
    image: 2D numpy array - the image to be padded.
    tile_size: the size of one side of the square tiling window.
    strides: tuple of step sizes of moving the tiling window along the image's
      height and width, respectively.

  Returns:
    The padded image.
  """

  height, width = image.shape
  leftover_rows = (height - tile_size) % strides[0]
  if leftover_rows > 0:
    pad_height = strides[0] - leftover_rows
  else:
    pad_height = 0

  leftover_cols = (width - tile_size) % strides[1]
  if leftover_cols > 0:
    pad_width = strides[1] - leftover_cols
  else:
    pad_width = 0

  if leftover_rows > 0 or leftover_cols > 0:
    return np.pad(image, ((0, pad_height), (0, pad_width)), 'constant')

  return image


@attr.s(auto_attribs=True)
class PadRightBottom(preprocessing.TensorTransform):
  """Pads the right and bottom edges of an image in tf.Tensor format with zeros.

    Padding is done if after tiling the image, there are still leftovers that
    do not fit in tiles.

  Attributes:
    tile_size: the size of one side of the square tiling window.
    strides: tuple of step sizes of moving the tiling window along the image's
      height and width, respectively.
  """
  tile_size: int
  strides: Tuple[int, int]

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    """Pads the right and bottom edges of the image with zeros.

    Padding is done if after tiling the image, there are still leftovers that
    do not fit in tiles.

    Args:
      input_: The image to be padded.

    Returns:
      The padded image.
    """
    tf.debugging.assert_rank_in(input_, (2, 3), 'Input must be either 2D or 3D')

    height, width = tf.shape(input_)[0], tf.shape(input_)[1]

    leftover_rows = (height - self.tile_size) % self.strides[0]
    pad_height = (self.strides[0] - leftover_rows) % self.strides[0]

    leftover_cols = (width - self.tile_size) % self.strides[1]
    pad_width = (self.strides[1] - leftover_cols) % self.strides[1]

    if leftover_rows == 0 and leftover_cols == 0:
      return input_

    if len(tf.shape(input_)) == 2:
      return tf.pad(input_, [[0, pad_height], [0, pad_width]], 'constant')

    return tf.pad(input_, [[0, pad_height], [0, pad_width], [0, 0]], 'constant')


@attr.s(auto_attribs=True)
class ResizeImage(preprocessing.TensorTransform):
  """Resizes an image to a target size.

  Note, if the new shape doesn't have the same aspect ratio of the original
  image, padding is done to avoid image distortion.

  Attributes:
    new_shape: (target_height, target_width) of the resized image.
  """
  shape: Tuple[int, int]

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    target_height, target_width = self.shape
    return tf.image.resize_with_pad(
        input_, target_height, target_width, method='bilinear', antialias=False)


@attr.s(auto_attribs=True)
class ResizeImageWithCropOrPad(preprocessing.TensorTransform):
  """Resizes an image to a target size without any interpolation.

  Note, if the new shape doesn't have the same aspect ratio of the original
  image, cropping or padding is done to completely avoid image distortion.

  Attributes:
    new_shape: (target_height, target_width) of the resized image.
  """
  shape: Tuple[int, int]

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    target_height, target_width = self.shape
    return tf.image.resize_with_crop_or_pad(input_, target_height, target_width)


@attr.s(auto_attribs=True)
class ResizeLabel(preprocessing.TensorTransform):
  """Resizes a label to a target size.

  Note, if the new shape doesn't have the same aspect ratio of the original
  label, padding is done to avoid image distortion.

  Attributes:
    shape: (target_height, target_width) of the resized label.
  """
  shape: Tuple[int, int]

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    target_height, target_width = self.shape
    label = tf.image.resize_with_pad(
        input_, target_height, target_width, method='bilinear', antialias=False)
    return tf.cast(tf.math.rint(label), tf.int32)


@attr.s(auto_attribs=True)
class ResizeLabelWithCropOrPad(preprocessing.TensorTransform):
  """Resizes a label to a target size without any interpolation.

  Note, if the new shape doesn't have the same aspect ratio of the original
  label, cropping or padding is done to completely avoid image distortion.

  Attributes:
    shape: (target_height, target_width) of the resized label.
  """
  shape: Tuple[int, int]

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    target_height, target_width = self.shape
    label = tf.image.resize_with_crop_or_pad(input_, target_height,
                                             target_width)
    return tf.cast(tf.math.rint(label), tf.int32)


@attr.s(auto_attribs=True)
class Crop(preprocessing.TensorTransform):
  """Crops an image on the right and bottom edges to the target shape.

  Original height and width must accommodate the cropping that:
  ```original_size >= target_size + offset```

  Attributes:
    shape: (target_height, target_width) of the cropped image.
    offset: (offset_height, offset_width), vertical and horizontal coordinates
      of the top-left corner of the result in the input.
  """
  shape: Tuple[int, int]
  offset: Tuple[int, int] = (0, 0)

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return tf.image.crop_to_bounding_box(input_, *self.offset, *self.shape)


class EqualizeHistogram(preprocessing.TensorTransform):
  """Equalizes image histogram."""

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return tfa_image.equalize(input_)


@attr.s(repr=True)
class PreserveTwoContrastiveLabels(preprocessing.TensorTransform):
  """Preserves two contrastive labels in data and converts them to -1 and 1.

  Only two labels that are in the input list are preserved. The rest are
  converted to "unlabeled" 0.

  Attributes:
    two_labels: The list of two labels to be preserved.
  """
  all_class_label_values = attr.ib(default=attr.Factory(dict))
  two_labels = attr.ib(default=attr.Factory(list))

  @two_labels.validator
  def check_class_labels(self, attribute, values):
    # check that list has exactly two labels:
    if len(values) != 2:
      raise ValueError(
          f'Expecting exactly two classes, but getting two_labels = {values}')

    # check invalid labels:
    for label in values:
      if label not in self.all_class_label_values:
        raise ValueError(
            f'Input class {label} is not one of {self.all_class_label_values}')

  def __attrs_post_init__(self):
    # list of values corresponding to the input labels:
    label_values = [
        self.all_class_label_values[label] for label in self.two_labels
    ]
    # convert to tf.Tensor format:
    label_tensor = tf.constant(label_values)
    # two new labels:
    new_labels = tf.constant([-1, 1])
    # labels not included in input are assigned the value of zero:
    otherwise = 0
    self._lookup_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(label_tensor, new_labels),
        otherwise)

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    return self._lookup_table.lookup(input_)


@attr.s(repr=True)
class PreserveSplitLabels(preprocessing.TensorTransform):
  """Preserves a list of labels in data and split them into separate channels.

  Only labels that are in the input list are preserved. The rest are converted
  to "unlabeled", i.e., zeros.

  Attributes:
    all_class_label_values: Input dict of all class labels and values. Note that
      the dict must have field "unlabeled".
    class_labels: Input list of labels to be preserved.
  """
  all_class_label_values = attr.ib(default=attr.Factory(dict))
  class_labels = attr.ib(default=attr.Factory(list))

  @all_class_label_values.validator
  def check_all_class_label_values(self, attribute, values):
    # It must have field "unlabeled".
    if 'unlabeled' not in values:
      raise ValueError(
          f'Input dict `all_class_label_values` does not have field "unlabel" '
          f'as expected: {values}')

  @class_labels.validator
  def check_class_labels(self, attribute, values):
    # Check empty list of labels
    if not values:
      raise ValueError('Input class_labels is an empty list: {}'.format(values))

    # Check invalid labels
    for label in values:
      if label not in self.all_class_label_values:
        raise ValueError(
            'input label {} is not in `all_class_label_values` = {}'.format(
                label, self.all_class_label_values))

  def __attrs_post_init__(self):
    # List of label values corresponding to the input class_labels
    self._labels = [
        self.all_class_label_values[item] for item in self.class_labels
    ]

    # Convert to tf.Tensor
    label_tensor = tf.constant(self._labels)

    # Classes not included in input are assigned the value of class "unlabeled"
    otherwise = self.all_class_label_values['unlabeled']

    self._lookup_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(label_tensor, label_tensor),
        otherwise)

  def call(self, input_: tf.Tensor) -> tf.Tensor:
    """Preserves then separates labels in the input data.

    Args:
      input_: The input labels.

    Returns:
      The transformed label data.
    """
    # Remove the color channel dimension if there is only one color
    if input_.shape[-1] == 1:
      input_ = tf.squeeze(input_, axis=-1)

    # Preserve the input classes and assign the unwanted classes to 'unlabeled'
    output = self._lookup_table.lookup(input_)

    # Create one-hot channels for all potential classes
    outputs = tf.one_hot(
        output, depth=len(self.all_class_label_values), dtype=tf.int32)

    return tf.gather(outputs, self._labels, axis=-1)


def merge_labels(label: types.TensorLike) -> tf.Tensor:
  """Merges the channels of the label.

  Assuming the input `label` has different channels representing different
  classes that are defined by `CLASS_LABELS_VALUES`. The input shape is
  (height, width, channels). This functions merges all separated channels so
  that the output shape is (height, width).

  Args:
    label: The input label with the classes separated into different channels.

  Returns:
    The merged label.
  """
  label = tf.convert_to_tensor(label)
  return tf.argmax(label, axis=-1, output_type=tf.int32)


def tiles_to_image(tiles: np.ndarray,
                   image_shape: Tuple[int, int],
                   strides: Tuple[int, int],
                   padding=True) -> np.ndarray:
  """Reconstructs an image from a stack of tiles.

  Suppose an image is converted to stack of tiles using function image_to_tiles,
  then the stack is transformed via some model without changing its shape, this
  function unpacks the transformed stack to reconstruct the transformed image.

  Args:
    tiles: stack of tiles with shape (num_tiles, tile_size, tile_size).
    image_shape: the desire shape of the constructed image.
    strides: tuple of step sizes of moving the tiling window along the image's
      height and width, respectively, as used in image_to_tiles.
    padding: boolean, if True, take into account the original image was expanded
      and padded with zeros on the right and top edges, so that all tiles are
      fully inside the image.

  Returns:
    The reconstructed image.
  """
  height, width = image_shape
  _, tile_size, _ = tiles.shape

  if padding:
    leftover_rows = (height - tile_size) % strides[0]
    if leftover_rows > 0:
      height = height + strides[0] - leftover_rows

    leftover_cols = (width - tile_size) % strides[1]
    if leftover_cols > 0:
      width = width + strides[1] - leftover_cols

  image = np.zeros(shape=(height, width))
  count = np.zeros(shape=(height, width))

  num_tiles_per_height = (height - tile_size) // strides[0] + 1
  num_tiles_per_width = (width - tile_size) // strides[1] + 1

  tile_idx = 0

  for row in range(0, num_tiles_per_height * strides[0], strides[0]):

    for col in range(0, num_tiles_per_width * strides[1], strides[1]):
      # add tile to image and count occurrences of each pixel:
      image[row:row + tile_size, col:col + tile_size] += tiles[tile_idx, :, :]
      count[row:row + tile_size, col:col + tile_size] += 1

      # move to the next tile_idx:
      tile_idx += 1

  # original shape:
  height, width = image_shape
  return image[:height, :width] / count[:height, :width]


def circular_mask(height: int,
                  width: int,
                  radius: float,
                  dtype=np.bool_) -> np.ndarray:
  """Creates a 2D circular mask centered at the middle of the output matrix.

  Elements in the circle are 1s and the rest is 0.

  Args:
    height: Height of the output matrix.
    width: Width of the output matrix.
    radius: Radius of the circle.
    dtype: Output dtype.

  Returns:
    A 2D circular mask.
  """
  vertical_points = height - 1
  horizontal_points = width - 1
  i, j = np.meshgrid(
      np.linspace(-vertical_points / 2, vertical_points / 2, height),
      np.linspace(-horizontal_points / 2, horizontal_points / 2, width),
      indexing='ij')
  distance_2 = i**2 + j**2
  radius_2 = radius**2

  return np.less_equal(distance_2, radius_2).astype(dtype)


def gaussian_mask(height: int,
                  width: int,
                  radius: float,
                  dtype=np.float32) -> np.ndarray:
  """Creates a 2D isotropic Gaussian mask centered at the middle of the matrix.

  We omit the 1 / (2 pi sigma^2) term in calculating the Gaussian distribution.
  The max value in the mask is 1.

  Args:
    height: Height of the output matrix.
    width: Width of the output matrix.
    radius: Standard deviation of the Gaussian kernel.
    dtype: Output dtype.

  Returns:
    A 2D Gaussian mask.
  """
  vertical_points = height - 1
  horizontal_points = width - 1
  i, j = np.meshgrid(
      np.linspace(-vertical_points / 2, vertical_points / 2, height),
      np.linspace(-horizontal_points / 2, horizontal_points / 2, width),
      indexing='ij')
  distance_2 = i**2 + j**2
  radius_2 = radius**2

  return np.exp(-(distance_2 / (2.0 * radius_2)), dtype=dtype)
