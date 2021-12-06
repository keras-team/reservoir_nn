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

"""Tests for reservoir_nn.utils.image_operations."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reservoir_nn.utils import image_operations
import tensorflow as tf


class TileImageTest(parameterized.TestCase):

  def test_tiles_have_correct_sizes(self):
    image = tf.reshape(tf.range(2004 * 1004), (2004, 1004))
    tile_sizes = (27, 53)
    transformation = image_operations.TileImage(
        sizes=tile_sizes, strides=(27 // 2, 53 // 2))
    result = transformation(image)
    self.assertEqual(result.shape[1:], tile_sizes)

  @parameterized.parameters((1234, 2345, 123), (2314, 1634, 113))
  def test_it_gives_correct_num_tiles(self, height, width, tile_size):
    image = tf.reshape(tf.range(height * width), (height, width))
    tile_sizes = (tile_size, tile_size)
    transformation = image_operations.TileImage(
        sizes=tile_sizes, strides=(tile_size // 2, tile_size // 2))
    result = transformation(image)

    # Number of tiles should be correct
    slide = tile_size // 2
    num_on_height = (height - tile_size) // slide + 1
    num_on_width = (width - tile_size) // slide + 1
    self.assertEqual(result.shape[0], num_on_height * num_on_width)

  def test_it_produces_correct_result(self):
    image = tf.reshape(tf.range(20), (5, 4))
    expected = tf.constant([[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                            [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                            [[4, 5, 6], [8, 9, 10], [12, 13, 14]],
                            [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
                            [[8, 9, 10], [12, 13, 14], [16, 17, 18]],
                            [[9, 10, 11], [13, 14, 15], [17, 18, 19]]])
    transformation = image_operations.TileImage(sizes=(3, 3), strides=(1, 1))
    result = transformation(image)
    tf.debugging.assert_equal(result, expected)

  def test_padding_produces_correct_result(self):
    image = tf.reshape(tf.range(16), (4, 4))
    expected = tf.constant([[[0, 0, 0, 0], [0, 0, 1, 2], [0, 4, 5, 6],
                             [0, 8, 9, 10]],
                            [[0, 0, 0, 0], [1, 2, 3, 0], [5, 6, 7, 0],
                             [9, 10, 11, 0]],
                            [[0, 4, 5, 6], [0, 8, 9, 10], [0, 12, 13, 14],
                             [0, 0, 0, 0]],
                            [[5, 6, 7, 0], [9, 10, 11, 0], [13, 14, 15, 0],
                             [0, 0, 0, 0]]])
    transformation = image_operations.TileImage(
        sizes=(4, 4), strides=(2, 2), padding='SAME')
    result = transformation(image)
    tf.debugging.assert_equal(result, expected)

  def test_padding_produces_num_pixels_if_stride_1(self):
    num_pixels = 64
    image = tf.reshape(tf.range(num_pixels), (8, 8))
    transformation = image_operations.TileImage(
        sizes=(4, 4), strides=(1, 1), padding='SAME')
    result = transformation(image)
    tf.debugging.assert_equal(result.shape[0], num_pixels)


class TileImage3DTest(parameterized.TestCase):

  def test_tiles_have_correct_sizes(self):
    image = tf.reshape(tf.range(42 * 24 * 3), (42, 24, 3))
    tile_sizes = (5, 5, 3)
    transformation = image_operations.TileImage3D(
        sizes=tile_sizes, strides=(5, 5))
    result = transformation(image)
    self.assertEqual(result.shape[1:], tile_sizes)

  @parameterized.parameters((1234, 2345, 2, 123), (2314, 1634, 1, 113))
  def test_it_gives_correct_num_tiles(self, height, width, channels, tile_size):
    image = tf.reshape(
        tf.range(height * width * channels), (height, width, channels))
    tile_sizes = (tile_size, tile_size, channels)
    stride = tile_size // 2
    transformation = image_operations.TileImage3D(
        sizes=tile_sizes, strides=(stride, stride))
    result = transformation(image)

    # Number of tiles should be correct
    num_on_height = (height - tile_size) // stride + 1
    num_on_width = (width - tile_size) // stride + 1
    print(num_on_height)
    print(num_on_width)
    self.assertEqual(result.shape[0], num_on_height * num_on_width)
    self.assertEqual(result.shape[-1], channels)


class TileNoOverlapTest(parameterized.TestCase):

  @parameterized.parameters(
      ((10, 20, 2), (5, 5), (8, 5, 5, 2)),
      ((11, 23, 3), (3, 4), (15, 3, 4, 3)),
  )
  def test_it_returns_expected_result(self, image_shape, tile_shape,
                                      all_tiles_shape):
    image = tf.reshape(tf.range(tf.reduce_prod(image_shape)), image_shape)
    tiles = image_operations.TileNoOverlap(
        input_shape=image_shape[:2], tile_shape=tile_shape)(
            image)
    self.assertEqual(tiles.shape, all_tiles_shape)


class MixRGBChannelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('red', 1, 0, 0, tf.constant([[0, 3, 6, 9]])),
      ('green', 0, 1, 0, tf.constant([[1, 4, 7, 10]])),
      ('blue', 0, 0, 1, tf.constant([[2, 5, 8, 11]])),
      ('mixed', 0.42, 0.33, 0.25, tf.constant([[0.83, 3.83, 6.83, 9.83]])),
      ('negative', -0.42, 0.33, 0.2, tf.constant([[0.73, 1.06, 1.39, 1.72]])),
  )
  def test_it_should_take_the_correct_channel(self, red, green, blue, expected):
    image = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]],
                     dtype=np.float32)
    transformation = image_operations.MixRGBChannels(
        red=red, green=green, blue=blue)
    result = transformation(image)
    expected = tf.cast(expected, tf.float32)

    tf.debugging.assert_near(result, expected)


class SelectChannelsTest(parameterized.TestCase):

  def test_it_raises_error_without_any_true_channel(self):
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    with self.assertRaisesRegex(ValueError, 'Expect at least one channel'):
      image_operations.SelectChannels()(image)

  @parameterized.parameters(
      (True, False, False, tf.constant([[[1], [4]], [[7], [10]]])),
      (True, True, False, tf.constant([[[1, 2], [4, 5]], [[7, 8], [10, 11]]])),
      (True, False, True, tf.constant([[[1, 3], [4, 6]], [[7, 9], [10, 12]]])))
  def test_it_returns_correct_output(self, red, green, blue, expected):
    image = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    select = image_operations.SelectChannels(red=red, green=green, blue=blue)
    result = select(image)
    tf.debugging.assert_equal(result, expected)


class TileImageFuncTest(parameterized.TestCase):

  def test_tiles_are_correct_size(self):
    """Expected Result:  all tiles are square with both dimensions = tile_size."""
    image = np.arange(2004 * 1004).reshape((2004, 1004))
    tile_size = 27
    tiles = image_operations.tile_image(image, tile_size=tile_size)
    for tile in tiles:
      self.assertEqual(tile.shape[0], tile_size)
      self.assertEqual(tile.shape[1], tile_size)

  @parameterized.parameters((1234, 2345, 123), (2314, 1634, 113))
  def test_tile_image_gives_correct_num_tiles(self, height, width, tile_size):
    """Test image_operations.tile_image returns expected number of tiles."""
    image = np.arange(height * width).reshape((height, width))
    tiles = image_operations.tile_image(image, tile_size)

    # check results
    slide = tile_size // 2
    num_on_height = (height - tile_size) // slide + 1
    num_on_width = (width - tile_size) // slide + 1
    self.assertEqual(num_on_height * num_on_width, tiles.shape[0])

  def test_tile_image_produces_correct_result(self):
    image = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                      [12, 13, 14, 15], [16, 17, 18, 19]])
    expected = np.array([[[0, 1, 2], [4, 5, 6], [8, 9, 10]],
                         [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
                         [[4, 5, 6], [8, 9, 10], [12, 13, 14]],
                         [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
                         [[8, 9, 10], [12, 13, 14], [16, 17, 18]],
                         [[9, 10, 11], [13, 14, 15], [17, 18, 19]]])
    result = image_operations.tile_image(image, tile_size=3)
    np.testing.assert_equal(result, expected)


class PadImageTest(parameterized.TestCase):

  def test_pad_image_right_bottom_returns_expected_result(self):
    image = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],
                      [12, 13, 14, 15], [16, 17, 18, 19]])
    tile_size = 3
    stride = (3, 2)
    expected = np.array([[0, 1, 2, 3, 0], [4, 5, 6, 7, 0], [8, 9, 10, 11, 0],
                         [12, 13, 14, 15, 0], [16, 17, 18, 19, 0],
                         [0, 0, 0, 0, 0]])
    result = image_operations.pad_image_right_bottom(image, tile_size, stride)
    np.testing.assert_equal(result, expected)

  @parameterized.named_parameters(
      (
          'padding_no_channel',
          tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
          2,
          (2, 2),
          tf.constant([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0]]),
      ),
      (
          'padding_with_channel',
          tf.constant([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]),
          2,
          (2, 2),
          tf.constant([[[1], [2], [3], [0]], [[4], [5], [6], [0]],
                       [[7], [8], [9], [0]], [[0], [0], [0], [0]]]),
      ),
      (
          'no_pad',
          tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
          2,
          (1, 1),
          tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
      ),
  )
  def test_pad_right_bottom_class_returns_expected_result(
      self, image, tile_size, strides, expected):
    result = image_operations.PadRightBottom(tile_size, strides)(image)
    tf.debugging.assert_equal(result, expected)


class ImageToTilesAndTilesToImageTest(absltest.TestCase):

  def test_tiles_to_image_reverses_image_to_tiles_with_true_padding(self):
    image_shape = (121, 134)
    image = np.arange(121 * 134).reshape(121, 134)
    tile_size = 13
    stride = (8, 9)
    tiles = image_operations.image_to_tiles(image, tile_size, stride)
    reconstructed = image_operations.tiles_to_image(tiles, image_shape, stride)
    np.testing.assert_equal(reconstructed, image)

  def test_image_to_tiles_with_false_padding_is_the_same_as_tileimage(self):
    image = np.arange(121 * 134).reshape(121, 134)
    tile_size = 13
    stride = (8, 9)
    tiles1 = image_operations.image_to_tiles(
        image, tile_size, stride, padding=False)
    tiles2 = image_operations.TileImage((tile_size, tile_size),
                                        stride)(image).numpy()
    np.testing.assert_equal(tiles1, tiles2)

  def test_image_to_tile_and_tile_image_class_return_same_output(self):
    """Tests image_operations.image_to_tiles and image_operations.TileImage.

    These two operate on numpy array and tf.Tensor, respectively. They should
    return consistent results.
    """
    height = 1234
    width = 2345
    tile_size = 23
    strides = (11, 13)
    tensor = tf.reshape(tf.range(height * width), (height, width))
    array = np.arange(height * width).reshape(height, width)

    # padding:
    tensor = image_operations.PadRightBottom(tile_size, strides)(tensor)
    array = image_operations.pad_image_right_bottom(array, tile_size, strides)

    # tiling:
    transform = image_operations.TileImage((tile_size, tile_size), strides)
    tensor_tiles = transform(tensor)
    array_tiles = image_operations.image_to_tiles(array, tile_size, strides)

    # check result
    np.testing.assert_equal(array_tiles, tensor_tiles.numpy())


class ResizeImageTest(absltest.TestCase):

  def test_it_returns_correct_shape(self):
    image = tf.reshape(tf.range(1234 * 2345), (1234, 2345))
    image = image[..., tf.newaxis]
    new_shape = (123, 234)
    new_image = image_operations.ResizeImage(new_shape)(image)
    self.assertEqual((new_image.shape[0], new_image.shape[1]), new_shape)


class ResizeLabelTest(absltest.TestCase):

  def test_it_returns_correct_shape(self):
    label = tf.reshape(tf.range(1234 * 2345), (1234, 2345))
    label = label[..., tf.newaxis]
    new_shape = (123, 234)
    new_label = image_operations.ResizeLabel(new_shape)(label)
    self.assertEqual((new_label.shape[0], new_label.shape[1]), new_shape)
    tf.debugging.assert_integer(new_label)


class CropTest(parameterized.TestCase):

  def test_it_behaves_as_tf_image_crop_to_bounding_box_default_offset(self):
    image = tf.reshape(tf.range(3 * 4 * 5), (3, 4, 5))
    shape = (2, 2)
    cropped1 = image_operations.Crop(shape)(image)
    cropped2 = tf.image.crop_to_bounding_box(image, 0, 0, *shape)
    tf.debugging.assert_equal(cropped1, cropped2)

  def test_it_behaves_as_tf_image_crop_to_bounding_box_non_default_offset(self):
    image = tf.reshape(tf.range(3 * 4 * 5), (3, 4, 5))
    shape = (2, 2)
    offset = (1, 1)
    cropped1 = image_operations.Crop(shape, offset)(image)
    cropped2 = tf.image.crop_to_bounding_box(image, *offset, *shape)
    tf.debugging.assert_equal(cropped1, cropped2)


_TEST_CLASS_LABELS_VALUES = {
    'unlabeled': 0,
    'building': 1,
    'vehicle': 2,
    'vegetation': 3,
    'road': 4,
    'water': 5,
    'power_infrastructure': 6,
    'light_pole': 7,
    'people': 8,
    'driveway': 9,
    'lawn': 10,
    'sidewalk': 11,
    'package': 12,
    'clutter': 13,
    'traffic_sign': 14,
    'traffic_light': 15,
    'winch_hook': 16,
    'winch_line': 17,
    'unknown': 18
}


class PreserveTwoContrastiveLabelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('one_label', ['building']),
      ('three_labels', ['building', 'road', 'vegetation']))
  def test_it_raises_error_wrong_num_classes(self, labels):
    with self.assertRaisesRegex(ValueError, 'Expecting exactly two classes'):
      image_operations.PreserveTwoContrastiveLabels(
          all_class_label_values=_TEST_CLASS_LABELS_VALUES, two_labels=labels)

  def test_preserve_two_labels_returns_expected_result(self):
    input_data = tf.constant(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    two_labels = ['road', 'vehicle']  # values [4, 2] are converted to [-1, 1]
    expected = tf.constant(
        [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    output = image_operations.PreserveTwoContrastiveLabels(
        all_class_label_values=_TEST_CLASS_LABELS_VALUES,
        two_labels=two_labels,
    )(
        input_data)
    tf.debugging.assert_equal(output, expected)


class PreserveSplitLabelsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('vehicle = 2', ['vehicle'],
       tf.constant([
           [
               [0],
               [0],
               [1],
               [0],
               [0],
           ],
           [
               [0],
               [0],
               [0],
               [0],
               [0],
           ],
       ])),
      ('[building, road] = [1, 4]', ['building', 'road'],
       tf.constant([
           [
               [0, 0],
               [1, 0],
               [0, 0],
               [0, 0],
               [0, 1],
           ],
           [
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
           ],
       ])),
      ('[building, people, road]=[1, 8, 4]', ['building', 'people', 'road'],
       tf.constant([
           [
               [0, 0, 0],
               [1, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 1],
           ],
           [
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 1, 0],
               [0, 0, 0],
           ],
       ])),
      ('water, unlabeled = 5, 0', ['water', 'unlabeled'],
       tf.constant([
           [
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
           ],
           [
               [1, 0],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
           ],
       ])),
  )
  def test_preserve_split_labels_returns_expected_result(
      self, label_list, expected):

    input_data = tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    output = image_operations.PreserveSplitLabels(_TEST_CLASS_LABELS_VALUES,
                                                  label_list)(
                                                      input_data)
    tf.debugging.assert_equal(output, expected)


class MergeLabelTest(absltest.TestCase):

  def test_merge_labels_returns_expected_result(self):
    x = tf.constant([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    y = image_operations.merge_labels(x)
    expected = tf.constant([1, 1, 3, 2])
    tf.debugging.assert_equal(y, expected)

  def test_merge_labels_revert_preserve_split_labels(self):
    """Note, this is true only in a certain condition.

    The condition is that there are no unwanted classes when calling
    `PreserveSplitLabels`.
    """
    x = tf.constant([[0, 1, 3, 2], [1, 4, 1, 2], [3, 0, 2, 1], [0, 0, 1, 2]])
    classes = ['unlabeled', 'building', 'vehicle', 'vegetation', 'road']
    f = image_operations.PreserveSplitLabels(_TEST_CLASS_LABELS_VALUES, classes)
    y = f(x)
    z = image_operations.merge_labels(y)
    tf.debugging.assert_equal(z, x)


class MaskTest(absltest.TestCase):

  def test_circular_mask_creates_a_filled_circle(self):
    result = image_operations.circular_mask(4, 5, 1.5)
    expected = np.array([
        [False, False, True, False, False],
        [False, True, True, True, False],
        [False, True, True, True, False],
        [False, False, True, False, False],
    ])
    np.testing.assert_equal(result, expected)

  def test_gaussian_mask_creates_a_gaussian_mask(self):
    result = image_operations.gaussian_mask(4, 5, 1.5)
    expected = np.array(
        [[0.24935221, 0.48567179, 0.60653066, 0.48567179, 0.24935221],
         [0.38889556, 0.75746513, 0.94595947, 0.75746513, 0.38889556],
         [0.38889556, 0.75746513, 0.94595947, 0.75746513, 0.38889556],
         [0.24935221, 0.48567179, 0.60653066, 0.48567179, 0.24935221]])
    np.testing.assert_allclose(result, expected)


if __name__ == '__main__':
  absltest.main()
