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

"""Tests for reservoir_nn.models.transformer."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from reservoir_nn.models import transformer
import tensorflow as tf

BuildReservoirParameters = parameterized.named_parameters(
    ('with_reservoir', True),
    ('without_reservoir', False),
)


class TransformerPreprocessingTest(absltest.TestCase):

  def test_patchcreation_should_produce_correct_shape(self):
    batch = 6
    patch_size = 8
    image_size = 72
    image_shape = (image_size, image_size, 7)
    num_patches = (image_size // patch_size)**2
    image_set = tf.range(batch * image_shape[0] * image_shape[1] *
                         image_shape[2])
    image_set = tf.reshape(image_set, (batch, *image_shape))
    model = transformer.PatchCreation(patch_size=patch_size)
    result = model(image_set)
    self.assertEqual(
        result.shape,
        (batch, num_patches, patch_size * patch_size * image_shape[2]))

  def test_patchencoder_should_produce_correct_shape(self):
    projection_dim = 5
    batch = 6
    patch_size = 8
    image_size = 72
    image_shape = (image_size, image_size, 4)
    num_patches = (image_size // patch_size)**2
    image_set = tf.range(batch * image_shape[0] * image_shape[1] *
                         image_shape[2])
    image_set = tf.reshape(image_set, (batch, *image_shape))
    patches = transformer.PatchCreation(patch_size=patch_size)(image_set)
    model = transformer.PatchEncoder(
        num_patches=num_patches,
        projection_dim=projection_dim,
    )
    result = model(patches)
    self.assertEqual(result.shape, (batch, num_patches, projection_dim))

  def test_positionalembedding_should_produce_correct_shape(self):
    batch = 6
    maxlen = 5
    projection_dim = 3
    token_size = 50
    data = tf.range(batch * maxlen)
    data = tf.reshape(data, (batch, maxlen))
    model = transformer.PositionalEmbedding(
        sequence_length=maxlen, vocab_size=token_size, embed_dim=projection_dim)
    result = model(data)
    self.assertEqual(result.shape, (batch, maxlen, projection_dim))

  def test_speech_feature_embedding_produces_correct_shape(self):
    num_hidden = 100
    model = transformer.SpeechFeatureEmbedding(embed_dim=num_hidden)
    batch_size = 64
    data = np.random.rand(batch_size, 2754, 129)
    result = model(data)
    # 345 = output of convolutional layer (condensed number of time steps)
    self.assertEqual(result.shape, (batch_size, 345, num_hidden))


class BuildReservoirTransformerTest(parameterized.TestCase):

  @BuildReservoirParameters
  def test_transformer_should_produce_correct_shape_in_segmentation(
      self, build_reservoir):
    reservoir = np.ones((42, 42))
    batch = 6
    num_output_channels = 10
    image_size = (8, 8, 4)
    image_set = tf.range(batch * image_size[0] * image_size[1] * image_size[2])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = transformer.reservoir_transformer(
        input_shape=image_size,
        num_output_channels=num_output_channels,
        reservoir_weight=reservoir,
        task='segmentation',
        build_reservoir=build_reservoir)
    result = model(image_set)
    self.assertEqual(result.shape,
                     (batch, image_size[0], image_size[1], num_output_channels))

  @BuildReservoirParameters
  def test_transformer_should_produce_correct_shape_in_classification(
      self, build_reservoir):
    reservoir = np.ones((42, 42))
    batch = 6
    num_output_channels = 10
    image_size = (2, 3, 4)
    image_set = tf.range(batch * image_size[0] * image_size[1] * image_size[2])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = transformer.reservoir_transformer(
        input_shape=image_size,
        num_output_channels=num_output_channels,
        reservoir_weight=reservoir,
        task='classification',
        build_reservoir=build_reservoir)
    result = model(image_set)
    self.assertEqual(result.shape, (batch, num_output_channels))

  @BuildReservoirParameters
  def test_transformer_should_produce_correct_shape_in_nlp(
      self, build_reservoir):
    reservoir = np.ones((42, 42))
    batch = 6
    num_output_channels = 2
    maxlen = 5
    token_size = 50
    input_shape = (maxlen,)
    data = tf.range(batch * maxlen)
    data = tf.reshape(data, (batch, maxlen))
    model = transformer.reservoir_transformer(
        input_shape=input_shape,
        num_output_channels=num_output_channels,
        reservoir_weight=reservoir,
        transformer_layers=2,
        token_size=token_size,
        task='nlp',
        build_reservoir=build_reservoir)
    result = model(data)
    self.assertEqual(result.shape, (batch, num_output_channels))

  @BuildReservoirParameters
  def test_transformer_should_produce_correct_shape_keyword_spotting(
      self, build_reservoir):
    reservoir = np.ones((42, 42))
    batch = 6
    output_classes = 11
    input_shape = (30, 3, 32)
    model = transformer.reservoir_transformer(
        input_shape=input_shape,
        num_output_channels=output_classes,
        reservoir_weight=reservoir,
        transformer_layers=2,
        task='keyword_spotting',
        build_reservoir=build_reservoir)
    data = np.ones((batch, 30, 3, 32))
    result = model(data)
    self.assertEqual(result.shape, (batch, output_classes))


class Seq2SeqTransformerTest(parameterized.TestCase):

  @BuildReservoirParameters
  def test_reservoir_seq2seq_should_produce_correct_shape_for_asr(
      self, build_reservoir):
    reservoir = np.ones((42, 42))
    maxlen = 200
    batch_size = 64
    num_classes = 34
    num_hidden = 100
    model, _, _ = transformer.reservoir_seq2seq_model(
        reservoir_weight=reservoir,
        sequence_length=maxlen,
        task='asr',
        token_size=num_classes,
        projection_dim=num_hidden,
        build_reservoir=build_reservoir)
    data = (np.random.rand(batch_size, 2754,
                           129), np.random.rand(batch_size, maxlen))
    result = model(data)
    self.assertEqual(result.shape, (batch_size, maxlen, num_classes))

  @BuildReservoirParameters
  def test_asr_encoder_produces_correct_shape(self, build_reservoir):
    reservoir = np.ones((42, 42))
    maxlen = 30
    batch_size = 64
    num_classes = 34
    num_hidden = 100
    model, _, _ = transformer.reservoir_seq2seq_model(
        reservoir_weight=reservoir,
        sequence_length=maxlen,
        task='asr',
        token_size=num_classes,
        projection_dim=num_hidden,
        build_reservoir=build_reservoir)
    data = (np.random.rand(batch_size, 2754,
                           129), np.random.rand(batch_size, maxlen))
    result = model.encoder(data[0])
    # 345 = output of convolutional layer (condensed number of time steps)
    self.assertEqual(result.shape, (batch_size, 345, num_hidden))

  @BuildReservoirParameters
  def test_asr_decoder_input_produces_correct_shape(self, build_reservoir):
    reservoir = np.ones((42, 42))
    maxlen = 30
    batch_size = 64
    num_classes = 34
    num_hidden = 100
    model, _, _ = transformer.reservoir_seq2seq_model(
        reservoir_weight=reservoir,
        sequence_length=maxlen,
        task='asr',
        token_size=num_classes,
        projection_dim=num_hidden,
        build_reservoir=build_reservoir)
    data = (np.random.rand(batch_size, 2754,
                           129), np.random.rand(batch_size, maxlen))
    decoder_input = model.dec_input(data[1])
    self.assertEqual(decoder_input.shape, (batch_size, maxlen, num_hidden))

  @BuildReservoirParameters
  def test_asr_decoder_produces_correct_shape(self, build_reservoir):
    reservoir = np.ones((42, 42))
    maxlen = 30
    batch_size = 64
    num_classes = 34
    num_hidden = 100
    model, _, _ = transformer.reservoir_seq2seq_model(
        reservoir_weight=reservoir,
        sequence_length=maxlen,
        task='asr',
        token_size=num_classes,
        projection_dim=num_hidden,
        build_reservoir=build_reservoir)
    data = (np.random.rand(batch_size, 2754,
                           129), np.random.rand(batch_size, maxlen))
    encoder_output = model.encoder(data[0])
    decoder_output = model.decode(encoder_output, data[1])
    self.assertEqual(decoder_output.shape, (batch_size, maxlen, num_hidden))


class ReservoirAttentionTest(absltest.TestCase):

  def test_reservoir_attention_produces_correct_shape_in_segmentation(self):
    reservoir = np.ones((42, 42))
    batch = 6
    num_output_channels = 10
    image_size = (2, 3, 4)
    image_set = tf.range(batch * image_size[0] * image_size[1] * image_size[2])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = transformer.reservoir_attention_model(
        input_shape=image_size,
        num_output_channels=num_output_channels,
        reservoir_weights=(reservoir, reservoir, reservoir),
        task='segmentation')
    result = model(image_set)
    self.assertEqual(result.shape,
                     (batch, image_size[0], image_size[1], num_output_channels))

  def test_reservoir_attention_produce_correct_shape_in_classification(self):
    reservoir = np.ones((42, 42))
    batch = 6
    num_output_channels = 10
    image_size = (2, 3, 4)
    image_set = tf.range(batch * image_size[0] * image_size[1] * image_size[2])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = transformer.reservoir_attention_model(
        input_shape=image_size,
        num_output_channels=num_output_channels,
        reservoir_weights=(reservoir, reservoir, reservoir),
        task='classification')
    result = model(image_set)
    self.assertEqual(result.shape, (batch, num_output_channels))

  def test_reservoir_attention_should_work(self):
    batch = 6
    num_output_channels = 10
    image_size = (2, 3, 4)
    num_heads = 9
    projection_dim = 12
    reservoir = np.random.uniform(size=(42, 42))
    attention_reservoir = np.random.uniform(size=(num_heads, projection_dim))
    image_set = tf.range(batch * image_size[0] * image_size[1] * image_size[2])
    image_set = tf.reshape(image_set, (batch, *image_size))
    model = transformer.reservoir_attention_model(
        input_shape=image_size,
        num_output_channels=num_output_channels,
        reservoir_weights=(reservoir, reservoir, reservoir),
        task='classification',
        num_heads=num_heads,
        projection_dim=projection_dim,
        attention_reservoir_weight=attention_reservoir)
    result = model(image_set)
    self.assertEqual(result.shape, (batch, num_output_channels))


class GPTTest(parameterized.TestCase):

  @BuildReservoirParameters
  def test_gpt_model_outputs_correct_shape(self, build_reservoir):
    weights = np.ones((42, 42))
    batch_size = 5
    vocab_size = 50
    maxlen = 20
    embed_dim = 10
    model = transformer.mini_gpt_reservoir_model(
        reservoir_weight=weights,
        embed_dim=embed_dim,
        maxlen=maxlen,
        vocab_size=vocab_size,
        build_reservoir=build_reservoir)

    x_data = np.ones((batch_size, maxlen))
    result = model.predict(x_data)
    self.assertEqual(result.shape, (batch_size, maxlen, vocab_size))

  @BuildReservoirParameters
  def test_gpt_model_more_than_one_layer(self, build_reservoir):
    weights = np.ones((42, 42))
    batch_size = 5
    vocab_size = 50
    maxlen = 20
    embed_dim = 10
    num_layers = 4
    model = transformer.mini_gpt_reservoir_model(
        reservoir_weight=weights,
        embed_dim=embed_dim,
        maxlen=maxlen,
        vocab_size=vocab_size,
        num_transformer_layers=num_layers,
        build_reservoir=build_reservoir)

    # Check that model runs and returns the correct shape.
    x_data = np.ones((batch_size, maxlen))
    result = model.predict(x_data)
    self.assertEqual(result.shape, (batch_size, maxlen, vocab_size))

    # Check that all layers have been created.
    for i in range(num_layers):
      model.get_layer(f'transformer_block_{i}')


if __name__ == '__main__':
  absltest.main()
