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

"""Transformer models."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from reservoir_nn.keras import reservoir_registry
import tensorflow as tf


class PatchCreation(tf.keras.layers.Layer):
  """Patch creation layer for transformer."""

  def __init__(self, patch_size: int):
    super(PatchCreation, self).__init__()
    self.patch_size = patch_size

  def call(self, images: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, self.patch_size, self.patch_size, 1],
        strides=[1, self.patch_size, self.patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches


class PatchEncoder(tf.keras.layers.Layer):
  """Patch encoder layer for transformer."""

  def __init__(self, num_patches: int, projection_dim: int):
    super(PatchEncoder, self).__init__()
    self.num_patches = num_patches
    self.projection = tf.keras.layers.Dense(units=projection_dim)
    self.position_embedding = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim)

  def call(self, patch: tf.Tensor) -> tf.Tensor:
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    encoded = self.projection(patch) + self.position_embedding(positions)
    return encoded


class PositionalEmbedding(tf.keras.layers.Layer):
  """Position and Token Embeddings for reservoir-based Transformer."""

  def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int,
               **kwargs):
    super(PositionalEmbedding, self).__init__(**kwargs)
    self.token_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim)
    self.position_embedding = tf.keras.layers.Embedding(
        input_dim=sequence_length, output_dim=embed_dim)
    self.sequence_length = sequence_length
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    length = tf.shape(inputs)[-1]
    positions = tf.range(start=0, limit=length, delta=1)
    embedded_tokens = self.token_embedding(inputs)
    embedded_positions = self.position_embedding(positions)
    return embedded_tokens + embedded_positions

  def compute_mask(self: tf.Tensor, inputs, mask=None) -> tf.Tensor:
    return tf.math.not_equal(inputs, 0)  # pytype: disable=wrong-arg-types


class SpeechFeatureEmbedding(tf.keras.layers.Layer):
  """Speech Feature Embedding for reservoir-based Transformer."""

  def __init__(self, embed_dim: int = 64):
    super().__init__()
    self.conv1 = tf.keras.layers.Conv1D(
        embed_dim, 11, strides=2, padding='same', activation='relu')
    self.conv2 = tf.keras.layers.Conv1D(
        embed_dim, 11, strides=2, padding='same', activation='relu')
    self.conv3 = tf.keras.layers.Conv1D(
        embed_dim, 11, strides=2, padding='same', activation='relu')

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    x = self.conv1(inputs)
    x = self.conv2(x)
    return self.conv3(x)


class ReservoirTransformerEncoder(tf.keras.layers.Layer):
  """Encoder model for Seq2Seq Reservoir-based Transformer."""

  def __init__(self,
               embed_dim: int,
               num_heads: int,
               reservoir_base: str,
               reservoir_params: Dict[str, Any],
               dense_dim: int = 2048,
               build_reservoir: bool = False,
               **kwargs):
    super(ReservoirTransformerEncoder, self).__init__(**kwargs)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dense_dim = dense_dim
    self.build_reservoir = build_reservoir
    self.attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)

    if self.build_reservoir:
      self.reservoir_top = tf.keras.layers.Dense(
          reservoir_params['weight'].shape[0])
      self.reservoir_layer = reservoir_registry.get_reservoir(reservoir_base)(
          **reservoir_params)
      self.reservoir_bottom = tf.keras.layers.Dense(embed_dim)
    else:
      self.dense_proj = tf.keras.Sequential([
          tf.keras.layers.Dense(dense_dim, activation='relu'),
          tf.keras.layers.Dense(embed_dim),
      ])

    self.layernorm_1 = tf.keras.layers.LayerNormalization()
    self.layernorm_2 = tf.keras.layers.LayerNormalization()
    self.supports_masking = True

  def call(self, inputs: tf.Tensor, mask=None) -> tf.Tensor:
    if mask is not None:
      padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype='int32')
    else:
      padding_mask = None
    attention_output = self.attention(
        query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
    proj_input = self.layernorm_1(inputs + attention_output)
    if self.build_reservoir:
      proj_output = self.reservoir_top(proj_input)
      proj_output = self.reservoir_layer(proj_output)
      proj_output = self.reservoir_bottom(proj_output)
    else:
      proj_output = self.dense_proj(proj_input)
    return self.layernorm_2(proj_input + proj_output)


class ReservoirTransformerDecoder(tf.keras.layers.Layer):
  """Decoder model for Seq2Seq Transformer."""

  def __init__(self,
               embed_dim: int,
               num_heads: int,
               reservoir_base: str,
               reservoir_params: Dict[str, Any],
               latent_dim: int = 2048,
               build_reservoir: bool = False,
               **kwargs):
    super(ReservoirTransformerDecoder, self).__init__(**kwargs)
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.latent_dim = latent_dim
    self.build_reservoir = build_reservoir

    self.attention_1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)
    self.attention_2 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)

    if self.build_reservoir:
      self.reservoir_top = tf.keras.layers.Dense(
          reservoir_params['weight'].shape[0])
      self.reservoir_layer = reservoir_registry.get_reservoir(reservoir_base)(
          **reservoir_params)
      self.reservoir_bottom = tf.keras.layers.Dense(embed_dim)
    else:
      self.dense_proj = tf.keras.Sequential([
          tf.keras.layers.Dense(latent_dim, activation='relu'),
          tf.keras.layers.Dense(embed_dim),
      ])
    self.layernorm_1 = tf.keras.layers.LayerNormalization()
    self.layernorm_2 = tf.keras.layers.LayerNormalization()
    self.layernorm_3 = tf.keras.layers.LayerNormalization()
    self.supports_masking = True

  def call(self,
           inputs: tf.Tensor,
           encoder_outputs: tf.Tensor,
           mask=None) -> tf.Tensor:
    causal_mask = self.get_causal_attention_mask(inputs)
    if mask is not None:
      padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype='int32')
      padding_mask = tf.minimum(padding_mask, causal_mask)
    else:
      padding_mask = None

    attention_output_1 = self.attention_1(
        query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
    out_1 = self.layernorm_1(inputs + attention_output_1)

    attention_output_2 = self.attention_2(
        query=out_1,
        value=encoder_outputs,
        key=encoder_outputs,
        attention_mask=padding_mask,
    )
    out_2 = self.layernorm_2(out_1 + attention_output_2)

    if self.build_reservoir:
      proj_output = self.reservoir_top(out_2)
      proj_output = self.reservoir_layer(proj_output)
      proj_output = self.reservoir_bottom(proj_output)
    else:
      proj_output = self.dense_proj(out_2)
    return self.layernorm_3(out_2 + proj_output)

  def get_causal_attention_mask(self, inputs: tf.Tensor) -> tf.Tensor:
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype='int32')
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1),
         tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    return tf.tile(mask, mult)


class ReservoirTransformerASR(tf.keras.Model):
  """Seq2Seq Transformer with reservoir for automatic speech recognition."""

  def __init__(self,
               reservoir_base: str = 'DenseReservoir',
               reservoir_params: Optional[Dict[str, Any]] = None,
               embed_dim: int = 64,
               num_heads: int = 2,
               target_sequence_length: int = 100,
               num_encoder_layers: int = 4,
               num_decoder_layers: int = 1,
               token_size: int = 10,
               build_reservoir: bool = False):
    """Reservoir-based Transformer for ASR tasks.

    Args:
      reservoir_base: the reservoir base to use. Default is 'DenseReservoir'.
      reservoir_params: the parameters to initialize the reservoir_base. (Any
        field provided MUST be a Correct argument for the reservoir base, e.g.
        common options include {
        'recurrence_degree': 3,
        'keep_memory': True,
        'trainable_reservoir': True,
        'use_bias': True,
        'activation_within_recurrence': True,
        'kernel_local_learning': 'hebbian',
        'kernel_local_learning_params': {'eta': 0.1},
        'recurrent_kernel_local_learning': 'hebbian',
        'recurrent_kernel_local_learning_params': {'eta': 0.1},
        'state_discount': 1.0, }. If variable not included in the params, the
          default values are used.)
      embed_dim: the embedding dimension for multiheaded attention.
      num_heads: the number of heads in multiheaded attention.
      target_sequence_length: maximum length for output sequence.
      num_encoder_layers: number of encoder layers.
      num_decoder_layers: number of decoder layers.
      token_size: size of the patches to be extract from the input images if the
        task is vision; size of the vocab to be extracted from the sentences if
        the task is a natural language processing task.
      build_reservoir: A reservoir sandwich is build if this is true. When
        false, the model has a more standard transformer architecture.
    """
    super().__init__()
    self.loss_metric = tf.keras.metrics.Mean(name='loss')
    self.num_encoder_layers = num_encoder_layers
    self.num_decoder_layers = num_decoder_layers
    self.target_sequence_length = target_sequence_length
    self.token_size = token_size

    self.enc_input = SpeechFeatureEmbedding(embed_dim=embed_dim)
    self.dec_input = PositionalEmbedding(
        vocab_size=token_size,
        sequence_length=target_sequence_length,
        embed_dim=embed_dim)

    encoder_layers = [self.enc_input]
    for _ in range(num_encoder_layers):
      encoder_layers.append(
          ReservoirTransformerEncoder(
              reservoir_base=reservoir_base,
              num_heads=num_heads,
              embed_dim=embed_dim,
              reservoir_params=reservoir_params,
              build_reservoir=build_reservoir))
    self.encoder = tf.keras.Sequential(encoder_layers)

    for i in range(num_decoder_layers):
      setattr(
          self,
          f'reservoir_dec_layer_{i}',
          ReservoirTransformerDecoder(
              embed_dim=embed_dim,
              num_heads=num_heads,
              reservoir_base=reservoir_base,
              reservoir_params=reservoir_params,
              build_reservoir=build_reservoir),
      )

    self.classifier = tf.keras.layers.Dense(token_size)

  def decode(self, enc_out: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    y = self.dec_input(target)
    for i in range(self.num_decoder_layers):
      y = getattr(self, f'reservoir_dec_layer_{i}')(
          inputs=y, encoder_outputs=enc_out, mask=None)
    return y

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    source = inputs[0]
    target = inputs[1]
    x = self.encoder(source)
    y = self.decode(x, target)
    return self.classifier(y)

  @property
  def metrics(self):
    return [self.loss_metric]

  def train_step(self, batch):
    """Processes one batch inside model.fit()."""
    source = batch['source']
    target = batch['target']
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]
    with tf.GradientTape() as tape:
      preds = self([source, dec_input])
      one_hot = tf.one_hot(dec_target, depth=self.token_size)
      mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
      loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.loss_metric.update_state(loss)
    return {'loss': self.loss_metric.result()}

  def test_step(self, batch):
    source = batch['source']
    target = batch['target']
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]
    preds = self([source, dec_input])
    one_hot = tf.one_hot(dec_target, depth=self.token_size)
    mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
    loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
    self.loss_metric.update_state(loss)
    return {'loss': self.loss_metric.result()}

  def generate(self, source, target_start_token_idx):
    """Performs inference over one batch of inputs using greedy decoding."""
    bs = tf.shape(source)[0]
    enc = self.encoder(source)
    dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
    dec_logits = []
    for _ in range(self.target_sequence_length - 1):
      dec_out = self.decode(enc, dec_input)
      logits = self.classifier(dec_out)
      logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
      last_logit = tf.expand_dims(logits[:, -1], axis=-1)
      dec_logits.append(last_logit)
      dec_input = tf.concat([dec_input, last_logit], axis=-1)
    return dec_input


def reservoir_seq2seq_model(
    reservoir_weight: np.ndarray,
    sequence_length: int,
    reservoir_base: str = 'DenseReservoir',
    reservoir_params: Optional[Dict[str, Any]] = None,
    projection_dim: int = 256,
    num_heads: int = 8,
    token_size: int = 2048,
    model_base: str = 'transformer',
    task: str = 'nlp',
    build_reservoir=False
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
  """Builds a Seq2Seq model with a reservoir-based Encoder and a Decoder.

  Args:
    reservoir_weight:  weights to use in the reservoir part of the model
    sequence_length:  the maximum length of the sequence.
    reservoir_base: the reservoir base to use. Default is 'DenseReservoir'.
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a Correct argument for the reservoir base, e.g.
      common options include {
      'recurrence_degree': 3,
      'keep_memory': True,
      'trainable_reservoir': True,
      'use_bias': True,
      'activation_within_recurrence': True,
      'kernel_local_learning': 'hebbian',
      'kernel_local_learning_params': {'eta': 0.1},
      'recurrent_kernel_local_learning': 'hebbian',
      'recurrent_kernel_local_learning_params': {'eta': 0.1},
      'state_discount': 1.0, }. If variable not included in the params, the
        default values are used.)
    projection_dim: the projection dimension for the embedding and attention.
    num_heads: the number of heads in multiheaded attention.
    token_size: size of the patches to be extract from the input images if the
      task is vision; size of the vocab to be extracted from the sentences if
      the task is a natural language processing task.
    model_base: the base for decoder and encoder models. Options: transformer.
    task: which task this model is used for. Options: nlp, asr.
    build_reservoir: A reservoir sandwich is build if this is true. When false,
      the returned model has a more standard transformer architecture.

  Returns:
    model: a reservoir seq2seq model instance.
    encoder: a reservoir encoder model instance.
    decoder: a reservoir decoder model instance.

  Raises:
    ValueError: if task not in accepted tasks (segmentation, classification).
  """

  if task not in ['nlp', 'asr']:
    raise ValueError(f'Task not defined in accepted tasks (nlp). Got {task}')

  if reservoir_params is None:
    reservoir_params = {}
  reservoir_params['weight'] = reservoir_weight
  reservoir_params['activation'] = 'relu'

  if task == 'asr':
    return (ReservoirTransformerASR(
        reservoir_base=reservoir_base,
        reservoir_params=reservoir_params,
        embed_dim=projection_dim,
        num_heads=num_heads,
        target_sequence_length=sequence_length,
        token_size=token_size,
        build_reservoir=build_reservoir), None, None)  # pytype: disable=bad-return-type  # typed-keras

  if model_base == 'transformer':
    reservoir_encoder = ReservoirTransformerEncoder
    reservoir_decoder = ReservoirTransformerDecoder

  encoder_inputs = tf.keras.Input(
      shape=(None,), dtype='int64', name='encoder_inputs')
  encoder_features = PositionalEmbedding(sequence_length, token_size,
                                         projection_dim)(
                                             encoder_inputs)

  encoder_outputs = reservoir_encoder(
      embed_dim=projection_dim,
      num_heads=num_heads,
      reservoir_base=reservoir_base,
      reservoir_params=reservoir_params,
      build_reservoir=build_reservoir)(
          encoder_features)
  encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

  decoder_inputs = tf.keras.Input(
      shape=(None,), dtype='int64', name='decoder_inputs')
  encoded_seq_inputs = tf.keras.Input(
      shape=(None, projection_dim), name='decoder_state_inputs')
  decoder_features = PositionalEmbedding(
      sequence_length=sequence_length,
      vocab_size=token_size,
      embed_dim=projection_dim)(
          decoder_inputs)
  decoder_features = reservoir_decoder(
      embed_dim=projection_dim,
      num_heads=num_heads,
      reservoir_base=reservoir_base,
      reservoir_params=reservoir_params,
      build_reservoir=build_reservoir)(decoder_features, encoded_seq_inputs)
  decoder_features = tf.keras.layers.Dropout(0.5)(decoder_features)
  decoder_outputs = tf.keras.layers.Dense(
      token_size, activation='softmax')(
          decoder_features)
  decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs],
                           decoder_outputs)

  decoder_outputs = decoder([decoder_inputs, encoder_outputs])
  seq2seq_model = tf.keras.Model([encoder_inputs, decoder_inputs],
                                 decoder_outputs,
                                 name='seq2seq')
  return seq2seq_model, encoder, decoder


def reservoir_transformer(
    input_shape: Tuple[int, ...],
    reservoir_weight: np.ndarray,
    num_output_channels: int,
    reservoir_base: str = 'DenseReservoir',
    reservoir_params: Optional[Dict[str, Any]] = None,
    projection_dim: int = 64,
    num_heads: int = 4,
    image_size: int = 72,
    token_size: int = 6,
    transformer_layers: int = 8,
    final_activation: str = 'sigmoid',
    downsample_rate: int = 1,
    task: str = 'segmentation',
    transformer_dropout_rate: float = 0.1,
    output_dropout_rate: float = 0.5,
    build_reservoir: bool = False,
) -> tf.keras.Model:
  """Builds a Transformer with reservoir in each transformer block.

  Args:
    input_shape:  tuple describing the shape of the input (e.g., (32,32, 3)).
    reservoir_weight:  weights to use in the reservoir part of the model
    num_output_channels:  the output dimension.
    reservoir_base: the reservoir base to use. Default is 'DenseReservoir'.
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a Correct argument for the reservoir base, e.g.
      common options include {
      'recurrence_degree': 3,
      'keep_memory': True,
      'trainable_reservoir': True,
      'use_bias': True,
      'activation_within_recurrence': True,
      'kernel_local_learning': 'hebbian',
      'kernel_local_learning_params': {'eta': 0.1},
      'recurrent_kernel_local_learning': 'hebbian',
      'recurrent_kernel_local_learning_params': {'eta': 0.1},
      'state_discount': 1.0, }. If variable not included in the params, the
        default values are used.)
    projection_dim: the projection dimension for multiheaded attention.
    num_heads: the number of heads in multiheaded attention.
    image_size: the size that we resize input images to (if the task is vision).
    token_size: size of the patches to be extract from the input images if the
      task is vision; size of the vocab to be extracted from the sentences if
      the task is a natural language processing task.
    transformer_layers: number of transformer layers.
    final_activation: 'sigmoid', 'softmax', or 'tanh'.
    downsample_rate: the rate used to constrain input size to avoid OOM.
    task: which task this model is used for (options includes: 'segmentation',
      'classification', 'nlp')
    transformer_dropout_rate: dropout rate within transformer blocks.
    output_dropout_rate: dropout weight applied to task-specific output layers.
    build_reservoir: A reservoir sandwich is build if this is true. When false,
      the returned model has a more standard transformer architecture.

  Returns:
    model: a reservoir transformer model instance.

  Raises:
    ValueError: if task not in accepted tasks (segmentation, classification).
  """

  if task not in ['segmentation', 'classification', 'nlp', 'keyword_spotting']:
    raise ValueError(
        'Task not defined in accepted tasks (segmentation, classification, ' +
        f'nlp, keyword_spotting). Got {task}')

  if reservoir_params is None:
    reservoir_params = {}
  reservoir_params['weight'] = reservoir_weight

  inputs = tf.keras.layers.Input(shape=input_shape)

  # Goal of this step is to get inputs with shape (sequence_length, features)
  # where features is the features associated with a particular step in
  # sequence.
  if task == 'nlp':
    # input_shape = (200,) = (sequence_length,)
    sequence_length = input_shape[0]
    embedding_layer = PositionalEmbedding(
        sequence_length=sequence_length,
        vocab_size=token_size,
        embed_dim=projection_dim)
    encoded_tokens = embedding_layer(inputs)
  elif task == 'keyword_spotting':
    # input_shape = (30, 3, 32) = (Frames, Length, Mel bins)
    sequence_length = input_shape[0]
    reshaped_input = tf.keras.layers.Reshape(
        (-1, input_shape[1] * input_shape[2]), name='input_flatten')(
            inputs)
    encoded_tokens = PatchEncoder(input_shape[0], projection_dim)(
        reshaped_input)
  else:
    # e.g. input_shape = (32, 32, 3) = (image_dim, image_dim, rgb)
    # Augment data.
    num_patches = (image_size // token_size)**2
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Normalization(),
            tf.keras.layers.experimental.preprocessing.Resizing(
                image_size, image_size),
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(
                factor=0.02),
            tf.keras.layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2),
        ],
        name='data_augmentation',
    )
    augmented = data_augmentation(inputs)

    # Create and Encode patches.
    patches = PatchCreation(token_size)(augmented)
    encoded_tokens = PatchEncoder(num_patches, projection_dim)(patches)

  # Create multiple layers of the Transformer block.
  for _ in range(transformer_layers):

    # Layer normalization 1.
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_tokens)

    # Create a multi-head attention layer.
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=transformer_dropout_rate)(x1, x1)

    # Skip connection 1.
    x2 = tf.keras.layers.Add()([attention_output, encoded_tokens])

    # Layer normalization 2.
    x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

    if build_reservoir:
      # Reservoir sandwich layer.
      x3 = tf.keras.layers.Dense(
          reservoir_weight.shape[0], activation=tf.nn.gelu)(
              x3)
      x3 = tf.keras.layers.Dropout(rate=transformer_dropout_rate)(x3)

      x3 = reservoir_registry.get_reservoir(reservoir_base)(**reservoir_params)(
          x3)

      x3 = tf.keras.layers.Dense(
          units=projection_dim, activation=tf.nn.gelu)(
              x3)
      x3 = tf.keras.layers.Dropout(rate=transformer_dropout_rate)(x3)
    else:
      x3 = tf.keras.layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
      x3 = tf.keras.layers.Dropout(rate=transformer_dropout_rate)(x3)
      x3 = tf.keras.layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)
      x3 = tf.keras.layers.Dropout(rate=transformer_dropout_rate)(x3)

    # Skip connection 2.
    encoded_tokens = tf.keras.layers.Add()([x3, x2])

  # Create a [batch_size, projection_dim] tensor.
  representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
      encoded_tokens)

  # Final layers.
  features = representation
  if task == 'segmentation':
    features = tf.keras.layers.Flatten()(features)
    features = tf.keras.layers.Dropout(rate=output_dropout_rate)(features)
    dense_shape = (int(input_shape[0] / downsample_rate),
                   int(input_shape[1] / downsample_rate), num_output_channels)
    outputs = tf.keras.layers.Dense(
        units=tf.reduce_prod(dense_shape), activation=final_activation)(
            features)
    outputs = tf.keras.layers.Reshape(dense_shape)(outputs)
    outputs = tf.keras.layers.UpSampling2D(downsample_rate)(outputs)
  elif task in {'classification', 'keyword_spotting'}:
    features = tf.keras.layers.Flatten()(features)
    features = tf.keras.layers.Dropout(rate=output_dropout_rate)(features)
    features = tf.keras.layers.Dense(
        units=2048, activation=tf.nn.gelu)(
            features)
    features = tf.keras.layers.Dropout(rate=output_dropout_rate)(features)
    features = tf.keras.layers.Dense(
        units=1024, activation=tf.nn.gelu)(
            features)
    features = tf.keras.layers.Dropout(rate=output_dropout_rate)(features)
    outputs = tf.keras.layers.Dense(
        units=num_output_channels, activation=final_activation)(
            features)
  elif task == 'nlp':
    features = tf.keras.layers.GlobalAveragePooling1D()(features)
    features = tf.keras.layers.Dropout(rate=output_dropout_rate)(features)
    features = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)(features)
    features = tf.keras.layers.Dropout(rate=output_dropout_rate)(features)
    outputs = tf.keras.layers.Dense(
        units=num_output_channels, activation=final_activation)(
            features)

  # Create the tf.keras model.
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


def _attention_reservoir_kernel_initializer(weight: np.ndarray):
  """Kernel initializer for attention layer.

  Args:
    weight: reservoir weight.

  Returns:
    callable function to initialize the weight.
  """

  def _attention_initializer(shape, dtype, weight):
    if shape[0] == weight.shape[0] and shape[1] == weight.shape[1]:
      rotate_dim = -1
    else:
      rotate_dim = 0
    weights = [tf.expand_dims(weight, axis=rotate_dim)] * shape[rotate_dim]
    weight = tf.concat(weights, rotate_dim)
    return tf.cast(tf.ensure_shape(weight, shape), dtype)

  return lambda shape, dtype: _attention_initializer(shape, dtype, weight)


def reservoir_attention_model(
    input_shape: Tuple[int, ...],
    reservoir_weights: Tuple[np.ndarray, ...],
    num_output_channels: int,
    reservoir_base: str = 'DenseReservoir',
    reservoir_params: Optional[Dict[str, Any]] = None,
    projection_dim: int = 64,
    num_heads: int = 4,
    final_activation: str = 'sigmoid',
    downsample_rate: int = 1,
    task: str = 'segmentation',
    attention_reservoir_weight: Optional[np.ndarray] = None,
) -> tf.keras.Model:
  """Builds a multi-reservoir model modulated by a multihead attention.

  Args:
    input_shape:  tuple describing the shape of the input (e.g., (32,32, 3)).
    reservoir_weights:  reservoir weights to use in each reservoir.
    num_output_channels:  how many output channels to use.
    reservoir_base: the reservoir base to use. Default is 'DenseReservoir'.
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a Correct argument for the reservoir base, e.g.
      common options include {
      'recurrence_degree': 3,
      'keep_memory': True,
      'trainable_reservoir': True,
      'use_bias': True,
      'activation_within_recurrence': True,
      'kernel_local_learning': 'hebbian',
      'kernel_local_learning_params': {'eta': 0.1},
      'recurrent_kernel_local_learning': 'hebbian',
      'recurrent_kernel_local_learning_params': {'eta': 0.1},
      'state_discount': 1.0, }. If variable not included in the params, the
        default values are used.)
    projection_dim: the projection dimension for multiheaded attention.
    num_heads: the number of heads in multiheaded attention.
    final_activation: 'sigmoid', 'softmax', or 'tanh'.
    downsample_rate: the rate used to constrain input size to avoid OOM.
    task: which task this model is used for (options includes: 'segmentation',
      'classification')
    attention_reservoir_weight: reservoir weight to use in attention. If none
      specified, default multi-headed attention is used.

  Returns:
    model: an attention reservoir model instance.

  Raises:
    ValueError: if task not in accepted tasks (segmentation, classification).
  """

  if task not in ['segmentation', 'classification']:
    raise ValueError(
        f'Task not defined in accepted tasks (segmentation, classification). Got {task}'
    )

  if reservoir_params is None:
    reservoir_params = {}

  inputs = tf.keras.Input(shape=input_shape)

  x = tf.keras.layers.Conv2D(
      32, 3, strides=downsample_rate, padding='same')(
          inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  # Create a bank of reservoirs.
  reservoir_bank = []
  for w in reservoir_weights:
    reservoir_params['weight'] = w
    sandwich_top = tf.keras.layers.Dense(w.shape[0], activation='elu')(x)
    reservoir_layer = reservoir_registry.get_reservoir(reservoir_base)(
        **reservoir_params)(
            sandwich_top)
    sandwich_bottom = tf.keras.layers.Dense(
        units=projection_dim, activation=tf.nn.gelu)(
            reservoir_layer)
    reservoir_sandwich = tf.keras.layers.Dropout(rate=0.1)(sandwich_bottom)

    # Add layer normalization
    reservoir_sandwich = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        reservoir_sandwich)

    reservoir_bank.append(reservoir_sandwich)

  # Concat reservoir activations
  multi_reservoir_layer = tf.keras.layers.concatenate(reservoir_bank)

  # Create a multi-head attention layer.
  if attention_reservoir_weight is None:
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim,
        dropout=0.1)(multi_reservoir_layer, multi_reservoir_layer)
  else:
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        kernel_initializer=_attention_reservoir_kernel_initializer(
            attention_reservoir_weight),
        dropout=0.1,
    )
    attention_output = attention_layer(multi_reservoir_layer,
                                       multi_reservoir_layer)

  features = tf.keras.layers.Add()([attention_output, multi_reservoir_layer])
  features = tf.keras.layers.LayerNormalization(epsilon=1e-6)(features)

  # Finally project it out with a dense layer.
  features = tf.keras.layers.UpSampling2D(downsample_rate)(features)
  if task == 'segmentation':
    outputs = tf.keras.layers.Conv2D(
        num_output_channels, 3, activation=final_activation, padding='same')(
            features)
  elif task == 'classification':
    outputs = tf.keras.layers.Flatten()(features)
    outputs = tf.keras.layers.Dense(
        units=num_output_channels, activation=final_activation)(
            outputs)

  # Create the tf.keras model.
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


class ReservoirTransformerBlockGPT(tf.keras.layers.Layer):
  """A reservoir-base transformer block for GPT models.

  This is very similar to ReservoirTransformerEncoder. The key difference is
  that it includes causal masking, which means that for each token, the model
  only
  considers previous tokens (like in a transformer decoder). However, unlike
  the transformer decoders above, where the input and output types are different
  (different languages in translation; speech vs. text in ASR), GPT output types
  are the same as their input types (English tokens).
  """

  def __init__(self,
               embed_dim: int,
               num_heads: int,
               rate: float = 0.1,
               reservoir_params: Optional[Dict[str, Any]] = None,
               reservoir_base: str = 'DenseReservoir',
               build_reservoir=False,
               name=None):
    super(ReservoirTransformerBlockGPT, self).__init__(name=name)
    self.build_reservoir = build_reservoir

    self.att = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
    if self.build_reservoir:
      self.reservoir_top = tf.keras.layers.Dense(
          reservoir_params['weight'].shape[0])
      self.reservoir_layer = reservoir_registry.get_reservoir(reservoir_base)(
          **reservoir_params)
      self.reservoir_bottom = tf.keras.layers.Dense(embed_dim)
    else:
      self.ffn = tf.keras.Sequential([
          tf.keras.layers.Dense(embed_dim, activation='relu'),
          tf.keras.layers.Dense(embed_dim),
      ])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, inputs: tf.Tensor):
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len,
                                             tf.bool)
    attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
    attention_output = self.dropout1(attention_output)
    out1 = self.layernorm1(inputs + attention_output)
    if self.build_reservoir:
      out2 = self.reservoir_top(out1)
      out2 = self.reservoir_layer(out2)
      out2 = self.reservoir_bottom(out2)
    else:
      out2 = self.ffn(out1)
    out2 = self.dropout2(out2)
    return self.layernorm2(out1 + out2)

  def causal_attention_mask(self,
                            batch_size: tf.Tensor,
                            n_dest: tf.Tensor,
                            n_src: tf.Tensor,
                            dtype: tf.dtypes.DType = tf.bool) -> tf.Tensor:
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1),
         tf.constant([1, 1], dtype=tf.int32)], 0)
    return tf.tile(mask, mult)


def mini_gpt_reservoir_model(
    reservoir_weight: np.ndarray,
    vocab_size: int = 20000,
    maxlen: int = 80,
    embed_dim: int = 256,
    num_heads: int = 2,
    num_transformer_layers: int = 1,
    reservoir_params: Optional[Dict[str, Any]] = None,
    reservoir_base: str = 'DenseReservoir',
    build_reservoir: bool = False,
) -> tf.keras.Model:
  """Builds a mini GPT model.

  This model only has 1 decoder block, while GPT-2 has 12.

  Args:
    reservoir_weight: weights to use in the reservoir of the model.
    vocab_size: number of tokens in the vocabulary.
    maxlen: maximum sequencnce length (number of tokens in a sentence).
    embed_dim: size of word embeddings to train for each vocab entry (input to
      LSTM cells).
    num_heads: the number of attention heads in transformer block layers.
    num_transformer_layers: the number of transformer blocks in the model.
    reservoir_params: the parameters to initialize the reservoir_base. (Any
      field provided MUST be a Correct argument for the reservoir base, e.g.
      common options include {
      'recurrence_degree': 3,
      'keep_memory': True,
      'trainable_reservoir': True,
      'use_bias': True,
      'activation_within_recurrence': True,
      'kernel_local_learning': 'hebbian',
      'kernel_local_learning_params': {'eta': 0.1},
      'recurrent_kernel_local_learning': 'hebbian',
      'recurrent_kernel_local_learning_params': {'eta': 0.1},
      'state_discount': 1.0, }. If variable not included in the params, the
        default values are used.)
    reservoir_base: the reservoir base to use. Must be in RNN_LAYERS.
    build_reservoir: A reservoir sandwich is build if this is true. When false,
      the returned model has a more standard transformer architecture.

  Returns:
    A mini GPT model instance.
  """
  if reservoir_params is None:
    reservoir_params = {}
  reservoir_params['weight'] = reservoir_weight

  inputs = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
  embedding_layer = PositionalEmbedding(maxlen, vocab_size, embed_dim)
  x = embedding_layer(inputs)
  for layer_i in range(num_transformer_layers):
    transformer_block = ReservoirTransformerBlockGPT(
        reservoir_params=reservoir_params,
        embed_dim=embed_dim,
        num_heads=num_heads,
        reservoir_base=reservoir_base,
        name=f'transformer_block_{layer_i}',
        build_reservoir=build_reservoir)
    x = transformer_block(x)
  outputs = tf.keras.layers.Dense(vocab_size)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile('adam', loss=loss_fn)
  return model
