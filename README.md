#  reservoir_nn
This package enables the use of reservoir computing architectures in Keras.

It enables the flexible creation of reservoir layers that can be used just like
any other type of Keras layer.

What principally distinguishes reservoir layers from other layer types is that
they have fixed weights.  In practice, effective reservoir layers are often
also very (>99%) sparse, although this is not a requirement.

N.B: Reservoir systems can overfit VERY fast, so training of models with these
layers may need to be adjusted in order to prevent overfitting.

## This package enables:
--creation of reservoir layers of a specified size and sparsity
and drawn from any arbitrary distribution.

--insertion of reservoir layers into conventional architectures
(e.g., Unets, LSTMs, etc).

--creation of "trainable" reservoirs that fix only the zero weights
of the layer, but not the non-zero weights.

--local learning (e.g., contrastive Hebbian learning) within the reservoir
layers.

--recurrence within reservoir layers.

--"next generation" reservoir computing wherein raw inputs are replaced with
concatenated sequences of non-linear transforms of the input. We refer to this
as "sideways time".

For more details on the package components and their usage, see the docstrings
for the individual functions.

## Examples

Here we build a simple input -> reservoir -> output model.

```
input_shape = (32,32)
reservoir_size = (1024, 1024)
reservoir_weight = np.random.uniform(low=0.0, high=1.0, size=reservoir_size)

model = reservoir_nn.models.segmentation_models.minimal_reservoir_model(
  input_shape = input_shape,
  reservoir_weight = reservoir_weight
  )

```

This will build a model that has a trainable input layer, a fixed reservoir
with random uniform weights of size (1024, 1024), and a trainable output layer.

The output of the model is the same size as the input, which is appropriate in,
for example, image segmentation problems.

Instead of using a random matrix, it is usually advantageous to use a **sparse**
random matrix.  This is done like so:

```
input_shape = (32,32)
reservoir_size = (1024, 1024)
reservoir_weight = np.random.uniform(low=0.0, high=1.0, size=reservoir_size)
sparse_weights = reservoir_nn.utils.weight_transforms.make_sparse(
  reservoir_weight,
  zero_weight_proportion = .95)

model = models.segmentation_models.minimal_reservoir_model(
  input_shape = input_shape,
  reservoir_weight = sparse_weights
  )
```

It can also be effective to fix only the **zero** valued weights of the sparse
reservoir, and allow the non-zeroed weights to train (i.e., fix the topology
of the reservoir but not the magnitude of the non-zero weights).  This can
be accomplished like so:

```
input_shape = (32,32)
reservoir_size = (1024, 1024)
reservoir_weight = np.random.uniform(low=0.0, high=1.0, size=reservoir_size)
sparse_weights = reservoir_nn.utils.weight_transforms.make_sparse(
  reservoir_weight,
  zero_weight_proportion = .95)

model = models.segmentation_models.minimal_reservoir_model(
  input_shape = input_shape,
  reservoir_weight = sparse_weights,
  trainable_reservoir = True
  )
```

There are a lot of options of other tweaks that can be made to even the base
reservoir model (e.g., recurrence within the reservoir, batchnorm, dropout, etc).  To see the full list of these, examine the docstring in
reservoir_nn.models.segmentation_models.minimal_reservior_model().

## Support
You can ask questions and join the development discussion:

[In the TensorFlow forum.] (https://discuss.tensorflow.org/)\
[On the Keras Google group.] (https://groups.google.com/g/keras-users)\
[On the Keras Slack channel.] (https://kerasteam.slack.com/)

Use [this link] (https://keras-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

## Opening an issue
You can also post bug reports and feature requests (only) in GitHub issues.

## Opening a PR
We welcome contributions! Before opening a PR, please read our contributor guide, and the API design guideline.
