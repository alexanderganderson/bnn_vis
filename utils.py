import numpy as np
from builtins import zip
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from graph_ops import DataGenerator


def batch_generator(data, batch_size):
    """
    data : array, shape (n_samples, ...)
        All of your data in a matrix.
    batch_size : int
        Batch size.

    Yields
    ------
    datum : shape (batch_size, ...)
        A batch of data.
    """
    n_samples = data.shape[0]

    num_batches = n_samples / batch_size

    for i in range(num_batches):
        yield data[i * batch_size: (i+1) * batch_size]


class ClassifierGenerator(DataGenerator):
    """
    Data generator for a classification problem.
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        assert inputs.shape[0] == outputs.shape[0]

    def get_generator(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data_size()
        inputs_gen = batch_generator(self.inputs, batch_size)
        outputs_gen = batch_generator(self.outputs, batch_size)
        return zip(inputs_gen, outputs_gen)

    def data_size(self):
        return self.inputs.shape[0]


def clip(x, m1=-1., m2=1.):
    return tf.maximum(m1, tf.minimum(m2, x))


def change_forward_backward(x, f, g, scope_name='binarize'):
    """Apply f on the forward pass to x, but use g on the backward pass."""
    # FIXME: Could be done more efficiently
    t = g(x)
    with tf.variable_scope(scope_name):
        y = t + tf.stop_gradient(f(x) - t)
    return y


def binarize(x, quantize):
    """
    Binarize function with smoothed backward pass.
    """
    if not quantize:
        return x
    return change_forward_backward(x, f=tf.sign, g=clip)


def ternarize(x, quantize, threshold):
    """
    Ternarize function with smoothed backward pass.
    """
    if not quantize:
        return x
    assert abs(threshold) < 1.

    def f(u):
        return tf.where(
            tf.greater(tf.abs(u), threshold),
            tf.ones_like(u),
            tf.zeros_like(u)) * tf.sign(u)
    return change_forward_backward(x, f=f, g=clip, scope_name='ternarize')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def flatten_conv(x):
    shape = x.get_shape().as_list()  # a list: [None, h, w, c]
    dim = np.prod(shape[1:])         # dim = prod([h, w, c])
    xp = tf.reshape(x, [-1, dim])    # -1 means "all"
    return xp, dim


def add_binary_layer(
    top, name, filter_shape, bin_act=True, bin_weights=True, max_pool=False,
    levels=2, threshold=0, training=False
):
    """
    Adds a binary layer to a neural network and keeps track of the relevant
        tensors.

    Parameters
    ----------
    top: Tensor, shape (batches, h, w, channels) or (batches, length)
        Input tensor to the layer
    name: str
        Name of desired layer
    filter_shape: tuple, length of at least 2
        Determines shape of filter, whether convolutional or fully connected layer is used
    bin_act: bool
        True = binarize output, False = leave output alone
    bin_weights: bool
        True = binarize weights, False = don't binarize weights
    max_pool: bool
        True = add 2x2 max pooling to convolutional layers
    levels: int
        Number of levels to quantize the weights and activations to.
    training : bool
        True if training

    Returns
    -------
    layer_out: Tensor, shape (batches, h, w, channels) or (batches, length)
                Output of layer
    tensor_dict : dictionary of relevant tensors
    """

    if levels == 2:
        quantization_func = binarize
    elif levels == 3:
        def quantization_func(*args):
            return ternarize(*args, threshold=threshold)
    else:
        raise ValueError('Invalid number of levels')

    with tf.variable_scope(name):
        W = tf.get_variable('W', shape=filter_shape)
        Wb = quantization_func(W, bin_weights)

        tensor_dict = {'wc': W, 'wb': Wb}

        if len(filter_shape) == 4:
            top = conv2d(top, Wb)
            if max_pool:
                top = max_pool_2x2(top)
        elif len(filter_shape) == 2:
            top = tf.matmul(top, Wb)
        else:
            raise ValueError('Invalid filter_shape')

        top = batch_norm(top, is_training=training)
        tensor_dict['pre_bin_act'] = top

        top = quantization_func(top, bin_act)
        tensor_dict['post_bin_act'] = top
    return top, tensor_dict
