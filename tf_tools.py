"""A toolbox of functions for tensorflow."""

import tensorflow as tf

from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables


def cosine_similarity(x, y, name=None):
    x = tf.math.l2_normalize(x)
    y = tf.math.l2_normalize(y)
    return math_ops.matmul(x, y, name=name)


def tf_matmul(x, y, name=None):
    """tf.matmul modified to accept standard inner product."""
    if x.ndim == y.ndim == 1:
        x_ = tf.reshape(x, shape=(1, -1))
        y_ = tf.reshape(y, shape=(-1, 1))
        return tf.matmul(x_, y_, name=name)
    else:
        return tf.matmul(x, y, name=name)


def compute_sequence_lengths(inputs, mask=None):
    """Calculates sequence lengths from mask and inputs."""
    if mask is None:
        input_shape = tf.slice(tf.shape(inputs), [0], [2])
        mask = tf.ones(input_shape)
    sequence_lengths = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-1)
    return sequence_lengths


def lp_normalize(x, p, axis=None, epsilon=1e-12, name=None, dim=None):
    r"""Normalizes a vector x with respect to its Lp norm:
        :math:`x_{norm} = \frac{x}{\max(\lVert x \rVert_p, \epsilon)}.`
    """
    square_sum = math_ops.reduce_sum(math_ops.pow(math_ops.abs(x), p), axis, keepdims=True)
    x_norm = math_ops.pow(math_ops.maximum(square_sum, epsilon), 1/p)
    x_norm_inv = math_ops.reciprocal_no_nan(x_norm)
    return math_ops.multiply(x, x_norm_inv, name=name)