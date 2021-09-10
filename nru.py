"""Implementation of the Non-Saturating Recurrent Unit(NRU).

Model proposed by the paper [1], the implementation in this code follows
the official github repository [2] which uses pytorch.

References:
    [1] Chandar, S., Sankar, C., Vorontsov, E., Kahou, S. E., & Bengio, Y. (2019, July).
    Towards non-saturating recurrent units for modelling long-term dependencies.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 3280-3287).
    [2] https://github.com/apsarath/NRU
"""

import math

import tensorflow as tf

from tf_tools import lp_normalize


class NRUCell(tf.keras.layers.Layer):
    """Non-Saturating Recurrent Unit Cell.

    Attributes:
        hidden_size: Size of the hidden layer.
        memory_size: Dimensions for hidden memory gates.
        k: Number of parameters for alpha and beta. k * memory_size must be a
            square.
        activation: Activation function one of 'tanh' or 'sigmoid'.
        use_relu: Decides whether to use ReLU as activation or not. It is
                recommended in Bengio et al to use ReLU to seperate the memory
                and forget gates.
        layer_norm: Boolean. Decides whether to do layer normalization while
                    calculatiing the hidden state.
    """

    def __init__(self,
                 hidden_size,
                 memory_size=64,
                 k=4,
                 activation="tanh",
                 use_relu=False,
                 layer_norm=False,
                 **kwargs):
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.k = k
        self._use_relu = use_relu
        self._layer_norm = layer_norm
        if activation == "tanh":
            self.activation = tf.nn.tanh
        elif activation == "sigmoid":
            self.activation = tf.nn.sigmoid
        super(NRUCell, self).__init__()
        if not math.sqrt(self.memory_size * self.k).is_integer():
            raise ValueError("memory_size * k must be a square.")

    @property
    def state_size(self):
        return [tf.TensorShape([self.memory_size]),
                tf.TensorShape([self.hidden_size])]

    @property
    def output_size(self):
        return self.memory_size + self.hidden_size

    def build(self, input_size):
        sqrt_memk = int(math.sqrt(self.memory_size * self.k))
        self.hm2v_alpha = tf.keras.layers.Dense(2 * sqrt_memk)
        self.hm2v_beta = tf.keras.layers.Dense(2 * sqrt_memk)
        self.hm2alpha = tf.keras.layers.Dense(self.k)
        self.hm2beta = tf.keras.layers.Dense(self.k)
        if self._layer_norm:
            # Not sure if this the same as torch's LayerNorm
            self._ln_h = tf.keras.layers.LayerNormalization()
        self.hmi2h = tf.keras.layers.Dense(self.hidden_size)

    def _opt_relu(self, x):
        return tf.nn.relu(x) if self._use_relu else x

    def _opt_layernorm(self, x):
        return self._ln_h(x) if self._layer_norm else x

    def call(self, inputs, states):
        memory, hidden_state = states
        concatenated_input = tf.concat([inputs, hidden_state, memory], axis=1)
        h = tf.nn.relu(self._opt_layernorm(self.hmi2h(concatenated_input)))

        # Flat memory equations
        alpha = self._opt_relu(self.hm2alpha(tf.concat([h, memory], axis=1)))
        beta = self._opt_relu(self.hm2beta(tf.concat([h, memory], axis=1)))

        u_alpha = self.hm2v_alpha(tf.concat([h, memory], axis=1))
        u_alpha = tf.split(u_alpha, 2, axis=1)
        v_alpha = tf.matmul(tf.expand_dims(u_alpha[0], axis=2),
                            tf.expand_dims(u_alpha[1], axis=1))
        v_alpha = tf.reshape(v_alpha, shape=(-1, self.k, self.memory_size))
        v_alpha = self._opt_relu(v_alpha)
        v_alpha = lp_normalize(v_alpha, p=5, axis=2, epsilon=1e-12)
        add_memory = tf.expand_dims(alpha, axis=2) * v_alpha

        u_beta = self.hm2v_beta(tf.concat([h, memory], axis=1))
        u_beta = tf.split(u_beta, 2, axis=1)
        v_beta = tf.matmul(tf.expand_dims(u_beta[0], axis=2),
                           tf.expand_dims(u_beta[1], axis=1))
        v_beta = tf.reshape(v_beta, shape=(-1, self.k, self.memory_size))
        v_beta = self._opt_relu(v_beta)
        v_beta = lp_normalize(v_beta, p=5, axis=2, epsilon=1e-12)
        forget_memory = tf.expand_dims(beta, axis=2) * v_beta

        memory = memory + tf.reduce_mean(add_memory - forget_memory, axis=1)
        hidden = (memory, h)
        output = tf.concat([memory, hidden_state], axis=-1)
        return output, hidden

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "memory_size": self.memory_size,
            "k": self.k,
            "activation": self.activation,
            "use_relu": self._use_relu,
            "layer_norm": self._layer_norm}
        base_config = super(NRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))