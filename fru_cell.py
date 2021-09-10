"""Implements Statistical Recurrent Unit (SRU) & Fourier Recurrent Unit (FRU).

These are two similar RNN that claim to be more succesful than LSTMs in
understanding long range dependencies. For the official repositories see
[3] and [4] for SRU and FRU respectively.

References:
    [1] Oliva, J. B., Póczos, B., & Schneider, J. (2017, July).  The statistical recurrent unit.
        In International Conference on Machine Learning (pp. 2671-2680). PMLR.
    [2] Zhang, J., Lin, Y., Song, Z., & Dhillon, I. (2018, July). Learning long term dependencies via fourier recurrent units.
        In International Conference on Machine Learning (pp. 5815-5823). PMLR.
    [3] https://github.com/mirandawork/sru.
    [4] https://github.com/limbo018/FRU.
"""

import math

import tensorflow as tf


def _linear(args, matrix, bias):
    """Helper function for an affine transformation w.r.t matrix."""
    if isinstance(args, (tuple, list)):
        return tf.matmul(tf.concat(args, axis=1), matrix) + bias
    else:
        return tf.matmul(args, matrix) + bias


class FRUCell(tf.keras.layers.Layer):
    """Fourier Recurrent Unit."""

    def __init__(self,
                 num_stats,
                 freqs,
                 recur_dims,
                 freqs_mask=1., # I don't understand the point of this.
                 linear_out=False,
                 include_input=False,
                 activation=tf.nn.gelu,
                 initializer = tf.initializers.glorot_uniform,
                 sequence_length = None,
                 **kwargs):
        self.num_stats = num_stats
        self.recur_dims = recur_dims
        self.freqs_array = freqs
        self.n_freqs = len(freqs)
        self.freqs_mask_array = [0.0 if w == 0 and len(freqs) > 1 else freqs_mask for w in freqs]
        self.linear_out = linear_out
        self.include_input = include_input
        self.activation = activation
        self.initializer = initializer
        self.sequence_length = sequence_length  # This is assigned by FRU.
        super(FRUCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return [tf.TensorShape([int(self.n_freqs * self.num_stats)]), tf.TensorShape([1])]

    @property
    def output_size(self):
        return int(self.n_freqs * self.num_stats)

    def build(self, input_shape):
        _state_size = self.state_size[0][0]
        if self.recur_dims > 0:
            self.recur_feats_matrix = self.add_weight(shape=(_state_size, self.recur_dims), initializer='uniform', name='recur_feats_matrix')
            self.recur_feats_bias = self.add_weight(shape=(self.recur_dims,), initializer=tf.keras.initializers.Constant(0), name='recur_feats_bias')

        rows = input_shape[-1]
        if self.recur_dims > 0:
            rows += self.recur_dims
        self.stats_matrix = self.add_weight(shape=(rows, self.num_stats), initializer='uniform', name='stats_matrix')
        self.stats_bias = self.add_weight(shape=(self.num_stats,), initializer=tf.keras.initializers.Constant(0), name='stats_bias')
        
        rows = _state_size
        if self.include_input > 0:
            rows += input_shape[-1]
        self.output_matrix = self.add_weight(shape=(rows, self.output_size), initializer='uniform', name='output_matrix')
        self.output_bias = self.add_weight(shape=(self.output_size,), initializer=tf.keras.initializers.Constant(0), name='output_bias')

        self.freqs = tf.reshape(tf.Variable(self.freqs_array, trainable=False, name="frequency"), shape=(1, -1, 1))
        self.phases = tf.Variable(tf.initializers.truncated_normal(stddev=0.1)(shape=(1, self.n_freqs, 1), dtype=tf.float32), trainable=True, name="phase")
        self.freqs_mask = tf.Variable(tf.reshape(self.freqs_mask_array, shape=(1, -1, 1)), trainable=False, name="frequency_mask")
        self.built = True

    def call(self, inputs, states, mask=None, constants=None):
        # cur_time_step signifies the element in the sequence for which
        # the weights are calculated. (i.e. the word being processed)
        # sequence_length is the length of each sequence. (i.e. #words)
        # Currently there is no good way to give sequence_length.
        states, c = states
        cur_time_step = c[0][0] + 1

        if self.recur_dims > 0:
            recur_output = self.activation(_linear(
                states, self.recur_feats_matrix, self.recur_feats_bias
            ))
            stats = self.activation(_linear(
                [inputs, recur_output], self.stats_matrix, self.stats_bias
            ))
        else:
            stats = self.activation(
                    tf.matmul(inputs, self.stats_matrix) + self.stats_bias)

        state_tensor = tf.reshape(states, shape=(-1, self.n_freqs, self.num_stats))
        stats_tensor = tf.reshape(stats, shape=(-1, 1, self.num_stats))

        # Calculation: mu_t = mask*mu_{t-1} + cos(2*pi*w*t/T + 2*pi*phase)*phi_t
        angle1 = (2. * math.pi / self.sequence_length) * cur_time_step * self.freqs
        angle2 = 2. * math.pi * self.phases
        first_term = self.freqs_mask * state_tensor
        second_term = (1 / self.sequence_length) * tf.cos(angle1 + angle2) * stats_tensor
        output_tensor = first_term + second_term
        out_state = tf.reshape(output_tensor, shape=(-1, self.state_size[0][0]), name='out_state')

        # Compute the output.
        if self.include_input:
            output_vars = [out_state, inputs]
        else:
            output_vars = out_state

        output = _linear(output_vars, self.output_matrix, self.output_bias)

        if not self.linear_out:
            output = self.activation(output, name='output_act')
        
        c += 1
        out_state = (out_state, c)
        return output, out_state


class SRUCell(tf.keras.layers.AbstractRNNCell):
    """Statistical Recurrent Unit Cell.

    Attributes:
        mavg_alphas: List of summary statistics tracked.
        num_stats: output size for each summary statistic. 
                The output size of the RNN is determined by len(mavg_alphas * num_stats).
        recur_dims: Number of previous states that are taken into account for each layer.
        activation: Activation function between the layers. Default is `tf.nn.gelu`
    """

    def __init__(self,
                 num_stats,
                 mavg_alphas,
                 recur_dims,
                 learn_alphas=False,
                 linear_out=False,
                 include_input=False,
                 activation=tf.nn.gelu,
                 **kwargs):
        super(SRUCell, self).__init__(**kwargs)
        self.num_stats = num_stats
        self.recur_dims = recur_dims
        if learn_alphas:
            logit_alphas = tf.Variable(initial_value=-tf.math.log(1.0 / mavg_alphas - 1), name='logit_alphas')
            self.mavg_alphas = tf.reshape(tf.nn.sigmoid(logit_alphas), shape=(1, -1, 1))
        else:
            self.mavg_alphas = tf.reshape(mavg_alphas, shape=(1, -1, 1))
        self.n_alphas = len(mavg_alphas)
        self.linear_out = linear_out
        self.activation = activation
        self.include_input = include_input

    @property
    def units(self):
        return self.state_size

    @property
    def output_size(self):
        return self.state_size

    @property
    def state_size(self):
        return int(self.n_alphas * self.num_stats)

    def build(self, input_shape):
        if self.recur_dims > 0:
            self.recur_feats_matrix = tf.Variable(tf.initializers.glorot_uniform()(shape=(self.state_size, self.recur_dims)), name='recur_feats_matrix')
            self.recur_feats_bias = tf.Variable(tf.zeros(shape=(self.recur_dims)), name='recur_feats_bias')

        rows = input_shape[-1]
        if self.recur_dims > 0:
            rows += self.recur_dims
        self.stats_matrix = tf.Variable(tf.initializers.glorot_uniform()(shape=(rows, self.num_stats)), name='stats_matrix')
        self.stats_bias = tf.Variable(tf.zeros(self.num_stats), name='stats_bias')
        
        rows = self.state_size
        if self.include_input > 0:
            rows += input_shape[-1]
        self.output_matrix = tf.Variable(tf.initializers.glorot_uniform()(shape=(rows, self.output_size)), name='output_matrix')
        self.output_bias = tf.Variable(tf.zeros(self.output_size), name='output_bias')
        self.built = True

    def call(self, inputs, states, training=None):
        if self.recur_dims > 0:
            recur_output = self.activation(_linear(
                states, self.recur_feats_matrix, self.recur_feats_bias
            ))
            stats = self.activation(_linear(
                [inputs, recur_output], self.stats_matrix, self.stats_bias
            ))
        else:
            stats = self.activation(
                    tf.matmul(inputs, self.stats_matrix) + self.stats_bias)
            
        state_tensor = tf.reshape(states, shape=(-1, self.n_alphas, self.num_stats))
        stats_tensor = tf.reshape(stats, shape=(-1, 1, self.num_stats))
        output_tensor = self.mavg_alphas * state_tensor + (1 - self.mavg_alphas) * stats_tensor
        out_state = tf.reshape(output_tensor, shape=(-1, self.state_size))

        # Compute the output.
        if self.include_input:
            output_vars = [out_state, inputs]
        else:
            output_vars = out_state
            
        output = _linear(output_vars, self.output_matrix, self.output_bias)

        if not self.linear_out:
            output = self.activation(output)
        return output, [out_state]

    def get_config(self):
        config = {'num_stats': self.num_stats,}
        base_config = super(SRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FRU(tf.keras.layers.RNN):
    """Fourier Recurrent Unit."""

    def __init__(self,
                num_stats,
                freqs,
                recur_dims,
                freqs_mask=1.,
                linear_out=False,
                include_input=False,
                activation=tf.nn.gelu,
                initializer = tf.initializers.glorot_uniform,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):
        if 'enable_caching_device' in kwargs: # Not supported
            cell_kwargs = {'enable_caching_device':
                    kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = FRUCell(
            num_stats,
            freqs,
            recur_dims,
            freqs_mask=1.,
            linear_out=False,
            include_input=False,
            activation=tf.nn.gelu,
            initializer = tf.initializers.glorot_uniform,
            **cell_kwargs)
        super(FRU, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
    #self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        self.cell.sequence_length = inputs.shape[1]  # Sequence length is sent to FRUCell.
        return super(FRU, self).call(
                inputs, mask=mask, training=training, initial_state=initial_state, constants=constants)

    @property
    def num_stats(self):
        return self.cell.num_stats

    @property
    def frequencies(self):
        return self.cell.freqs_array

    @property
    def recur_dims(self):
        return self.cell.recur_dims

    @property
    def linear_out(self):
        return self.cell.linear_out

    @property
    def include_input(self):
        return self.cell.include_input

    @property
    def sequence_length(self):
        return self.cell.sequence_length

    @property
    def activation(self):
        return self.cell.activation

    @property
    def initializer(self):
        return self.cell.initializer

    def get_config(self):
        config = {
            'num_stats': self.num_stats,
            'freqs': self.frequencies,
            'recur_dims': self.recur_dims,
            'linear_out': self.linear_out,
            'include_input': self.include_input,
            'sequence_length': self.sequence_length,
            'activation': self.activation,
            'initializer': self.initializer}
        base_config = super(FRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))