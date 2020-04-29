import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Layer
import numpy as np
from bert_utils.utils import create_initializer, mask_op


class EncoderLayer(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_attention = SelfAttention(config,
                                            name='attention')
        self.intermediate = Intermediate(config.intermediate_size,
                                         config.initializer_range,
                                         name='intermediate')
        self.outputs = Output(config.hidden_size,
                              config.initializer_range,
                              config.hidden_dropout_prob,
                              name='output')

    def call(self, inputs, training=None, mask=None):
        attention_out = self.self_attention([inputs, inputs, inputs], mask=mask)
        out = self.intermediate(attention_out)
        out = self.outputs([attention_out, out])
        return out


class SelfAttention(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d_k = config.hidden_size // config.num_attention_heads
        self.dropout_rate = config.hidden_dropout_prob
        self.attention = MultiHeadAttention(config, self.d_k, name='self')
        self.attention_out = AttentionOutput(config, name='output')

    def call(self, inputs, training=None, mask=None):
        out, attention_input = self.attention(inputs, mask=mask)

        out = self.attention_out([out, attention_input])
        return out


class MultiHeadAttention(Layer):
    def __init__(self, config, d_k, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_k = d_k
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.q_matrix = Dense(self.hidden_size,
                              kernel_initializer=create_initializer(config.initializer_range),
                              name='query')
        self.k_matrix = Dense(self.hidden_size,
                              kernel_initializer=create_initializer(config.initializer_range),
                              name='key')
        self.v_matrix = Dense(self.hidden_size,
                              kernel_initializer=create_initializer(config.initializer_range),
                              name='value')

    def call(self, inputs, training=None, mask=None):
        query, key, value = inputs
        query = self.q_matrix(query)
        key = self.k_matrix(key)
        value = self.v_matrix(value)

        query = tf.reshape(query, [-1, self.num_attention_heads, tf.shape(query)[1], self.d_k])
        key = tf.reshape(key, [-1, self.num_attention_heads, tf.shape(key)[1], self.d_k])
        value = tf.reshape(value, [-1, self.num_attention_heads, tf.shape(value)[1], self.d_k])

        out = tf.matmul(query, tf.transpose(key, [0, 1, 3, 2])) / (self.d_k ** 0.5)

        out = mask_op(out, mask, mode='add')
        out = tf.nn.softmax(out, axis=-1)

        out = tf.matmul(out, value)
        out = tf.transpose(out, [0, 1, 3, 2])
        out = tf.reshape(out, [-1, tf.shape(out)[3], self.hidden_size])
        out = mask_op(out, mask, mode='mul')
        return out, inputs[0]


class AttentionOutput(Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = Dense(config.hidden_size,
                           kernel_initializer=create_initializer(config.initializer_range),
                           name='dense')
        self.LayerNorm = LayerNormalization(name='LayerNorm')
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        attention_outputs, attention_inputs = inputs
        out = self.dense(attention_outputs)
        out = self.dropout(out)
        out = self.LayerNorm(out + attention_inputs)
        return out


class Intermediate(Layer):
    def __init__(self,
                 intermediate_size,
                 initializer_range,
                 **kwargs):
        super(Intermediate, self).__init__(**kwargs)
        self.dense = Dense(intermediate_size, kernel_initializer=create_initializer(initializer_range), name='dense')
        self.act = Activation(gelu)

    def call(self, inputs, training=None, mask=None):
        out = self.dense(inputs)
        out = self.act(out)
        return out


class Output(Layer):
    def __init__(self,
                 hidden_size,
                 initializer_range,
                 hidden_dropout_prob,
                 **kwargs):
        super(Output, self).__init__(**kwargs)
        self.dense = Dense(hidden_size, kernel_initializer=create_initializer(initializer_range), name='dense')
        # self.act = Activation(gelu)
        self.drop_out = Dropout(hidden_dropout_prob)
        self.layerNorm = LayerNormalization(name="LayerNorm")

    def call(self, inputs, training=None, mask=None):
        attention_out, out = inputs
        out = self.dense(out)
        out = self.drop_out(out)
        out = self.layerNorm(attention_out + out)
        return out


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

