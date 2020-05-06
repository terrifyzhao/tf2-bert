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

        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d_k = config.hidden_size // config.num_attention_heads
        self.dropout_rate = config.hidden_dropout_prob
        self.attention = MultiHeadAttention(config, self.d_k, name='attention/self')
        self.dense = Dense(config.hidden_size,
                           kernel_initializer=create_initializer(config.initializer_range),
                           name='attention/output/dense')
        self.dropout1 = Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = LayerNormalization(name='attention/output/LayerNorm')

        self.dense1 = Dense(config.intermediate_size, kernel_initializer=create_initializer(config.initializer_range),
                            name='intermediate/dense', activation=gelu)
        self.dense2 = Dense(config.hidden_size, kernel_initializer=create_initializer(config.initializer_range),
                            name='output/dense')
        self.dropout2 = Dropout(config.hidden_dropout_prob)
        self.out_layer_norm = LayerNormalization(name="output/LayerNorm")

    def call(self, inputs, training=None, mask=None):
        out = self.attention([inputs, inputs, inputs], mask=mask)

        out = self.dense(out)
        out = self.dropout1(out, training=training)
        attention_out = self.attention_layer_norm(tf.add(out, inputs))

        out = self.dense1(attention_out)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.out_layer_norm(tf.add(out, attention_out))

        return out


class MultiHeadAttention(Layer):
    def __init__(self, config, d_k, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_k = d_k
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.drop_out = Dropout(config.attention_probs_dropout_prob)
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

        # [batch_size, n_heads, seq_len, head_size]
        query = tf.reshape(query, [-1, self.num_attention_heads, tf.shape(query)[1], self.d_k])
        key = tf.reshape(key, [-1, self.num_attention_heads, tf.shape(key)[1], self.d_k])
        value = tf.reshape(value, [-1, self.num_attention_heads, tf.shape(value)[1], self.d_k])
        # [batch_size, n_heads, seq_len, seq_len]
        out = tf.matmul(query, tf.transpose(key, [0, 1, 3, 2])) / (self.d_k ** 0.5)
        out = mask_op(out, mask, mode='add')
        out = tf.nn.softmax(out, axis=-1)
        out = self.drop_out(out)
        #  [batch_size, n_heads, seq_len, head_size]
        out = tf.matmul(out, value)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, tf.shape(out)[1], self.hidden_size])
        # return out, inputs[0]
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
