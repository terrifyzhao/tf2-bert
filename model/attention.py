import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import numpy as np


class Attention(Model):
    def __init__(self, d_k):
        super(Attention, self).__init__()
        self.d_k = d_k

    def call(self, inputs, training=None, mask=None):
        query, key, value = inputs
        if query.ndim == 4:
            query = tf.expand_dims(query, 2)
            key = tf.expand_dims(key, 2)
            value = tf.expand_dims(value, 2)
        out = tf.matmul(query, tf.transpose(key, [0, 1, 2, 4, 3])) / (self.d_k ** 0.5)
        out = tf.nn.softmax(out)
        out = tf.matmul(out, value)
        if out.ndim == 5:
            out = tf.squeeze(out, axis=2)

        return out


class MultiHeadAttention(Model):
    def __init__(self,
                 n_head,
                 hidden_size,
                 dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.head_size = hidden_size
        self.d_k = hidden_size // n_head
        self.dropout_rate = dropout_rate
        self.attention = Attention(self.d_k)
        self.q_matrix = Dense(hidden_size)
        self.k_matrix = Dense(hidden_size)
        self.v_matrix = Dense(hidden_size)
        self.head_matrix = Dense(hidden_size)

    def call(self, inputs, training=None, mask=None):
        query, key, value = inputs
        q = self.q_matrix(query)
        k = self.k_matrix(key)
        v = self.v_matrix(value)

        q = tf.reshape(q, [-1, self.n_head, tf.shape(q)[1], self.d_k])
        k = tf.reshape(k, [-1, self.n_head, tf.shape(k)[1], self.d_k])
        v = tf.reshape(v, [-1, self.n_head, tf.shape(v)[1], self.d_k])

        out = self.attention(inputs=[q, k, v])
        out = tf.reshape(out, [-1, tf.shape(out)[2], self.head_size])
        out = self.head_matrix(out)

        return out


class SelfAttention(Model):
    def __init__(self,
                 num_attention_heads,  # attention 头数
                 hidden_size,  # embedding维度
                 intermediate_size,  # FeedForward隐藏层维度
                 hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(num_attention_heads, hidden_size)
        self.drop_out = Dropout(hidden_dropout_prob)
        self.layerNorm1 = LayerNormalization()
        self.layerNorm2 = LayerNormalization()
        self.fnn = FeedForward(hidden_size, intermediate_size)

    def call(self, inputs, training=None, mask=None):
        attention_out = self.drop_out(self.attention([inputs, inputs, inputs]))
        out = inputs + attention_out
        out = self.layerNorm1(out)
        fnn_out = self.drop_out(self.fnn(out))
        out = out + fnn_out
        out = self.layerNorm2(out)
        return out


class FeedForward(Model):
    def __init__(self,
                 hidden_size,
                 intermediate_size):
        super(FeedForward, self).__init__()
        self.d1 = Dense(intermediate_size, activation=gelu)
        self.d2 = Dense(hidden_size)

    def call(self, inputs, training=None, mask=None):
        out = self.d1(inputs)
        out = self.d2(out)
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


if __name__ == '__main__':
    model = MultiHeadAttention(12, 768)
    model(tf.random.uniform([3, 32, 15, 768]))
