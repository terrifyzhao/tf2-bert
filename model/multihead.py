import tensorflow as tf
from tensorflow.keras import Model
from model.attention import Attention
from tensorflow.keras.layers import *


class MultiHeadAttention(Model):
    def __init__(self,
                 n_head,
                 head_size,
                 dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.d_k = head_size // n_head
        self.dropout_rate = dropout_rate
        self.attention = Attention(self.d_k)
        self.q_matrix = Dense(head_size)
        self.k_matrix = Dense(head_size)
        self.v_matrix = Dense(head_size)
        self.head_matrix = Dense(head_size)

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


if __name__ == '__main__':
    model = MultiHeadAttention(12, 768)
    model(tf.random.uniform([3, 32, 15, 768]))
