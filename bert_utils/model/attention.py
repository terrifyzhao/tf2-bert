import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Layer
from bert_utils.utils import create_initializer


class MultiHeadAttention(Layer):
    def __init__(self, config, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert config.hidden_size % config.num_attention_heads == 0
        self.d_k = config.hidden_size // config.num_attention_heads
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

    def call(self, inputs, training=False, mask=None):
        query, key, value = inputs

        mask = tf.cast(tf.expand_dims(mask, axis=1), dtype=tf.float32)
        ones = tf.expand_dims(tf.ones(shape=tf.shape(query)[:2], dtype=tf.float32), axis=-1)
        attention_mask = ones * mask

        query = self.q_matrix(query)
        key = self.k_matrix(key)
        value = self.v_matrix(value)

        # [batch_size, seq_len, n_heads, head_size]
        query = tf.reshape(query, [-1, tf.shape(query)[1], self.num_attention_heads, self.d_k])
        key = tf.reshape(key, [-1, tf.shape(key)[1], self.num_attention_heads, self.d_k])
        value = tf.reshape(value, [-1, tf.shape(value)[1], self.num_attention_heads, self.d_k])

        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        # [batch_size, n_heads, seq_len, seq_len]
        out = tf.matmul(query, key, transpose_b=True) / (self.d_k ** 0.5)

        if attention_mask is not None:
            # [B, F, T] -> [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=1)

            # attention_mask is 1 for position and 0 for mask, but we want 0 for mask and -10000 for mask.
            # {1: position, 0: mask} -> {0: position, -10000: mask}
            adder = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * -10000.0
            out += adder

        out = tf.nn.softmax(out, axis=-1)
        out = self.drop_out(out, training=training)
        #  [batch_size, n_heads, seq_len, head_size]
        out = tf.matmul(out, value)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, tf.shape(out)[1], self.hidden_size])

        return out
