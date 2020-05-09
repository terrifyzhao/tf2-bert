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

    def call(self, inputs, training=False, mask=None):
        out = self.attention([inputs, inputs, inputs], mask=mask)

        out = self.dense(out)
        out = self.dropout1(out, training=training)
        attention_out = self.attention_layer_norm(out + inputs, training=training)

        out = self.dense1(attention_out)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.out_layer_norm(out + attention_out, training=training)
        # print(tf.reduce_mean(out[:, 1], axis=-1))
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
        # out = mask_op(out, mask, mode='add')
        out = tf.nn.softmax(out, axis=-1)
        # out = self.drop_out(out)
        #  [batch_size, n_heads, seq_len, head_size]
        out = tf.matmul(out, value)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, tf.shape(out)[1], self.hidden_size])
        # return out, inputs[0]
        return out


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name,
                 num_heads=12,
                 size_per_head=64,
                 query_activation=None,
                 key_activation=None,
                 value_activation=None,
                 attention_dropout=0.1,
                 initializer_range=0.02, ):
        super(AttentionLayer, self).__init__(name=name)
        self.num_heads = num_heads
        self.size_per_head = size_per_head

        self.query_layer = tf.keras.layers.Dense(
            num_heads * size_per_head,
            activation=query_activation,
            kernel_initializer=create_initializer(initializer_range),
            name="query"
        )
        self.key_layer = tf.keras.layers.Dense(
            num_heads * size_per_head,
            activation=key_activation,
            kernel_initializer=create_initializer(initializer_range),
            name="key"
        )
        self.value_layer = tf.keras.layers.Dense(
            num_heads * size_per_head,
            activation=value_activation,
            kernel_initializer=create_initializer(initializer_range),
            name="value"
        )

        self.dropout_layer = tf.keras.layers.Dropout(attention_dropout)

        """
        B = batch size
        F = sequence length 'from_tensor'
        T = sequence length 'to_tensor'
        N = num_heads
        H = size_per_head
        """

    def transpose_for_scores(self, input_tensor, batch_size, num_heads, seq_len, size_per_head):
        output_shape = [batch_size, seq_len, num_heads, size_per_head]
        output_tensor = tf.reshape(input_tensor, output_shape)

        # [B, N, F, H]
        return tf.transpose(output_tensor, [0, 2, 1, 3])

    @staticmethod
    def create_attention_mask_from_input_mask(from_shape, to_mask):
        """
        Creates 3D attention.
        :param from_shape:  [B, F]
        :param to_mask:  [B, T]
        :return: [B, F, T]
        """
        # [B, T] -> [B, 1, T]
        mask = tf.cast(tf.expand_dims(to_mask, axis=1), dtype=tf.float32)
        # [B, F] -> [B, F, 1]
        ones = tf.expand_dims(tf.ones(shape=from_shape[:2], dtype=tf.float32), axis=-1)
        # [B, 1, T] * [B, F, 1] -> [B, F, T]
        mask = ones * mask
        return mask

    def call(self, inputs, mask=None, training=None):
        # [B, F, from_width]
        from_tensor = inputs
        # [B, T, to_width]
        to_tensor = inputs

        if mask is None:
            sh = get_shape_list(from_tensor)
            mask = tf.ones(sh[:2], dtype=tf.int32)

        # [B, F, T]
        attention_mask = AttentionLayer.create_attention_mask_from_input_mask(
            from_shape=tf.shape(from_tensor), to_mask=mask)

        # from_tensor.shape = [B, F, from_width]
        input_shape = tf.shape(from_tensor)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        to_seq_len = from_seq_len

        # [B, F, from_width] -> [B, F, N*H]
        query = self.query_layer(from_tensor)

        # [B, T, to_width] -> [B, T, N*H]
        key = self.key_layer(to_tensor)

        # [B, T, to_width] -> [B, T, N*H]
        value = self.value_layer(to_tensor)

        # [B, F, N*H] -> [B, N, F, H]
        query = self.transpose_for_scores(input_tensor=query,
                                          batch_size=batch_size,
                                          num_heads=self.num_heads,
                                          seq_len=from_seq_len,
                                          size_per_head=self.size_per_head)
        # [B, T, N*H] -> [B, N, T, H]
        key = self.transpose_for_scores(input_tensor=key,
                                        batch_size=batch_size,
                                        num_heads=self.num_heads,
                                        seq_len=to_seq_len,
                                        size_per_head=self.size_per_head)

        # [B, N, F, H] * [B, N, H, T] -> [B, N, F, T]
        attention_score = tf.matmul(query, key, transpose_b=True)  # tf.transpose(key, perm=[0, 1, 3, 2])
        attention_score = attention_score / tf.sqrt(float(self.size_per_head))

        if attention_mask is not None:
            # [B, F, T] -> [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=1)

            # attention_mask is 1 for position and 0 for mask, but we want 0 for mask and -10000 for mask.
            # {1: position, 0: mask} -> {0: position, -10000: mask}
            adder = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * -10000.0
            attention_score += adder

        # [B, N, F, T]
        attention_prob = tf.nn.softmax(attention_score)
        attention_prob = self.dropout_layer(attention_prob, training=training)

        # [B, T, N*H] -> [B, T, N, H]
        value = tf.reshape(value, [batch_size, to_seq_len, self.num_heads, self.size_per_head])

        # [B, T, N, H] -> [B, N, T, H]
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # [B, N, F, T] * [B, N, T, H] -> [B, N, F, H]
        context_layer = tf.matmul(attention_prob, value)

        # [B, N, F, H] -> [B, F, N, H]
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        output_shape = [batch_size, from_seq_len, self.num_heads * self.size_per_head]

        # [B, F, N, H] -> [B, F, N*H]
        context_layer = tf.reshape(context_layer, output_shape)
        return context_layer

    def compute_mask(self, inputs, mask=None):
        return mask  # [B, F]



def get_shape_list(tensor, expected_rank=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """

    # Cannot convert this function to autograph.
    if expected_rank is not None:
        assert_rank(tensor, expected_rank)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

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
