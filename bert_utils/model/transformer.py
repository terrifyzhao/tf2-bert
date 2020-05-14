from bert_utils.model.attention import MultiHeadAttention
from bert_utils.model.embedding import InputEmbedding
from tensorflow.keras.layers import *
import tensorflow as tf
from bert_utils.utils import create_initializer
import numpy as np


class EncoderLayer(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.dropout_rate = config.hidden_dropout_prob
        self.attention = MultiHeadAttention(config, name='attention/self')
        self.dense = Dense(config.hidden_size,
                           kernel_initializer=create_initializer(config.initializer_range),
                           name='attention/output/dense')
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = LayerNormalization(name='attention/output/LayerNorm')

        self.dense1 = Dense(config.intermediate_size, kernel_initializer=create_initializer(config.initializer_range),
                            name='intermediate/dense', activation=gelu)
        self.dense2 = Dense(config.hidden_size, kernel_initializer=create_initializer(config.initializer_range),
                            name='output/dense')

        self.out_layer_norm = LayerNormalization(name="output/LayerNorm")

    def call(self, inputs, training=False, mask=None):
        out = self.attention([inputs, inputs, inputs], mask=mask)

        out = self.dense(out)
        out = self.dropout(out, training=training)
        attention_out = self.attention_layer_norm(out + inputs, training=training)

        out = self.dense1(attention_out)
        out = self.dense2(out)
        out = self.dropout(out, training=training)
        out = self.out_layer_norm(out + attention_out, training=training)
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


class TransformerEncoder(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.encoders = []
        for i in range(self.num_hidden_layers):
            self.encoders.append(EncoderLayer(config,
                                              name=f'layer_{i}'))

    def call(self, inputs, training=None, mask=None):
        out = inputs

        for i, encoder in enumerate(self.encoders):
            out = encoder(out, mask=mask)

        return out


class Pooler(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(Pooler, self).__init__(**kwargs)
        self.dense = Dense(
            config.hidden_size,
            kernel_initializer=create_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, inputs, **kwargs):
        first_token_tensor = inputs[:, 0]
        out = self.dense(first_token_tensor)
        return out


class Bert(Layer):
    def __init__(self,
                 config,
                 is_pool=False,
                 seq2seq=False,
                 **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.dict_path = config.dict_path
        self.is_pool = is_pool
        self.seq2seq = seq2seq
        self.config = config
        self.input_embedding = InputEmbedding(config, name='embeddings')
        self.encoder = TransformerEncoder(config, name='encoder')
        self.pool = Pooler(config, name='pooler')

    def call(self, inputs, training=None, mask=None):
        if mask is None:
            mask = self._compute_mask(inputs, self.seq2seq)
        out = self.input_embedding(inputs)
        out = self.encoder(out, mask=mask)
        if self.is_pool:
            out = self.pool(out)
            # outs = [pool_out]
        # else:
        #     outs = [encoder_out]
        # if self.is_mlm:
        #     mlm_out = self.mlm([encoder_out, self.input_embedding.token_embedding.weights])
        #     outs.append(mlm_out)

        return out

    def _compute_mask(self, inputs, seq2seq):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids = inputs
            token_type_ids = None
        if seq2seq:
            seq_len = tf.shape(token_type_ids)[1]
            ones = tf.ones((1, self.config.num_attention_heads, seq_len, seq_len), dtype=tf.int32)
            # 下三角
            a_mask = tf.linalg.band_part(ones, -1, 0)
            # [bs,1,1,seq_len,]
            s_ex12 = tf.expand_dims(tf.expand_dims(token_type_ids, 1), 2)
            # [bs, 1,seq_len,1]
            s_ex13 = tf.expand_dims(tf.expand_dims(token_type_ids, 1), 3)
            a_mask = (1 - s_ex13) * (1 - s_ex12) + s_ex13 * a_mask
            a_mask = tf.reshape(a_mask, (-1, self.config.num_attention_heads, seq_len, seq_len))
            return a_mask
        else:
            return tf.cast(tf.not_equal(input_ids, 0), 'float32')
