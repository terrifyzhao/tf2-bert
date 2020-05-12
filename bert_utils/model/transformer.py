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


class NSP(Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def build(self, input_shape):
        self.output_weights = self.add_weight("output_weights",
                                              shape=[2, self.config.hidden_size],
                                              initializer=create_initializer(self.config.initializer_range))
        self.output_bias = self.add_weight('output_bias',
                                           shape=[2],
                                           initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        out = tf.matmul(inputs, self.output_weights, transpose_b=True)
        out = tf.nn.bias_add(out, self.output_bias)
        return out


class MLM(Layer):

    def __init__(self, config, embedding_weights, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding_weights = embedding_weights
        self.dense = Dense(config.hidden_size,
                           activation=config.hidden_act,
                           kernel_initializer=create_initializer(config.initializer_range))
        self.layer_norm = LayerNormalization()

    def build(self, input_shape):
        self.output_bias = self.add_weights("output_bias",
                                            shape=[self.config.vocab_size],
                                            initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        out = self.dense(inputs)
        out = self.layer_norm(out)
        out = tf.matmul(out, self.embedding_weights, transpose_b=True)
        out += self.output_bias
        return out


class Bert(Layer):
    def __init__(self,
                 config,
                 is_pool=False,
                 **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.dict_path = config.dict_path
        self.is_pool = is_pool
        self.input_embedding = InputEmbedding(config, name='embeddings')
        self.encoder = TransformerEncoder(config, name='encoder')
        self.pool = Pooler(config, name='pooler')
        self.nsp = NSP(config, name='cls/seq_relationship')

    def call(self, inputs, training=None, mask=None):
        if mask is None:
            mask = self._compute_mask(inputs)
        out = self.input_embedding(inputs)
        out = self.encoder(out, mask=mask)
        if self.is_pool:
            out = self.pool(out)
        return out

    def _compute_mask(self, inputs):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids = inputs
            token_type_ids = None

        return tf.cast(tf.not_equal(input_ids, 0), 'float32')
