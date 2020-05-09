from bert_utils.model.attention import EncoderLayer, MultiHeadAttention,AttentionLayer
from bert_utils.model.embedding import InputEmbedding
from bert_utils.model.embeddings import BertEmbeddingsLayer
from tensorflow.keras.layers import *
import tensorflow as tf
from bert_utils.utils import create_initializer
import numpy as np


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

            # if i == 11:
            # print('-' * 100, i)
            # print(tf.reduce_mean(out[:, 0], axis=1))
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


class TransformerEncoderLayer(Layer):
    def __init__(self,
                 config,
                 hidden_size=768,
                 num_layers=12,
                 num_heads=12,
                 intermediate_size=3072,
                 intermediate_activation=gelu,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.2,
                 **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads))
        attention_head_size = int(hidden_size / num_heads)

        self.attention_heads = []
        self.attention_outputs = []
        self.attention_layer_norms = []
        self.intermediate_outputs = []
        self.layer_outputs = []
        self.output_layer_norms = []

        for layer_idx in range(num_layers):
            # attention_head = MultiHeadAttention(
            #     config, 64, name='layer_{}/attention/self'.format(layer_idx))
            attention_head = AttentionLayer(
                 name='layer_{}/attention/self'.format(layer_idx))
            self.attention_heads.append(attention_head)

            attention_output = tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name="layer_{}/attention/output/dense".format(layer_idx),
            )
            self.attention_outputs.append(attention_output)

            attention_layer_norm = tf.keras.layers.LayerNormalization(
                name="layer_{}/attention/output/LayerNorm".format(layer_idx))
            self.attention_layer_norms.append(attention_layer_norm)

            intermediate_output = tf.keras.layers.Dense(
                intermediate_size,
                activation=intermediate_activation,
                kernel_initializer=create_initializer(initializer_range),
                name="layer_{}/intermediate/dense".format(layer_idx)
            )
            self.intermediate_outputs.append(intermediate_output)

            layer_output = tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name="layer_{}/output/dense".format(layer_idx)
            )
            self.layer_outputs.append(layer_output)

            output_layer_norm = tf.keras.layers.LayerNormalization(
                name="layer_{}/output/LayerNorm".format(layer_idx)
            )
            self.output_layer_norms.append(output_layer_norm)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def call(self, inputs, mask=None, out_layer_idxs=None, training=False):
        # input_shape = self.get_shape_list(inputs, expected_rank=3)
        # batch_size, seq_length, input_width = input_shape[0], input_shape[1], input_shape[2]
        #
        # if input_width != self.hidden_size:
        #     raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
        #                      (input_width, self.hidden_size))

        prev_output = inputs
        all_layer_outputs = []
        for layer_idx in range(self.num_layers):
            layer_input = prev_output

            attention_heads = []
            attention_head = self.attention_heads[layer_idx](layer_input,
                                                             mask=mask,
                                                             training=training)
            attention_heads.append(attention_head)

            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                attention_output = tf.concat(attention_heads, axis=-1)

            attention_output = self.attention_outputs[layer_idx](attention_output)
            attention_output = self.dropout(attention_output, training=training)
            attention_output = self.attention_layer_norms[layer_idx](attention_output + layer_input)

            intermediate_output = self.intermediate_outputs[layer_idx](attention_output)

            layer_output = self.layer_outputs[layer_idx](intermediate_output)
            layer_output = self.dropout(layer_output, training=training)
            layer_output = self.output_layer_norms[layer_idx](layer_output + attention_output)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if out_layer_idxs is None:
            final_output = all_layer_outputs[-1]

        else:
            final_output = []
            for idx in out_layer_idxs:
                final_output.append(all_layer_outputs[idx])
            final_output = tuple(final_output)

        return final_output


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
                 **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.dict_path = config.dict_path
        self.is_pool = is_pool
        self.input_embedding = InputEmbedding(config, name='embeddings')
        # self.encoder = TransformerEncoder(config, name='encoder')
        self.encoder = TransformerEncoderLayer(config, name='encoder')
        self.pool = Pooler(config, name='pooler')

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
