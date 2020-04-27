from bert_utils.model.attention import EncoderLayer
from bert_utils.model.embedding import InputEmbedding
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from bert_utils.utils import create_initializer


class TransformerEncoder(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.self_attention = []
        for i in range(self.num_hidden_layers):
            self.self_attention.append(EncoderLayer(config,
                                                    name=f'layer_{i}'))

    def call(self, inputs, training=None, mask=None):
        out = inputs
        for attention in self.self_attention:
            out = attention(out, mask=mask)
        return out


class Pooler(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(Pooler, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
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
        self.encoder = TransformerEncoder(config, name='encoder')  # 初始化权重时的标准差
        self.pool = Pooler(config, name='pooler')

    def call(self, inputs, training=None, mask=None):
        if mask is None:
            mask = self._compute_mask(inputs)
        out = self.input_embedding(inputs)
        out = self.encoder(out, mask=mask)
        if self.is_pool:
            out = self.pool(out)
        return out

    def _compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids = inputs
            token_type_ids = None

        return tf.cast(tf.not_equal(input_ids, 0), 'float32')
