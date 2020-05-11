import tensorflow as tf
from tensorflow.keras.layers import *
from bert_utils.utils import create_initializer


class InputEmbedding(Layer):
    def __init__(self,
                 config,
                 use_token_type=True,
                 use_position_embedding=True,
                 **kwargs):
        super(InputEmbedding, self).__init__(**kwargs)
        self.max_position = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.initializer = create_initializer(config.initializer_range)
        self.use_token_type = use_token_type
        self.use_position_embedding = use_position_embedding

        # 词向量
        self.token_embedding = Embedding(config.vocab_size,
                                         config.hidden_size,
                                         name='word_embeddings',
                                         mask_zero=True,
                                         embeddings_initializer=self.initializer)
        # 段落向量
        if use_token_type:
            self.token_type_embedding = Embedding(2,
                                                  config.hidden_size,
                                                  name='token_type_embeddings',
                                                  embeddings_initializer=self.initializer)

        # 位置编码
        if self.use_position_embedding:
            self.position_embedding = PositionEmbedding(self.max_position,
                                                        self.hidden_size,
                                                        self.initializer,
                                                        name='position_embeddings')

        # drop_out & layer_normal
        self.drop_out = Dropout(config.hidden_dropout_prob)
        self.layer_normal = LayerNormalization(name="LayerNorm")

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids = inputs
            token_type_ids = None

        out_embedding = self.token_embedding(input_ids)
        if token_type_ids is not None:
            out_embedding += self.token_type_embedding(token_type_ids)

        # 位置编码
        if self.position_embedding is not None:
            seq_len = tf.shape(input_ids)[1]
            pos_embedding = self.position_embedding(seq_len)
            pos_embedding = tf.expand_dims(pos_embedding, 0)
            out_embedding += pos_embedding

        out = self.drop_out(self.layer_normal(out_embedding))
        return out


class PositionEmbedding(Layer):

    def __init__(self,
                 max_position_embeddings,
                 hidden_size,
                 initializer,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.initializer = initializer

    def build(self, input_shape):
        self.position_embedding = self.add_weight(name='position_embeddings',
                                                  shape=(self.max_position_embeddings,
                                                         self.hidden_size),
                                                  initializer=self.initializer)
        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        seq_len = inputs

        assert_op = tf.debugging.assert_less_equal(seq_len, self.max_position_embeddings)

        with tf.control_dependencies([assert_op]):
            # 取seq_len长的张量
            full_position_embeddings = tf.slice(self.position_embedding,
                                                [0, 0],
                                                [seq_len, -1])
        output = full_position_embeddings
        return output
