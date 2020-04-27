import tensorflow as tf
from tensorflow.keras.layers import *
from bert_utils.utils import create_initializer


class InputEmbedding(Layer):
    def __init__(self,
                 config,
                 **kwargs):
        super(InputEmbedding, self).__init__(**kwargs)
        self.max_position = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.initializer = create_initializer(config.initializer_range)
        # 词向量
        self.token_embedding = Embedding(config.vocab_size,
                                         config.hidden_size,
                                         name='word_embeddings',
                                         embeddings_initializer=self.initializer)
        # 段落向量
        self.segment_embedding = Embedding(2,
                                           config.hidden_size,
                                           name='token_type_embeddings',
                                           embeddings_initializer=self.initializer)
        # 位置向量
        # self.position_embedding = Embedding(self.max_position,
        #                                     self.hidden_size,
        #                                     name='position_embeddings',
        #                                     embeddings_initializer=create_initializer(config.initializer_range))
        # drop_out & layer_normal
        self.drop_out = Dropout(config.hidden_dropout_prob)
        self.layer_normal = LayerNormalization(name="LayerNorm")

    def build(self, input_shape):
        super(InputEmbedding, self).build(input_shape)
        self.position_embedding = self.add_weight(name='position_embeddings/embeddings',
                                                  shape=(self.max_position,
                                                         self.hidden_size),
                                                  initializer=self.initializer)

    def call(self, inputs, training=None, mask=None):
        token, segment = inputs
        token_embedding = self.token_embedding(token)
        segment_embedding = self.segment_embedding(segment)

        # 位置编码
        import tensorflow.keras.backend as K
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[1], input_shape[2]
        # [512, 768]
        position_embedding = self.position_embedding[:seq_len]
        position_embedding = tf.expand_dims(position_embedding, 0)
        position_embedding = tf.tile(position_embedding, [batch_size, 1, 1])
        # print(K.shape(position_embedding))
        out = token_embedding + segment_embedding + position_embedding
        out = self.drop_out(self.layer_normal(out))
        return out
