import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class InputEmbedding(Model):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position,
                 dropout_rate):
        super(InputEmbedding, self).__init__()
        self.max_position = max_position
        self.hidden_size = hidden_size
        # 词向量
        self.token_embedding = Embedding(vocab_size, hidden_size)
        # 段落向量
        self.segment_embedding = Embedding(2, hidden_size)
        # 位置向量
        # initializer = tf.random_uniform_initializer()
        # self.position_embedding = tf.Variable(initial_value=initializer([self.max_position, self.hidden_size]),
        #                                       name="position_embeddings",
        #                                       trainable=True)
        self.position_embedding = Embedding(self.max_position, self.hidden_size)
        # drop_out & layer_normal
        self.drop_out = Dropout(dropout_rate)
        self.layer_normal = LayerNormalization()

    def call(self, inputs, training=None, mask=None):
        token, segment = inputs
        token_embedding = self.token_embedding(token)
        segment_embedding = self.segment_embedding(segment)
        out = token_embedding + segment_embedding
        # 位置编码
        batch_size, seq_len = token_embedding.shape[0], token_embedding.shape[1]
        position_ids = tf.keras.backend.arange(seq_len)
        position_embedding = self.position_embedding(position_ids)
        # position_ids = tf.expand_dims(position_ids, 0).expand(input_shape)
        # position_embedding = self.position_embedding[:seq_len]
        position_embedding = tf.expand_dims(position_embedding, 0)
        position_embedding = tf.tile(position_embedding, [batch_size, 1, 1])
        # position_embedding = self.position_embedding(out)
        out = out + position_embedding
        out = self.layer_normal(self.drop_out(out))
        return out


if __name__ == '__main__':
    em = InputEmbedding(1000, 256, 512)
    em(tf.random.uniform([32, 100]))
