import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class InputEmbedding(Model):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 max_position):
        super(InputEmbedding, self).__init__()
        self.max_position = max_position
        self.hidden_size = hidden_size
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.segment_embedding = Embedding(2, hidden_size)
        initializer = tf.random_uniform_initializer()
        self.position_embedding = tf.Variable(initial_value=initializer([self.max_position, self.hidden_size]))

    def call(self, inputs, training=None, mask=None):
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        position_embedding = self.position_embedding[:seq_len]
        position_embedding = tf.expand_dims(position_embedding, 0)
        token_embedding = self.token_embedding(inputs)
        segment_embedding = self.segment_embedding(inputs)
        return token_embedding + segment_embedding + position_embedding


if __name__ == '__main__':
    em = InputEmbedding(1000, 256, 512)
    em(tf.random.uniform([32, 100]))
