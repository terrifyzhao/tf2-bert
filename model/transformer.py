import tensorflow as tf
from tensorflow.keras import Model
from model.attention import MultiHeadAttention
from model.embedding import InputEmbedding
from tensorflow.keras.layers import *
import numpy as np


class TransformerEncoder(Model):
    def __init__(self,
                 n_head,
                 hidden_size,
                 intermediate_size,
                 vocab_size,
                 max_position):
        super(TransformerEncoder, self).__init__()
        self.input_embedding = InputEmbedding(vocab_size, hidden_size, max_position)
        self.attention = MultiHeadAttention(n_head, hidden_size)
        self.layerNorm = LayerNormalization()
        self.fnn = FeedForward(hidden_size, intermediate_size)

    def call(self, inputs, training=None, mask=None):
        out = self.input_embedding(inputs)
        out = self.attention([out, out, out])
        out = out + inputs
        out = self.layerNorm(out)
        out = out + self.fnn(out)
        out = self.layerNorm(out)
        return out


class FeedForward(Model):
    def __init__(self,
                 hidden_size,
                 intermediate_size):
        super(FeedForward, self).__init__()
        self.d1 = Dense(intermediate_size, activation=gelu)
        self.d2 = Dense(hidden_size)

    def call(self, inputs, training=None, mask=None):
        out = self.d1(inputs)
        out = self.d2(out)
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


if __name__ == '__main__':
    model = TransformerEncoder(12, 768, 768, 411, 512)
