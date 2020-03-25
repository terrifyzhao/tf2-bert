import tensorflow as tf
from tensorflow.keras import Model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Attention(Model):
    def __init__(self, d_k):
        super(Attention, self).__init__()
        self.d_k = d_k

    def call(self, inputs, training=None, mask=None):
        if inputs.ndim == 4:
            inputs = tf.expand_dims(inputs, 2)
        query, key, value = inputs
        out = tf.matmul(query, tf.transpose(key, [0, 1, 3, 2])) / (self.d_k ** 0.5)
        out = tf.nn.softmax(out)
        out = tf.matmul(out, value)
        if out.ndim == 4:
            out = tf.squeeze(out, axis=1)

        return out


if __name__ == '__main__':
    model = Attention(64)
    model(tf.random.uniform([3, 8, 15, 768]))
