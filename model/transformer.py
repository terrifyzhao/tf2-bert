import tensorflow as tf
from tensorflow.keras import Model
from model.attention import SelfAttention
from model.embedding import InputEmbedding
from tensorflow.keras.layers import *
import numpy as np


class TransformerEncoder(Model):
    def __init__(self,
                 num_attention_heads,  # attention 头数
                 num_hidden_layers,  # tm层数
                 hidden_size,  # embedding维度
                 intermediate_size,  # FeedForward隐藏层维度
                 vocab_size,  # 词典大小
                 max_position_embeddings,  # 最大长度
                 hidden_dropout_prob,
                 initializer_range,  # 初始化权重时的标准差
                 **kwargs):
        super(TransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_embedding = InputEmbedding(vocab_size, hidden_size, max_position_embeddings, hidden_dropout_prob,initializer_range)
        self.self_attention = []
        for i in range(num_attention_heads):
            self.self_attention.append(SelfAttention(num_attention_heads,  # attention 头数
                                                     hidden_size,  # embedding维度
                                                     intermediate_size,  # FeedForward隐藏层维度
                                                     hidden_dropout_prob))

    def call(self, inputs, training=None, mask=None):
        out = self.input_embedding(inputs)
        for attention in self.self_attention:
            out = attention(out)
        return out


if __name__ == '__main__':
    model = TransformerEncoder(12, 768, 768, 411, 512)
