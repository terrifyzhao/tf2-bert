from tensorflow.keras import Model
from model.multiheadattention import EncoderLayer
from model.embedding import InputEmbedding
from tensorflow.keras.layers import *


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
            out = attention(out)
        return out


class Bert(Model):
    def __init__(self,
                 config,
                 **kwargs):
        super(Bert, self).__init__(**kwargs)
        self.input_embedding = InputEmbedding(config,
                                              name='embeddings')

        self.encoder = TransformerEncoder(config,
                                          name='encoder')  # 初始化权重时的标准差

    def call(self, inputs, training=None, mask=None):
        out = self.input_embedding(inputs)
        out = self.encoder(out)
        return out


# class Bert(object):
#     def __init__(self,
#                  config):
#         super(Bert, self).__init__()
#         self.bert = BertEncoder(config, name='bert')
#
#     def predict(self, inputs):
#         out = self.bert(inputs)
#         return out
