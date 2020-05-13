from bert_utils.model.transformer import Bert
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import Model
from bert_utils.utils import create_initializer


class NSP(Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def build(self, input_shape):
        self.output_weights = self.add_weight("output_weights",
                                              shape=[2, self.config.hidden_size],
                                              initializer=create_initializer(self.config.initializer_range))
        self.output_bias = self.add_weight('output_bias',
                                           shape=[2],
                                           initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        out = tf.matmul(inputs, self.output_weights, transpose_b=True)
        out = tf.nn.bias_add(out, self.output_bias)
        return out


class MLM(Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dense = Dense(config.hidden_size,
                           activation=gelu,
                           kernel_initializer=create_initializer(config.initializer_range),
                           name='transform')
        self.layer_norm = LayerNormalization(name='transform')

    def build(self, input_shape):
        self.output_bias = self.add_weight("output_bias",
                                           shape=[self.config.vocab_size],
                                           initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        inputs, embedding_weights = inputs
        out = self.dense(inputs)
        out = self.layer_norm(out)
        out = tf.matmul(out, embedding_weights, transpose_b=True)
        out += self.output_bias
        return out


class PreTrain(Model):

    def __init__(self,
                 config,
                 is_nsp=True,
                 is_mlm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.is_nsp = is_nsp
        self.is_mlm = is_mlm
        self.bert = Bert(config)
        self.nsp = NSP(config, name='cls/seq_relationship')
        self.mlm = MLM(config, name='cls/predictions')

    def call(self, inputs, **kwargs):
        out = self.bert(inputs)
        out_shape = tf.shape(out)
        outs = [out]
        if self.is_nsp:
            if len(out_shape) == 3:
                out = out[:, 0]
            outs.append(self.nsp(out))

        if self.is_mlm:
            mlm_out = self.mlm([out, self.bert.input_embedding.token_embedding.weights])
            outs.append(mlm_out)

        if len(outs) == 1:
            return outs[0]
        else:
            return outs[1:]
