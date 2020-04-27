from bert_utils.pretrain_model import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import pandas as pd
from tensorflow import keras
from bert_utils.tokenization import Tokenizer
from bert_utils.config import BertConfig
from bert_utils.model.transformer import Bert
import tensorflow as tf
import numpy as np

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

train_data = pd.read_csv('../data/sentiment_data.csv')

model = load_model(checkpoint_path, dict_path, is_pool=False)
tokenizer = Tokenizer(dict_path, do_lower_case=True)

text = train_data['text'].values


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def data_generator(batch_size):
    i = 0
    while True:
        token_ids = []
        segment_ids = []
        Y = []
        for t in text:
            t = t[0:30]
            token_id, segment_id = tokenizer.encode(first_text=t)
            token_ids.append(token_id)
            segment_ids.append(segment_id)
            Y.append([train_data['label'].values[i]])
            i += 1
            if len(token_ids) == batch_size or i == len(train_data) - 1:
                token_ids = seq_padding(token_ids)
                segment_ids = seq_padding(segment_ids)
                yield [token_ids, segment_ids], np.array(Y)
                token_ids, segment_ids, Y = [], [], []


out = Lambda(lambda x: x[:, 0, :])(model.output)
output = Dense(1, activation='sigmoid')(out)
model = Model(inputs=model.input, outputs=output)

optimizer = tf.keras.optimizers.Adam(1e-5)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')


@tf.function
def train_cls_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()

    for x, y in data_generator(16):
        train_cls_step(x, y)

        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100))
