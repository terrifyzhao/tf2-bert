import sys
import os

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootPath)

from bert_utils.pretrain_model import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import pandas as pd
from bert_utils.tokenization import Tokenizer
import tensorflow as tf
import numpy as np
from bert_utils.train import PiecewiseLinearLearningRate

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

train_data = pd.read_csv('../data/matching_data.csv')

model = load_model(checkpoint_path, dict_path, is_pool=True)
tokenizer = Tokenizer(dict_path, do_lower_case=True)

train_data = train_data.sample(frac=1)
first = train_data['sentence1'].values
second = train_data['sentence2'].values
label = train_data['label'].values


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
        for f, s, l in zip(first, second, label):
            f = f[0:30]
            s = s[0:30]
            token_id, segment_id = tokenizer.encode(first_text=f,
                                                    second_text=s)
            token_ids.append(token_id)
            segment_ids.append(segment_id)
            Y.append(l)
            i += 1
            if len(token_ids) == batch_size or i == len(train_data) - 1:
                token_ids = seq_padding(token_ids)
                segment_ids = seq_padding(segment_ids)
                yield [token_ids, segment_ids], np.array(Y)
                token_ids, segment_ids, Y = [], [], []


def data_generator2():
    token_ids = []
    segment_ids = []
    Y = []
    for f, s, l in zip(first[:3000], second[:3000], label[:3000]):
        f = f[0:30]
        s = s[0:30]
        token_id, segment_id = tokenizer.encode(first_text=f,
                                                second_text=s)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
        Y.append(l)
    token_ids = seq_padding(token_ids)
    segment_ids = seq_padding(segment_ids)
    return [token_ids, segment_ids], np.array(Y)


output = Dropout(rate=0.1)(model.output)
output = Dense(2, activation='softmax')(output)
model = Model(inputs=model.input, outputs=output)
print(model.summary())

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(1e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(tf.keras.optimizers.Adam(1e-4), {1000: 1e-4, 2000: 1e-5},
    #                                       name='adam'),
    metrics=['accuracy']
)
model.summary()

# model.fit_generator(data_generator(10), 200, epochs=10)
x, y = data_generator2()
model.fit(x, y, batch_size=16, epochs=10)
