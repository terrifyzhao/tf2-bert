from bert_utils.pretrain_model import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import pandas as pd
from bert_utils.tokenization import Tokenizer
import tensorflow as tf
import numpy as np
from bert_utils.utils import pad_sequences

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

max_len = 30
batch_size = 8
EPOCHS = 5

tf.config.experimental_run_functions_eagerly(True)

train_df = pd.read_csv('../data/sentiment_data.csv')
train_df = train_df.sample(frac=1)

model = load_model(checkpoint_path, dict_path, is_pool=False)
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def data_generator(bs):
    while True:
        token_ids = []
        token_type_ids = []
        labels = []
        for data in train_df.values:
            text = data[1][0:max_len]
            token_id, token_type_id = tokenizer.encode(text)
            token_ids.append(token_id)
            token_type_ids.append(token_type_id)
            labels.append(data[0])

            if len(token_ids) == bs or data == train_df.values[-1]:
                token_ids = pad_sequences(token_ids)
                token_type_ids = pad_sequences(token_type_ids)
                yield [token_ids, token_type_ids], np.array(labels)
                token_ids, token_type_ids, labels = [], [], []


output = Dense(1, activation='sigmoid')(model.output[:, 0, :])
model = Model(inputs=model.input, outputs=output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=['accuracy']
)
x, y = data_generator(batch_size)
model.fit(x, y, batch_size=batch_size, epochs=EPOCHS)
