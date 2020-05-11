from bert_utils.pretrain_model import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import pandas as pd
from bert_utils.tokenization import Tokenizer
import tensorflow as tf
import numpy as np
from bert_utils.utils import pad_sequences, Train

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

train_df = pd.read_csv('../data/matching_train_data.csv')
test_df = pd.read_csv('../data/matching_test_data.csv')
train_df = train_df.sample(frac=1)[0:4000]

max_len = 30
batch_size = 32
EPOCHS = 5
steps = int((len(train_df) + batch_size - 1) / batch_size)

model = load_model(checkpoint_path, dict_path, is_pool=False)
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def data_generator(df, bs):
    while True:
        token_ids = []
        token_type_ids = []
        labels = []
        for data in df.values:
            text1 = data[0][0:max_len]
            text2 = data[1][0:max_len]
            token_id, token_type_id = tokenizer.encode(text1, text2)
            token_ids.append(token_id)
            token_type_ids.append(token_type_id)
            labels.append([data[2]])

            if len(token_ids) == bs or data is train_df.values[-1]:
                token_ids = pad_sequences(token_ids)
                token_type_ids = pad_sequences(token_type_ids)
                yield [token_ids, token_type_ids], np.array(labels)
                token_ids, token_type_ids, labels = [], [], []


output = Dense(1, activation='sigmoid')(model.output[:, 0, :])
model = Model(inputs=model.input, outputs=output)

train = Train(EPOCHS, steps, debug=True)
train.train(model, data_generator(train_df, batch_size), data_generator(test_df, batch_size))

# optimizer = tf.keras.optimizers.Adam(1e-5)
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
#
#
# @tf.function
# def train_cls_step(inputs, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs)
#
#         loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, predictions)
#
#     gradients = tape.gradient(loss, model.trainable_variables)
#
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
#
# for epoch in range(EPOCHS):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     template = 'Epoch {}, Step {}, Loss: {}, Accuracy: {}'
#     for x, y in data_generator(train_df, batch_size):
#
#         for step in range(steps):
#             train_cls_step(x, y)
#             print(template.format(epoch + 1,
#                                   step + 1,
#                                   train_loss.result(),
#                                   train_accuracy.result() * 100))
