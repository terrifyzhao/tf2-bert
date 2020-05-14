import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootPath)

from bert_utils.pretrain_model import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import pandas as pd
from bert_utils.tokenization import Tokenizer
import numpy as np
from bert_utils.utils import pad_sequences, Train

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

train_df = pd.read_csv('../data/LCQMC/LCQMC_train.csv')
test_df = pd.read_csv('../data/LCQMC/LCQMC_dev.csv')
train_df = train_df.sample(frac=1)

max_len = 20
batch_size = 8
EPOCHS = 5

model = load_model(checkpoint_path, dict_path, is_pool=False)
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class DataGenerator:
    def __init__(self, df_data, bs):
        self.df_data = df_data
        self.bs = bs
        self.steps = int((len(df_data) + bs - 1) / bs)

    def __len__(self):
        return len(self.df_data)

    def __iter__(self):
        while True:
            token_ids = []
            token_type_ids = []
            labels = []
            for data in self.df_data.values:
                text1 = data[0][0:max_len]
                text2 = data[1][0:max_len]
                token_id, token_type_id = tokenizer.encode(text1, text2)
                token_ids.append(token_id)
                token_type_ids.append(token_type_id)
                labels.append([data[2]])

                if len(token_ids) == self.bs or data is self.df_data.values[-1]:
                    token_ids = pad_sequences(token_ids)
                    token_type_ids = pad_sequences(token_type_ids)
                    yield [token_ids, token_type_ids], np.array(labels)
                    token_ids, token_type_ids, labels = [], [], []


output = Dense(1, activation='sigmoid')(model.output[:, 0, :])
model = Model(inputs=model.input, outputs=output)

train = Train(EPOCHS, debug=True)
train_generator = DataGenerator(train_df, batch_size)
test_generator = DataGenerator(test_df, batch_size)
train.train(model, train_generator, test_generator, model_name='../output/text_matching_model')
