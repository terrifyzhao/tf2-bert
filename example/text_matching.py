from bert_utils.pretrain_model import PreTrainModel
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import pandas as pd
from tensorflow import keras
from bert_utils.tokenization import Tokenizer
from bert_utils.config import BertConfig
from bert_utils.model.transformer import Bert
import tensorflow as tf
import numpy as np

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

train_data = pd.read_csv('input/train.csv')
dev_data = pd.read_csv('input/dev.csv')


def map_name(name):
    # 如果包含embeddings:0说明是嵌入层，需要把最后的embeddings去除。其它的把结尾的0去除即可
    names = name.split('/')
    names[0] = names[0][:4]
    name = '/'.join(names)
    if 'embeddings:0' in name:
        return name[:-(len('embeddings:0') + 1)]  # +1是因为有一个斜杠
    else:
        return name[:-2]


def load_check_weights(bert, ckpt_path):
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    bert_params = bert.weights
    param_values = tf.keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_name(param.name)
        # print(stock_name)
        # print(param.name)
        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)

            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
    tf.keras.backend.batch_set_value(weight_value_tuples)


bert = PreTrainModel(checkpoint_path, dict_path, is_pool=True)
# inputs = bert.predict(first=train_data['sentence1'], second=train_data['sentence2'])

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
first = train_data['sentence1'].values
second = train_data['sentence2'].values
token_ids = []
segment_ids = []
for f, s in zip(first, second):
    token_id, segment_id = tokenizer.encode(first_text=f,
                                            second_text=s,
                                            first_length=10,
                                            second_length=10)
    token_ids.append(token_id)
    segment_ids.append(segment_id)

l_input_ids = Input(shape=(None,), dtype='int32')
l_token_type_ids = Input(shape=(None,), dtype='int32')
configs = BertConfig()
bert = Bert(configs, is_pool=True, name='bert')
output = bert([l_input_ids, l_token_type_ids])
output = Dense(100)(output)
output = Dense(1, activation='sigmoid')(output)
model = Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
load_check_weights(model, checkpoint_path)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=[np.array(token_ids), np.array(segment_ids)], y=train_data['label'].values, batch_size=2)
