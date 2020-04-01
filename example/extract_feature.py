from model.build_model import build_bert, load_check_weights
from tokenization import Tokenizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 构建模型
bert1 = build_bert(config_path)
max_seq_len = 10
tf.config.experimental_run_functions_eagerly(True)
l_input_ids = Input(shape=(max_seq_len,), batch_size=10, dtype='int32')
l_token_type_ids = Input(shape=(max_seq_len,), batch_size=10, dtype='int32')

output = bert1([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
model = Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

load_check_weights(model, checkpoint_path)
# 编码测试
token_ids, segment_ids = tokenizer.encode('语言模型', max_length=512)
out = model([np.array([token_ids]), np.array([segment_ids])])

print(out)
