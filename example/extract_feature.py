from model.build_model import build_bert
from tokenization import Tokenizer
import numpy as np
import tensorflow as tf

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
# model = TransformerEncoder(config_path, checkpoint_path)  # 建立模型，加载权重

model = build_bert(config_path)
# model.get_layer(name='123').set_weights()

model.load_weights(checkpoint_path)

ckpt_reader = tf.train.load_checkpoint(checkpoint_path)

stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())
for s in stock_weights:
    a = model.get_layer('encoder/layer_0')
    print(123)

# 编码测试
token_ids, segment_ids = tokenizer.encode('语言模型', max_length=512)
out = model.predict([np.array([token_ids]), np.array([segment_ids])])
test_summary_writer = tf.summary.create_file_writer('logs/test')
print(model.summary())
print(out)
