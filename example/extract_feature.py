from bert_utils.pretrain_model import build_bert
from bert_utils.tokenization import Tokenizer
import numpy as np

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 构建模型
max_seq_len = 10
batch_size = 10
model = build_bert(config_path, checkpoint_path, max_seq_len, batch_size)

# 编码测试
token_ids = []
segment_ids = []
for s in ['语言模型']:
    token_id, segment_id = tokenizer.encode(s)
    token_ids.append(token_id)
    segment_ids.append(segment_id)

out = model([np.array(token_ids), np.array(segment_ids)])

print(out)
