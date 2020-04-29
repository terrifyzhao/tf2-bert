from bert_utils.pretrain_model import load_model
from bert_utils.tokenization import Tokenizer
import numpy as np

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

model = load_model(checkpoint_path, dict_path, is_pool=False)

tokenizer = Tokenizer(dict_path, do_lower_case=True)

token_id, segment_id = tokenizer.encode('我老婆是喻言')

out = model([np.array([token_id]), np.array([segment_id])])
print(out)
