from bert_utils.pretrain_model import PreTrainModel
import tensorflow as tf

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

model = PreTrainModel(checkpoint_path, dict_path)
out = model.predict(['极简预训练模型的使用', '奥术大师多'])
print(out)
