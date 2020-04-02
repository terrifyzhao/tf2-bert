from bert_utils.pretrain_model import PreTrainModel

config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

max_seq_len = 10
batch_size = 10
model = PreTrainModel(max_seq_len, batch_size, checkpoint_path, dict_path)
out = model.predict(['极简预训练模型的使用'])
print(out)
