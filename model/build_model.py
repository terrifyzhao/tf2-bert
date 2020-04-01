import json
from model.transformer import Bert
from model.config import BertConfig
import tensorflow as tf


def build_bert(config_path):
    # configs = {}

    # if config_path:
    #     configs.update(json.load(open(config_path)))
    configs = BertConfig()
    model = Bert(configs, name='bert')
    return model


def map_name(name):
    # 如果包含embeddings:0说明是嵌入层，需要把最后的embeddings去除。其它的把结尾的0去除即可
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
