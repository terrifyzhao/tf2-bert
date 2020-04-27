from bert_utils.model.transformer import Bert
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from bert_utils.config import BertConfig



def load_model(checkpoint_path, dict_path, is_pool=False):
    p = PreTrainModel(checkpoint_path, dict_path, is_pool=is_pool)
    return p.model


class PreTrainModel(object):
    def __init__(self,
                 checkpoint_path,
                 dict_path,
                 is_pool=False):
        configs = BertConfig()
        self.bert = Bert(configs, is_pool=is_pool, name='bert')
        self.dict_path = dict_path

        l_input_ids = Input(shape=(None,), dtype='int32')
        l_token_type_ids = Input(shape=(None,), dtype='int32')

        output = self.bert([l_input_ids, l_token_type_ids])
        self.model = Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
        self.load_check_weights(self.bert, checkpoint_path)

    def _map_name(self, name):
        # 如果包含embeddings:0说明是嵌入层，需要把最后的embeddings去除。其它的把结尾的0去除即可
        if 'embeddings:0' in name:
            return name[:-(len('embeddings:0') + 1)]  # +1是因为有一个斜杠
        else:
            return name[:-2]

    def load_check_weights(self, bert, ckpt_path):
        ckpt_reader = tf.train.load_checkpoint(ckpt_path)

        loaded_weights = set()
        skip_count = 0
        weight_value_tuples = []
        skipped_weight_value_tuples = []

        bert_params = bert.weights
        param_values = tf.keras.backend.batch_get_value(bert.weights)
        for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
            stock_name = self._map_name(param.name)
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
