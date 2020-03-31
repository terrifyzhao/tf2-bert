import json
from model.transformer import Bert
from model.config import BertConfig


def build_bert(config_path):
    # configs = {}

    # if config_path:
    #     configs.update(json.load(open(config_path)))
    configs = BertConfig()
    model = Bert(configs, name='bert')
    return model


if __name__ == '__main__':
    config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
    m = build_bert(config_path)

    m.summary()
