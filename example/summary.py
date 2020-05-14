from bert_utils.pretrain_model import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import pandas as pd
from bert_utils.tokenization import Tokenizer
import tensorflow as tf
import numpy as np
from bert_utils.utils import pad_sequences
import json
import os
from tqdm import tqdm
import codecs

config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

min_count = 32
max_input_len = 256
max_output_len = 32
batch_size = 16
steps_per_epoch = 1000
epochs = 10000


def read_text():
    df = pd.read_csv('../data/train_tiny.csv')
    text = df['text'].values
    summary = df['summary'].values
    return text, summary


vocab_path = 'vocab.json'
if os.path.exists(vocab_path):
    chars = json.load(open(vocab_path, encoding='utf-8'))
else:
    chars = {}
    for a in tqdm(read_text(), desc=u'构建字表中'):
        for b in a:
            for w in b:  # 纯文本，不用分词
                chars[w] = chars.get(w, 0) + 1
    chars = [(i, j) for i, j in chars.items() if j >= min_count]
    chars = sorted(chars, key=lambda c: - c[1])
    chars = [c[0] for c in chars]
    json.dump(
        chars,
        open(vocab_path, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


def load_vocab(dict_path):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with codecs.open(dict_path, encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


_token_dict = load_vocab(dict_path)  # 读取词典
token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])

for c in chars:
    if c in _token_dict:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

model = load_model(checkpoint_path, dict_path, is_pool=False, seq2seq=True)
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


text, summary = read_text()


# def data_generator():
#     while True:
#         X, S = [], []
#         for a, b in zip(text, summary):
#             x, s = tokenizer.encode(a, b)
#             X.append(x)
#             S.append(s)
#             if len(X) == batch_size:
#                 X = padding(X)
#                 S = padding(S)
#                 yield [X, S], None
#                 X, S = [], []


class DataGenerator:
    def __init__(self, df_data, bs):
        self.df_data = df_data
        self.bs = bs
        self.steps = int((len(df_data) + bs - 1) / bs)

    def __len__(self):
        return len(self.df_data)

    def __iter__(self):
        while True:
            token_ids = []
            token_type_ids = []
            labels = []
            for data in self.df_data.values:
                text1 = data[0][0:20]
                text2 = data[1][0:50]
                token_id, token_type_id = tokenizer.encode(text1, text2)
                token_ids.append(token_id)
                token_type_ids.append(token_type_id)
                # labels.append()

                if len(token_ids) == self.bs or data is self.df_data.values[-1]:
                    token_ids = pad_sequences(token_ids)
                    token_type_ids = pad_sequences(token_type_ids)
                    yield [token_ids, token_type_ids], None
                    token_ids, token_type_ids, labels = [], [], []


model.summary()


class MyModel(Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, inputs, training=None, mask=None):
        # 交叉熵作为loss，并mask掉输入部分的预测
        out = self.model(inputs)
        y_in = inputs[0][:, 1:]  # 目标tokens
        y_mask = inputs[1][:, 1:]
        y_mask = tf.cast(y_mask, dtype=tf.float32)
        y = out[:, :-1]  # 预测tokens，预测与目标错开一位
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_in, y)
        cross_entropy = tf.reduce_sum(cross_entropy * y_mask) / tf.reduce_sum(y_mask)

        self.add_loss(cross_entropy)

        return y


model2 = MyModel(model)
model2.compile(optimizer=tf.keras.optimizers.Adam(1e-5))


def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(max_output_len):  # 强制要求输出不超过max_output_len字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model2.predict(
            [_target_ids, _segment_ids]
        )[:, -1, 3:]  # 直接忽略[PAD], [UNK], [CLS]
        _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
        for j, k in enumerate(_topk_arg):
            target_ids[j].append(_candidate_ids[k][-1])
            target_scores[j] = _candidate_scores[k]
        ends = [j for j, k in enumerate(target_ids) if k[-1] == 3]
        if len(ends) > 0:
            k = np.argmax([target_scores[j] for j in ends])
            return tokenizer.decode(target_ids[ends[k]])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def just_show():
    s1 = '人民检察院刑事诉讼涉案财物管理规定明确，不得查封、扣押、冻结与案件无关的财物，严禁在立案前查封、扣押、冻结财物，对查明确实与案件无关的，应当在三日内予以解除、退还。'
    s2 = '一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元'

    for s in [s1, s2]:
        print(u'生成标题:', gen_sent(s))
    print()


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()


from bert_utils.utils import Train

g = DataGenerator(pd.read_csv('../data/train_tiny.csv'), 8)
t = Train(5, is_binary=False)
t.train(model2, g)
