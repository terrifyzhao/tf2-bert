import numpy as np
import pandas as pd
import tensorflow as tf

# vocab = pd.read_csv('data/vocab.txt', header=None)
# dictionary = dict(zip(vocab[0].values, vocab.index.values))
#
#
# # 把数据转换成index
# def seq2index(seq):
#     seg = tokenize(seq)
#     for i in range(0, len(seg)):
#         if seg[i] in dictionary.keys():
#             seg[i] = dictionary[seg[i]]
#         else:
#             seg[i] = dictionary['[UNK]']
#     return seg
#
#
# # 分词，添加额外的专有名词
# def tokenize(string):
#     new_words = ['李子期', '未来城', '华一']
#     for word in new_words:
#         jieba.add_word(word, 1000)
#
#     res = jieba.lcut(string, cut_all=False)
#     if '#' in res:
#         for i in range(len(res) - 1):
#             if res[i + 1] == '#':
#                 tmp_s = res[i] + res[i + 1] + res[i + 2]
#                 res[i] = tmp_s
#                 for j in range(i + 1, len(res) - 2):
#                     res[j] = res[j + 2]
#                 break
#         res = res[:-2]
#
#     # 把编号改成统一的标识符
#     for i in range(len(res)):
#         if len(res[i]) >= 3 and (res[i][0].isdigit() or res[i][0].encode('UTF-8').isalpha()):
#             res[i] = '[CODE]'
#     return res
#
#
# # 统一长度
# def padding_seq(X, max_len=15):
#     return np.array([
#         np.concatenate([x, [0] * (max_len - len(x))]) if len(x) < max_len else x[:max_len] for x in X
#     ])


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
