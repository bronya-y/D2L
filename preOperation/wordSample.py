import collections
from operator import itemgetter

import torch
import random
import zipfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    with open('../tranDataset/zjl/jaychou_lyrics.txt', 'rt', encoding='utf-8') as f:
        corpus_chars = f.read()

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars.replace(' ', '')
    corpus_chars = corpus_chars[0:10000]
    counter = collections.Counter()
    for word in corpus_chars:
        counter[word] += 1
    sortedWordToCnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sortWords = [x[0] for x in sortedWordToCnt]
    # 建立字符串索引
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    # 建立字符索引
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)
    return corpus_indices, char_to_idx, idx_to_char, vocab_size
    # """加载周杰伦歌词数据集"""
    # with zipfile.ZipFile('../../data/jaychou_lyrics.txt.zip') as zin:
    #     with zin.open('jaychou_lyrics.txt') as f:
    #         corpus_chars = f.read().decode('utf-8')
    # corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    # corpus_chars = corpus_chars[0:10000]
    # idx_to_char = list(set(corpus_chars))
    # char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    # vocab_size = len(char_to_idx)
    # corpus_indices = [char_to_idx[char] for char in corpus_chars]
    # return corpus_indices, char_to_idx, idx_to_char, vocab_size


# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1，因为标签值是数据的最后一个的下一个，不减的话，如果本来正好能除尽label就会溢出
    # 长度/样本长度，能采出多少个样本
    num_examples = (len(corpus_indices) - 1) // num_steps
    # 每次拿batch_size个并行，能拿几次
    epoch_size = num_examples // batch_size
    # 打乱顺序
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    # print(corpus_indices)
    data_len = len(corpus_indices)
    # 分成batchSize份
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    # print("in", indices)
    # 每次取num_steps长度batch_len够几次的
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# my_seq = list(range(30))
# for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')
# for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')
