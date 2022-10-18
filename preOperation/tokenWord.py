import collections

import numpy as np
import pandas as pd
import string
import re
import jieba
from operator import itemgetter
import codecs
import sys

engNum = 300
chineseNum = 300


class Vocab():
    def __init__(self, path):
        self.data = self.getTxtData(path)
        self.model = 0
        self.wordToIndex, self.IndexToWord = self.buildVocab(self.countWord(self.wordTag(self.data[:, 0])),
                                                             chineseNum)
        self.indexData = self.dataSentence(self.data[:, 0], 1)

    # 获取txt数据
    def getTxtData(self, path):
        data = pd.read_csv(path, sep="\t", header=None)
        data = data.to_numpy()
        return data

    # 单个语句分词
    # 0代表中文，1代表英文
    def cutLine(self, sentence):
        if self.model == 0:
            senList = jieba.lcut(sentence)
        else:
            sentence = sentence.lower()  # 所有字母小写
            del_estr = string.punctuation + string.digits + string.whitespace  # 去除ASCII，标点符号，数字
            del_estr = del_estr.replace('-', '')  # 保留- 因为tag里有很多-有用
            replace = " " * len(del_estr)
            tran_tab = str.maketrans(del_estr, replace)
            sentence = sentence.translate(tran_tab)  # 完成上述去除标点符号的功能
            words = sentence.split(' ')  # 根据空格把句子分隔成单词
            while '' in words:
                words.remove('')  # 去除为空格的空值
            senList = words
        return senList

    # 获取词袋
    def wordTag(self, sentences):
        lineTotal = []
        wordTotal = []
        for sentence in sentences:
            lineTotal.append(self.cutLine(sentence))
        for line in lineTotal:
            wordTotal = wordTotal + line
        print(len(wordTotal))
        return wordTotal

    # 统计频率
    def countWord(self, wordTotal):
        counter = collections.Counter()
        for word in wordTotal:
            counter[word] += 1
        sortedWordToCnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
        sortWords = [x[0] for x in sortedWordToCnt]
        return sortWords, sortedWordToCnt

    # 获取词频前Num个
    # 建立词汇表
    def buildVocab(self, data, wordNum):
        sortWords, sortedWordToCnt = data
        vocab = sortWords[:wordNum]
        vocab = vocab + ['<unk>']
        wordToIndex = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
        indexToWord = {k: v for (k, v) in zip(range(len(vocab)), vocab)}
        return wordToIndex, indexToWord

    # 获取word对应Index
    def getId(self, word):
        # print(self.wordToIndex)
        if word in self.wordToIndex:
            return [self.wordToIndex[word]]
        else:
            return [self.wordToIndex['<unk>']]

    # 单sentence转index
    def indexSentence(self, sentence):
        tag = []
        for w in self.cutLine(sentence):
            tag = tag + self.getId(w)
        return tag

    # data从文字转index格式
    def dataSentence(self, data):
        tag = []
        for sentence in data:
            tag = tag + self.indexSentence(sentence)
        tagAns = np.array(tag)
        print(len(tagAns))
        return tagAns


if __name__ == '__main__':
    vocab = Vocab('../tranDataset/cmn-eng/cmn.txt')

