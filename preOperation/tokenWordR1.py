import collections

import numpy as np
import pandas as pd
import string
import re
import jieba
from operator import itemgetter
import codecs
import sys

import torch

# engNum = 300
# chineseNum = 300
# sentenceNum = 119987


class Vocab():
    def __init__(self, path, model=0, engNum=300, chineseNum=300, sentenceNum=119987, dataCol=1, labelCol=0):
        self.sentenceNum = sentenceNum
        self.chineseNum = chineseNum
        self.engNum = engNum
        self.maxLen = 30
        self.data = self.getTxtData(path)
        self.model = model
        self.wordToIndex, self.IndexToWord = self.buildVocab(self.countWord(self.wordTag(self.data[:sentenceNum, dataCol])),
                                                             self.chineseNum)
        self.indexData = torch.LongTensor(np.array(self.dataSentence(self.data[:sentenceNum, dataCol])))
        # self.label = torch.from_numpy(np.array(self.data[:, labelCol]).astype(float)).unsqueeze(1)
        # self.label = torch.tensor(self.label, dtype=torch.long)

    # 获取txt数据
    def getTxtData(self, path):
        # data = pd.read_csv(path, sep="\t", header=None)
        data = pd.read_csv(path, sep="\t", header=None)
        data = data.to_numpy()
        data1 = data[:1000]
        data2 = data[72892:73892]
        data = np.concatenate((data1, data2), axis=0)
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
        if self.model == 1:
            wordNum = self.engNum
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
        return self.pad(tag)

    def pad(self, sentence, words=301):
        if len(sentence) < self.maxLen:
            for i in range(self.maxLen - len(sentence)):
                sentence.append(words)
        else:
            sentence = sentence[:self.maxLen]
        return sentence

    # data从文字转index格式
    def dataSentence(self, data):
        tag = []
        for sentence in data:
            tag.append(self.indexSentence(sentence))
        return tag

    # 单sentence转文字
    def sentenceWordToIndex(self,sentence):
        tag = []
        for sen in sentence:
            tag.append(self.IndexToWord[sen])
        return tag
if __name__ == '__main__':
    vocab = Vocab('../tranDataset/cmn-eng/cmn.txt')
    # vocab1 = Vocab('../tranDataset/cls/weibo_senti_100k.csv')
