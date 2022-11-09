import collections
import string
from abc import abstractmethod, ABCMeta, ABC
from operator import itemgetter

import jieba

from enums.fileType import fileType
from enums.language import languageType
import numpy as np
import pandas as pd


# Vocab抽象类
class VocabRule(ABC):
    """
    path 数据文件路径
    language 语言类型
    seqNum 长度
    col 所需数据在文件中的位置
    wordNum 取频率最高的wordNum个词
    """

    def __init__(self, path, seqNum, language, col, wordNum, seq="\t"):
        self.seqNum = seqNum
        self.language = language
        self.wordNum = wordNum
        self.seq = seq
        self.dataInit = self.getData(path)
        self.dataIndex = self.dataSentence(self.dataInit)
        self.wordToIndex, self.IndexToWord = self.buildVocab(
            self.countWord(self.wordTag(self.data)), self.seqNum)

    # 读取数据
    def getData(self, path):
        type = path.split('.')[-1]
        data = None
        if type == fileType.txt:
            data = pd.read_csv(path, sep=self.seq, header=None)
            data = data.to_numpy()
        elif type == fileType.csv:
            data = pd.read_csv(path)
            data = data.to_numpy()
        return data

    # 单个语句分词
    def cutLine(self, sentence):
        senList = []
        if self.language == languageType.chinese:
            senList = jieba.lcut(sentence)
        elif self.language == languageType.english:
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
    @abstractmethod
    def wordTag(self, sentences):
        lineTotal = []
        wordTotal = []
        for sentence in sentences:
            lineTotal.append(self.cutLine(sentence))
        for line in lineTotal:
            wordTotal = wordTotal + line
        return wordTotal

    # 统计频率
    @abstractmethod
    def countWord(self, wordTotal):
        counter = collections.Counter()
        for word in wordTotal:
            counter[word] += 1
        sortedWordToCnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
        sortWords = [x[0] for x in sortedWordToCnt]
        return sortWords, sortedWordToCnt

    # 建立词汇表
    @abstractmethod
    def buildVocab(self, sortWords):
        vocab = sortWords[:self.wordNum]
        vocab = vocab + ['<unk>']
        wordToIndex = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
        indexToWord = {k: v for (k, v) in zip(range(len(vocab)), vocab)}
        return wordToIndex, indexToWord

    # 获取word对应Index
    @abstractmethod
    def getId(self, word):
        if word in self.wordToIndex:
            return [self.wordToIndex[word]]
        else:
            return [self.wordToIndex['<unk>']]

    # 单sentence转index
    @abstractmethod
    def indexSentence(self, sentence):
        pass

    @abstractmethod
    def pad(self, sentence, words=301):
        pass

    # data从文字转index格式
    @abstractmethod
    def dataSentence(self, data):
        pass
