import math
import time

import torch
from torch import nn

from nnModel.RnnModel import RnnModel, train_and_predict_rnn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from preOperation.tokenWord import Vocab
import d2lzh_pytorch as d2l


class weiboData(Dataset):
    def __init__(self, path):
        self.vocab = Vocab(path)

    def __getitem__(self, item):
        # print(self.vocab.indexData[item], self.vocab.data[item, 1], self.vocab.data[item, 0])
        return self.vocab.indexData[item], self.vocab.data[item, 1], self.vocab.label[item]

    def __len__(self):
        return len(self.vocab.indexData)


def train_and_predict_rnn(model, lr, num_epochs, corpus_indices, batch_size, num_steps, device, pred_period, prefixes,
                          pred_len, data):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    ac = 0.0
    dui = 0
    cuo = 0
    for epoch in range(num_epochs):
        l_sum = 0.0
        n = 0
        start = time.time()
        print(epoch)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)  # 使用DataLoader加载数据
        for index, value in enumerate(dataloader):
            indexSentence, sentence, label = value
            # indexSentence = torch.tensor(indexSentence, dtype=torch.float32, device=device)
            # lebel = torch.tensor(lebel, dtype=torch.float32, device=device)
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            # output --- num_steps, batch_size, vocab_size
            (output, state) = model(indexSentence.to(device), state)
            # print("000", output.shape)
            # contiguous()为这个.transpose新开内存存储
            # y = torch.transpose(label, 0, 1).contiguous().view(-1)
            y = torch.tensor(numpy_to_oneHot(label.numpy(), 2), dtype=torch.float)
            # print("ll",output.shape,y.long().shape)
            y = y.squeeze(1)
            pre = output[:, -1, :]
            l = loss(pre.to(device), y.to(device))
            optimizer.zero_grad()
            l.backward()
            # d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            if index == 1:
                if (y[0][0]==1 and pre[0][0]>pre[0][1]) or (y[0][1]==1 and pre[0][1]>pre[0][0]):
                    dui =dui +1
                else:
                    cuo = cuo +1
                ac = 1.0*dui/(dui+cuo)
                print(y[0], pre[0], l, dui, cuo, ac)
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            preplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            print(prefixes)
            # for prefix in prefixes:
            #     print(' -', predict_rnn_pytorch_1(
            #         prefix, pred_len, model, vocab_size, device, idx_to_char,
            #         char_to_idx))


def numpy_to_oneHot(label, classNum):
    ans = []
    for batchI in range(label.shape[0]):
        batchIList = []
        list = label[batchI]
        for num in list:
            tmp = []
            for j in range(classNum):
                if j == int(num):
                    tmp.append(1)
                else:
                    tmp.append(0)
            batchIList.append(tmp)
        ans.append(batchIList)
    return ans


if __name__ == '__main__':
    wei = weiboData('../tranDataset/cls/weibo_senti_100k.csv')
    model = RnnModel(vocab_size=wei.vocab.chineseNum + 2, num_hiddens=100, embedding_size=100, output_size=2)
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
    num_steps = 35
    batch_size = 3
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    corpus_indices = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_and_predict_rnn(model, lr, num_epochs, corpus_indices, batch_size, num_steps, device, pred_period, prefixes,
                          pred_len, wei)

    print(wei.__len__())
    print(wei[100])
    print(wei.vocab.cutLine(wei[100][1]))
