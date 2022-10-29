import math
import time

import jieba
import torch
from torch import nn
import sys
import numpy as np

sys.path.append("..")
from preOperation.tokenWord import Vocab
import preOperation.wordSample as wordSample
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(torch.__version__, device)


class RnnModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, embedding_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.rnn = nn.RNN(input_size=embedding_size, hidden_size=num_hiddens)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=num_hiddens) # 已测试
        self.lin = nn.Linear(num_hiddens, output_size)
        self.state = None

    def forward(self, X, state):
        # print("xxxxxxxxxxxx")
        x = self.embedding(X.long())
        x, self.state = self.rnn(x, state)
        # x = d2l.to_onehot(X, vocab_size)
        # x, self.state = self.rnn(torch.stack(x), state)
        # print(x.shape,self.state.shape)
        # output = self.lin(x.view(-1, x.shape[-1]))
        output = self.lin(x)
        # print(output.shape)
        return output, self.state


# def predict_rnn_pytorch(model, sentence, wordToIndex, indexToWord, numPredict):
#     print(sentence)
#     senList = jieba.lcut(sentence)
#     predictList = vocab.getId(senList[0])
#     state = None
#     for t in range(numPredict + len(senList) - 1):
#         X = torch.tensor([predictList[-1]], device=device).view(1, 1)
#         if state is not None:
#             if isinstance(state, tuple):  # LSTM, state:(h, c)
#                 state = (state[0].to(device), state[1].to(device))
#             else:
#                 state = state.to(device)
#         (Y, state) = model(X, state)
#         # print(predictList)
#         if t < len(senList) - 1:
#             predictList.append(wordToIndex[senList[t + 1]])
#         else:
#             predictList.append(int(Y.argmax(dim=1).item()))
#     return ''.join([indexToWord[i] for i in predictList])


def predict_rnn_pytorch_1(prefix, num_chars, model, vocab_size, device, idx_to_char,
                        char_to_idx):
    state = None
    print(prefix[0])
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    print(output)
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def train_and_predict_rnn(model, lr, num_epochs, corpus_indices, batch_size, num_steps, device, pred_period,prefixes, pred_len):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum = 0.0
        n = 0
        start = time.time()
        data_iter = wordSample.data_iter_consecutive(corpus_indices=corpus_indices, batch_size=batch_size,
                                                     num_steps=num_steps, device=device)
        print(epoch)
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            # output --- num_steps, batch_size, vocab_size
            (output, state) = model(X, state)
            # print("000", output.shape)
            # output = torch.cat(output, dim=0)
            # contiguous()为这个.transpose新开内存存储
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # print("ll",output.shape,y.long().shape)
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item()*y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(l_sum/n)
        except OverflowError:
            preplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            print(prefixes)
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch_1(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))



if __name__ == '__main__':
    # vocab = Vocab('../tranDataset/cmn-eng/cmn.txt')
    # print(vocab.data[0][1])
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = wordSample.load_data()
    print("vo", vocab_size)
    model = RnnModel(vocab_size=vocab_size, num_hiddens=500, embedding_size=vocab_size, output_size=vocab_size)
    # print(predict_rnn_pytorch(model, vocab.data[0][1], vocab.wordToIndex, vocab.IndexToWord, 10))
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
    num_steps = 35
    batch_size = 2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict_rnn(model,lr,num_epochs,corpus_indices,batch_size,num_steps,device,pred_period,prefixes,pred_len)
