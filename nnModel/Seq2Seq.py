import torch
import math
from torch import nn
import collections


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocabSize, embeddingSize, numHiddens, numLayers, dropout=0):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingSize)
        self.rnn = nn.GRU(embeddingSize, numHiddens, numLayers, dropout=dropout)

    def forward(self, X):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocabSize, embeddingSize, numHiddens, numLayers, dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embeddingSize)
        self.rnn = nn.GRU(embeddingSize + numHiddens, numHiddens, numLayers, dropout)
        self.dense = nn.Linear(numHiddens, vocabSize)

    def initState(self, encOutputs):
        return encOutputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        print(X.shape)
        print(context.shape)
        XAndContext = torch.cat( (X, context) , 2)
        print(XAndContext.shape)
        output, state = self.rnn(XAndContext, state)
        output = self.dense(output).permute(1, 0, 2)
        return output,state


if __name__ == '__main__':
    net = Seq2SeqEncoder(vocabSize=10, embeddingSize=8, numHiddens=16, numLayers=2)
    net.eval()
    # batch_size num_steps
    X = torch.zeros((4, 7), dtype=torch.long)
    # num_steps,batch_size,num_hidden
    output, state = net(X)
    print(output.shape)
    print(state.shape)
    decoder = Seq2SeqDecoder(10,8,16,2)
    decoder.eval()
    state = decoder.initState(net(X))
    output,state = decoder(X,state)
    print(output.shape)
    print(state.shape)
    print(net)
