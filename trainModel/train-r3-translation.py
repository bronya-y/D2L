import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys

from nnModel.RnnModel import RnnModel
from preOperation.tokenWordR1 import Vocab


class CmnData(Dataset):
    def __init__(self, path, seqLen):
        self.maxSeqLen = seqLen
        self.dataEng = Vocab(path=path, model=1, dataCol=0)
        self.dataChi = Vocab(path=path, model=0, dataCol=1)

    def __getitem__(self, item):
        return self.dataEng.indexData[item], self.dataChi.indexData[item]

    def __len__(self):
        return len(self.dataChi.indexData)


def train_and_predict(epochs, model, dataset, batch_size, vocabSizeChi, vocabSizeEng, lr, device):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    state = None
    for epoch in range(epochs):
        for index, data in enumerate(dataloader):
            if state is not None:
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            engSen, chiSen = data
            (output, state) = model(engSen, state)
            chiOneHot = torch.tensor(numpy_to_oneHot(chiSen.numpy(), vocabSizeChi), dtype=torch.float)
            l = loss(output.to(device), chiOneHot.to(device))
            optimizer.zero_grad()
            l.backward()
            print(l)
            chiSenn = chiSen.detach().numpy()
            outputn = output.detach().numpy()
            print(dataset.dataChi.sentenceWordToIndex(chiSenn[0]))
            print(dataset.dataChi.sentenceWordToIndex(outputn[0]))
            # d2l.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
    return


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
    cmnData = CmnData('../tranDataset/cmn-eng/cmn.txt', 30)
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RnnModel(vocab_size=cmnData.dataEng.engNum + 2, num_hiddens=256, embedding_size=128, output_size= cmnData.dataChi.chineseNum)
    train_and_predict(epochs=50, model=model, batch_size=256, dataset=cmnData, lr = 0.03, vocabSizeChi=cmnData.dataChi.chineseNum, vocabSizeEng=cmnData.dataEng.engNum, device=device)
