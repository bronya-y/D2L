import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size

        self.embedding = nn.Embedding(self._vocab_size, self._embedding_size)
        self.lstm = nn.LSTM(input_size=self._embedding_size,
                            hidden_size=self._hidden_size,
                            num_layers=1)

    def forward(self, sequence):
        # sequence shape: sequence_length, batch_size

        embedded = self.embedding(sequence)

        # embedded shape: sequence_length, batch_size, embedding_dim

        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

        self.embedding = nn.Embedding(self._vocab_size, self._embedding_size)
        self.lstm = nn.LSTM(input_size=self._embedding_size,
                            hidden_size=self._hidden_size,
                            num_layers=1)
        self.predictor = nn.Linear(self._hidden_size, self._vocab_size)

    def forward(self, input_sequence, hidden, cell):
        # input_token = [batch_size]

        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        input_sequence = input_sequence.unsqueeze(0)
        embedded = self.embedding(input_sequence)

        # embedded shape: 1 x batch_size x embedding_dim

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.predictor(output.squeeze(0))

        return prediction, hidden, cell
