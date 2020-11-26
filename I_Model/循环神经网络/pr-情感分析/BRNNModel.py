import torch
import torch.nn as nn
from abc import ABC


class BRNN(nn.Module, ABC):
    def __init__(self, vocal_size, embedding_size, hidden_size, num_layers, bidirectional, dropout, out_size):
        super(BRNN, self).__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocal_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=self.bidirectional,
                           dropout=dropout)
        if self.bidirectional:
            mul = 2
        else:
            mul = 1
        self.linear = nn.Linear(hidden_size * mul, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text.shape=[sent len, batch_size]

        # embedded.sahpe=[sen len, batch_size, embedding_size]
        embedded = self.dropout(self.embed(text))

        pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)  # pack sequence

        # hidden.shape=[num_layers * num directions, batch_size, hidden_size]
        # cell.shape=[num_layers * num directions, batch_size, hidden_size]
        pack_out, (hidden, cell) = self.rnn(pack_embedded)

        # output.shape=[seq len, batch size, hidden_size * num directions]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(pack_out)  # pad sequence;pad_packed_sequence使用的地方

        if self.bidirectional:  # 双向时
            # hidden = [batch_size, hid dim * num directions]
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 利用前向后后向的信息
        else:
            hidden = self.dropout(hidden[-1, :, :])
        result = self.linear(hidden)  # result.shape=[batch_size, out_size]
        return result
