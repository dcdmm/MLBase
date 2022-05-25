import torch
import torch.nn as nn
from abc import ABC


class TextRNN(nn.Module, ABC):
    """
    TextRNN模型的pytorch实现(具体任务对应修改)

    Parameters
    ---------
    vocab_size : int
        单词表的单词数目
    embedding_size : int
        输出词向量的维度大小
    hidden_size : int
        隐含变量的维度大小(权重矩阵W_{ih}、W_{hh}中h的大小)
    num_layers : int
        循环神经网络层数
    bidirectional : bool
        是否为设置为双向循环神经网络
    dropout_ratio : float
        元素归零的概率
    out_size : int
       输出特征的维度
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, bidirectional, dropout_ratio, out_size):
        super(TextRNN, self).__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=self.bidirectional,
                           dropout=dropout_ratio)
        if self.bidirectional:
            mul = 2
        else:
            mul = 1
        self.linear = nn.Linear(hidden_size * mul, out_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, text,
                text_lengths=None):  # 每个batch数据文本内容的真实长度
        # text.shape=[sent len, batch_size]

        # embedded.sahpe=[sen len, batch_size, embedding_size]
        embedded = self.dropout(self.embed(text))

        if text_lengths is not None:
            # ********************************************************************************************
            # 打包的变长序列
            pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,
                                                              enforce_sorted=False)  # pack sequence

            # hidden.shape=[num_layers * num directions, batch_size, hidden_size]
            # cell.shape=[num_layers * num directions, batch_size, hidden_size]
            pack_out, (hidden, cell) = self.rnn(pack_embedded)

            # Pads a packed batch of variable length sequences;pack_padded_sequence的反运算
            # output, output_lengths = nn.utils.rnn.pad_packed_sequence(pack_out)  # output.shape=[seq len, batch size, hidden_size * num directions]
            # ********************************************************************************************
        else:
            # ********************************************************************************************
            #  input=(sen len, batch_size, embedding_size)
            out_normal, (hidden, cell) = self.rnn(embedded)
            # ********************************************************************************************

        if self.bidirectional:  # 双向时
            # hidden = [batch_size, hid dim * num directions]
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 利用前向后后向的信息
        else:
            hidden = self.dropout(hidden[-1, :, :])
        result = self.linear(hidden)  # result.shape=[batch_size, out_size]
        return result
