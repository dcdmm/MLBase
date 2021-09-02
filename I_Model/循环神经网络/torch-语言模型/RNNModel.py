import torch.nn as nn
from abc import ABC


class RNNModel(nn.Module, ABC):
    """
    循环神经网络的每一个输出都包含前面所有的的信息,这种特性可以用来描述条件概率,同时可以使用
    循环神经网络的输出预测的单词来作为下一步的输入,这样就可以不断地预测新的单词,直到到达句子的末尾单词或达到预测的最大长度
    """

    def __init__(self, rnn_type, num_embeddings, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_size, num_layers, dropout=dropout)  # 选择循环神经网络种类
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]  # 选择激活函数
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.linear = nn.Linear(hidden_size, num_embeddings)
        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        """权重初始化"""
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, hidden):
        emb = self.drop(self.embed(text))  # emb.shape=[text.shape[0], text.shape[1], embedding_dim]
        # rnn_output.shape=[text.shape[0], BATCH_SIZE, hidden_size]
        # hidden.shape=[num_layers, BATCH_SIZE, hidden_size]
        rnn_output, hidden = self.rnn(emb, hx=hidden)  # 这里设置接受上次的记忆h/c

        output = self.drop(rnn_output)

        # decoded.shape=[text.shape[0]*BATCH_SIZE, num_embeddings]
        decoded = self.linear(output.view(output.size(0) * output.size(1), -1))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))  # 恢复形状(3 dims)
        return result, hidden
