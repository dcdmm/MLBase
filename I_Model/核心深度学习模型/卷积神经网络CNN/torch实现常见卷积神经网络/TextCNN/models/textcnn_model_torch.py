import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """
    TextCNN模型的pytorch实现(具体任务对应修改)

    Parameters
    ---------
    vocab_size : int
        单词表的单词数目
    embed_size : int
        输出词向量的维度大小
    kernel_sizes : tuple
        一般来说:不同大小卷积核的组合通常优于同样大小的卷积核
        不同卷积层卷积核的宽度;如kernel_sizes=(3, 4, 5)
    num_channels : tuple
        不同卷积层输出通道数;如num_channels=(100, 100, 100)
    dropput_ratio : float
        dropout层p值
    """

    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, dropout_ratio=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=None)
        # 预训练的词嵌入层
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.decoder = nn.Linear(sum(num_channels), 2)  # 二分类
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        # 通过nn.ModuleList()创建多个⼀维卷积层
        self.convs = nn.ModuleList()
        for out_channels, kernel_sizes in zip(num_channels, kernel_sizes):
            self.convs.append(
                # 两个嵌⼊的层连接,故in_channels=2 * embed_size
                nn.Conv1d(in_channels=2 * embed_size, out_channels=out_channels, kernel_size=kernel_sizes))

    def forward(self, inputs):
        # inputs.shape=(N, L);其实L表示序列长度
        # 沿着向量维度将两个嵌⼊层连接起来
        # embeddings.shape=(N, L, 2 * C);其中C表示输出词向量的维度大小
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据⼀维卷积层的输⼊格式,重新排列张量,以便通道作为第2维
        # embeddings.shape(N, 2 * C, L);
        embeddings = embeddings.permute(0, 2, 1)
        # conv(embeddings).shape=(N, out_channels, L_out);其中out_channelsh表示输出通道数,L_out表示每个输出通道的宽度
        # self.pool(conv(embeddings)).shape=(N, output_channels, 1)
        # torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1).shape=(N, output_channels)
        # encoding.shape=(N, output_channels1 + output_channels2 + output_channels3 + .......)
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1) for conv in self.convs],
                             dim=1)
        # outputs.shape=(N, 2)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


if __name__ == '__main__':
    model = TextCNN(100, 100, (3, 4, 5), (100, 100, 100))
    print(model)
