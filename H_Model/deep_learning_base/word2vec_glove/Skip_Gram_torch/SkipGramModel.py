from abc import ABC
import torch.nn as nn
import torch


class SkipGramModel(nn.Module, ABC):
    def __init__(self, embed_size, embed_dimension):
        super(SkipGramModel, self).__init__()
        self.embed_size = embed_size
        self.embed_dimension = embed_dimension
        self.v_embeddings = nn.Embedding(embed_size, embed_dimension, sparse=True)  # 中心词词向量矩阵
        self.u_embeddings = nn.Embedding(embed_size, embed_dimension, sparse=True)  # 其他词词向量矩阵
        self._init_emb()

    def _init_emb(self):
        """初始化函数"""
        initrange = 0.5 / self.embed_dimension
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, center, contexts_and_negatives):
        embed_v = self.v_embeddings(torch.LongTensor(center))
        embed_u = self.u_embeddings(torch.LongTensor(contexts_and_negatives))
        pred = torch.bmm(embed_v, embed_u.permute(0, 2, 1))  # (背景词,噪音词,填充)与中心词向量的内积
        return pred
