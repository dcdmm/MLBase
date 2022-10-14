import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    """缩放点积注意力(论文`Attention ls All You Need`的注意力计算方式;模仿pytorch)"""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: 查询
        keys: 键
        values: 值
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions.
            a True value indicates that the corresponding position is not allowed to attend
            attn_mask.shape=(N, ?q, ?k) or (?q, ?k)
        """
        # queries.shape = (b, ?q, d)
        # keys.shape = (b, ?k, d)
        # attn_output_weights.shape = (b, ?q, d) x (b, d, ?k) = (b, ?q, ?k)
        d = queries.shape[-1]
        # 除以d的平方根
        # 原因:当维度很大时,点积结果会很大,会导致softmax的梯度很小(见softmax-Softmax.ipynb).为了减轻这个影响,对点积进行缩放
        attn_output_weights = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if attn_mask is not None:
            # 被遮蔽的元素使用⼀个非常大的负值替换,使其softmax输出为0
            attn_output_weights.masked_fill_(attn_mask, -1e8)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # values.shape=(b, ?k, ?v)
        # attn_output.shape=(b, ?q, ?k) x (b, ?k, ?v) = (b, ?q, ?v)
        attn_output = torch.bmm(self.dropout(attn_output_weights), values)
        return attn_output, attn_output_weights


if __name__ == '__main__':
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)

    attn_mask = torch.tensor([[False, False, True, True, True, True, True, True, True, True],
                              [False, False, False, False, False, False, True, True, True, True]])
    # attn_mask = torch.tensor([True] * 10).reshape(1, 10)

    # attn_mask.shape=(2, 1, 10)
    attn_mask = attn_mask.unsqueeze(1)

    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    attn_output, attn_output_weights = attention(queries, keys, values, attn_mask)

    print(attn_output)
    # attn_output.shape=(2, 1, 4)
    print(attn_output.shape)

    print(attn_output_weights)
    # attn_output_weights.shape=(2, 1, 10)
    print(attn_output_weights.shape)
