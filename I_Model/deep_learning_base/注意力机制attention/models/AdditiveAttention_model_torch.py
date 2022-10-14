import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """加性注意力(模仿pytorch)"""

    def __init__(self,
                 # 键特征数目:key_size
                 key_size,
                 # 查询特征数据:query_size
                 query_size,
                 # 矩阵W_q、W_k,向量w_v输出特征维度
                 num_hiddens,
                 dropout):
        super(AdditiveAttention, self).__init__()

        # self.W_k.weight.shape = (num_hiddens, key_size)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # self.W_q.weight.shape = (num_hiddens, query_size)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # self.W_v.weight.shape = (1, num_hiddens)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
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
        # queries:(b, ?q, query_size) x (query_size, num_hiddens) = (b, ?q, num_hiddens)
        # keys:(b, ?k, key_size) x (key_size, num_hiddens) = (b, ?k, num_hiddens)
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries.unsqueeze(2).shape=(b, ?q, 1, num_hiddens)
        # keys.unsqueeze(1).shape=(b, 1, ?k, num_hiddens)
        # features.shape=(b, ?q, ?k, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # attn_output_weights:(b, ?q, ?k, num_hiddens) x (num_hiddens, 1) = (b, ?q, ?k, 1)
        # attn_output_weights.squeeze(-1).shape=(b, ?q, ?k)
        attn_output_weights = self.w_v(features).squeeze(-1)
        if attn_mask is not None:
            # 被遮蔽的元素使用⼀个非常大的负值替换,使其softmax输出为0
            attn_output_weights.masked_fill_(attn_mask, -1e8)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # values.shape=(b, ?k, ?v)
        # attn_output:(b, ?q, ?k) x (b, ?k, ?v) = (b, ?q, ?v)
        attn_output = torch.bmm(self.dropout(attn_output_weights), values)
        return attn_output, attn_output_weights


if __name__ == '__main__':
    # queries.shape(2, 1, 20)
    # keys.shape=(2, 10, 2)
    # values.shape=(2, 10, 4)
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)

    attn_mask = torch.tensor([[False, False, True, True, True, True, True, True, True, True],
                              [False, False, False, False, False, False, True, True, True, True]])
    # attn_mask = torch.tensor([True] * 10).reshape(1, 10)

    # attn_mask.shape=(2, 1, 10)
    attn_mask = attn_mask.unsqueeze(1)

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    attn_output, attn_output_weights = attention(queries, keys, values, attn_mask)

    print(attn_output)
    # attn_output.shape=(2, 1, 4)
    print(attn_output.shape)

    print(attn_output_weights)
    # attn_output_weights.shape=(2, 1, 10)
    print(attn_output_weights.shape)
