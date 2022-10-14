import torch.nn as nn
from DotProductAttention_model_torch import DotProductAttention
import torch


class MultiHeadAttention(nn.Module):
    """多头注意力(模仿pytorch)"""

    def __init__(self,
                 # 查询特征数目(E_q;E_q必须能整除num_heads)
                 query_size,
                 # 键特征数目(E_k)
                 key_size,
                 # 值特征数目(E_v)
                 value_size,
                 # 多头数
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert query_size % num_heads == 0, "query_size must be divisible by num_heads"
        # 可学习参数H^2 * 4
        self.W_q = nn.Linear(query_size, query_size, bias=bias)
        self.W_k = nn.Linear(key_size, query_size, bias=bias)
        self.W_v = nn.Linear(value_size, query_size, bias=bias)
        self.W_o = nn.Linear(query_size, query_size, bias=bias)
        self.attention = DotProductAttention(dropout)

    @staticmethod
    def transpose_qkv(X, num_heads):
        # 输入:X.shape=(N, L or S, E_q)
        # X.shape=(N, L or S, num_heads, E_q / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        # X.shape=(N, num_heads, L or S, E_q / num_heads)
        X = X.permute(0, 2, 1, 3)
        # 返回值.shape=(N * num_heads, L or S, E_q / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: 查询
        keys: 键
        values: 值
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions.
            a True value indicates that the corresponding position is not allowed to attend
            attn_mask.shape=(N * num_heads, L, S) or (L, S)
        """
        # queries.shape=(N, L, E_q);L is the target sequence length
        # self.W_q(queries).shape=(N, L, E_q)
        # queries.shape=(N * num_heads, L, E_q / num_heads)

        # keys.shape=(N, S, E_k);S is the source sequence length
        # self.W_k(queries).shape=(N, S, E_q)
        # keys.shape=(N * num_heads, S, E_q / num_heads)

        # values.shape=(N, S, E_v)
        # self.W_v(values).shape=(N, S, E_q)
        # values.shape=(N * num_heads, S, E_q / num_heads)
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        # attn_output.shape=(N * num_heads, L, E_q / num_heads)
        output, attn_output_weights = self.attention(queries, keys, values, attn_mask)
        # output.shape=(N, num_heads, L, E_q / num_heads)
        output = output.reshape(-1, self.num_heads, output.shape[1], output.shape[2])
        # output.shape=(N, L, num_heads, E_q / num_heads)
        output = output.permute(0, 2, 1, 3)
        # output.shape=(N, L, E_q)
        output_concat = output.reshape(output.shape[0], output.shape[1], -1)
        # attn_output.shape=(N, L, E_q)
        attn_output = self.W_o(output_concat)
        return attn_output, attn_output_weights


if __name__ == '__main__':
    query_size, key_size, value_size, num_heads = 100, 200, 200, 5
    multi_head_atten = MultiHeadAttention(query_size=query_size,
                                          key_size=key_size,
                                          value_size=value_size,
                                          num_heads=num_heads,
                                          dropout=0.1)
    multi_head_atten.eval()

    batch_size, num_queries, num_kvpairs = 2, 4, 6
    X = torch.randn((batch_size, num_queries, query_size))
    Y = torch.randn((batch_size, num_kvpairs, key_size))

    # attn_mask.shape=(4, 6)=(num_queries, num_kvpairs)
    attn_mask = torch.tensor([[False, False, False, False, False, True],
                              [False, False, False, True, True, True],
                              [False, False, True, True, True, True],
                              [False, False, False, False, True, True]])

    attn_output, attn_output_weights = multi_head_atten(X, Y, Y, attn_mask)

    print(attn_output.shape)
    # attn_output_weights.shape=(N * num_heads, L, S)
    print(attn_output_weights.shape)
