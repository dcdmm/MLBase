import torch
import torch.nn as nn


class RawGlobalPointer(nn.Module):
    """原理见:https://kexue.fm/archives/8373"""

    def __init__(self, hidden_size, ent_type_size, inner_dim, RoPE=True, tril_mask=True):
        super().__init__()
        self.hidden_size = hidden_size  # 即`预训练模型`.config.hidden_size
        self.ent_type_size = ent_type_size  # 实体或关系类型个数
        self.inner_dim = inner_dim
        self.linear = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE  # 是否添加旋转式位置编码RoPE
        self.tril_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        """旋转式位置编码RoPE"""
        # position_ids.shape=[seq_len, 1]
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # 绝对位置嵌入
        # indices.shape=[output_dim // 2]
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        # indices.shape=[output_dim // 2]
        indices = torch.pow(10000, -2 * indices / output_dim)
        # embeddings.shape=[seq_len, output_dim // 2]
        embeddings = position_ids * indices
        # embeddings.shape=[seq_len, output_dim // 2, 2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # embeddings.shape=[seq_len * batch_size, output_dim // 2, 2]
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        # embeddings.shape=[batch_size, seq_len, output_dim]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, context_outputs, attention_mask):
        self.device = attention_mask.device

        # last_hidden_state.shape=[batch_size, seq_len, hidden_size]
        last_hidden_state = context_outputs[0]
        batch_size, seq_len = last_hidden_state.size()[0], last_hidden_state.size()[1]

        # outputs.shape=[batch_size, seq_len, ent_type_size * inner_dim * 2]
        outputs = self.linear(last_hidden_state)
        # output[0].shape=[batch_size, seq_len, self.inner_dim * 2]
        # output[1].shape=[batch_size, seq_len, self.inner_dim * 2]
        # ......
        # output[-1].shape=[batch_size, seq_len, self.inner_dim * 2]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # 列表
        # outputs.shape=[batch_size, seq_len, ent_type_size, inner_dim * 2]
        outputs = torch.stack(outputs, dim=-2)
        # qw.shape=[batch_size, seq_len, ent_type_size, inner_dim]
        # kw.shape=[batch_size, seq_len, ent_type_size, inner_dim]
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb.shape=[batch_size, seq_len, inner_dim]
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos.shape=[batch_size, seq_len, 1, inner_dim]
            # sin_pos.shape=[batch_size, seq_len, 1, inner_dim]
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            # qw2.shape=[batch_size, seq_len, ent_type_size, inner_dim // 2, 2]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            # qw2.shape=[batch_size, seq_len, ent_type_size, inner_dim]
            qw2 = qw2.reshape(qw.shape)
            # qw.shape=[batch_size, seq_len, ent_type_size, inner_dim]
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits.shape=[batch_size, ent_type_size, seq_len, seq_len]
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # 计算内积

        # 排除padding
        # attention.shape=[batch_size, seq_len]
        # pad_mask.shape=[batch_size, ent_type_size, seq_len, seq_len]
        # 类似Multi-Head Attention的一个简化版,有多少种实体就对应多少个head(ent_type_size)
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12  # padding部分值为很小的负数

        # 排除下三角
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.


class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.encoder = encoder
        self.mention_detect = a
        self.s_o_head = b
        self.s_o_tail = c

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)  # 共享网络底层参数

        # 设计特定的标记策略
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs
