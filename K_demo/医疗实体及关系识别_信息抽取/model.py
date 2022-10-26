import torch
import torch.nn as nn


class GlobalPointer(nn.Module):
    """原理见:https://kexue.fm/archives/8373"""

    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True, tril_mask=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size  # 实体类型个数
        self.inner_dim = inner_dim
        self.dense = nn.Linear(encoder.config.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE
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

    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids.shape=[batch_size, seq_len]
        batch_size = input_ids.size()[0]
        seq_len = input_ids.size()[1]
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state.shape=[batch_size, seq_len, hidden_size]
        last_hidden_state = context_outputs[0]

        # outputs.shape=[batch_size, seq_len, ent_type_size * inner_dim * 2]
        outputs = self.dense(last_hidden_state)
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
            # qw2.shape=[batch_size, seq_len, inner_dim // 2, 2]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            # qw2.shape=[batch_size, seq_len, inner_dim]
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
        # pad_mask.shape=[batch_size, seq_len, seq_len, seq_len]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12  # padding部分值为很小的负数

        # 排除下三角
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class CustomRelation(nn.Module):
    """自定义关系抽取模型(借鉴GlobalPointer,加入旋转式位置编码RoPE)"""

    def __init__(self, encoder, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.inner_dim = inner_dim
        self.linear = nn.Linear(encoder.config.hidden_size, self.inner_dim * 2)
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, position_ids, output_dim):
        """旋转式位置编码RoPE"""
        # position_ids.shape=[batch_size, seq_len]        
        # indices.shape=[output_dim // 2]
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        # indices.shape=[output_dim // 2]
        indices = torch.pow(10000, -2 * indices / output_dim)
        # embeddings.shape=[batch_size, seq_len, output_dim // 2]
        embeddings = position_ids[..., None] * indices[None, None, :]
        # embeddings.shape=[batch_size, seq_len, output_dim // 2, 2]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        # embeddings.shape=[batch_size, seq_len, output_dim]
        embeddings = torch.reshape(embeddings, (*position_ids.shape, -1))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, relations_idx, labels_mask=None):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids, position_ids=position_ids.to(self.device))
        # last_hidden_state.shape=[batch_size, seq_len, hidden_size]
        last_hidden_state = context_outputs.last_hidden_state

        # outputs.shape=[batch_size, seq_len, self.inner_dim * 2]
        outputs = self.linear(last_hidden_state)
        # qw.shape=kw.shape=[batch_size, seq_len, inner_dim]
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb.shape=[batch_size, seq_len, inner_dim]
            pos_emb = self.sinusoidal_position_embedding(position_ids, self.inner_dim)
            # cos_pos.shape=[batch_size, seq_len, inner_dim]
            # sin_pos.shape=[batch_size, seq_len, inner_dim]
            cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
            # qw2.shape=[batch_size, seq_len, inner_dim // 2, 2]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            # qw2.shape=[batch_size, seq_len, inner_dim]
            qw2 = qw2.reshape(qw.shape)
            # qw.shape=[batch_size, seq_len, inner_dim]
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # np.array(batch_idx).shape=[batch_size, relation_max_len]
        batch_idx = [[i] * relations_idx.shape[1] for i in range(relations_idx.shape[0])]
        # sub_start_idx.shape=[batch_size, relation_max_len]
        sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx = relations_idx[:, :, 0], \
                                                                 relations_idx[:, :, 1], \
                                                                 relations_idx[:, :, 2], \
                                                                 relations_idx[:, :, 3]

        # sub_start_qw.shape=[batch_size, relation_max_len, inner_dim]
        # 所有可能有关系的实体对象(sub)的类型组合在sentence_text中的token(如:'[修饰描述]')的位置
        sub_start_qw = qw[batch_idx, sub_start_idx, :]  # 网格索引
        # 所有可能有关系的实体对象(sub)的类型组合在sentence_text中的token(如:'[/修饰描述]')的位置,example:[[94, 95, 96, 97], [94, 95, 98, 99], xxxxxx]
        sub_end_qw = qw[batch_idx, sub_end_idx, :]
        # concat首尾特殊标记符对应的embedding向量
        # sub_qw.shape=[batch_size, relation_max_len, inner_dim * 2]
        sub_qw = torch.cat([sub_start_qw, sub_end_qw], dim=-1)

        # 所有可能有关系的实体对象(ojb)的类型组合在sentence_text中的token(如:'[否定描述]')位置
        obj_start_kw = kw[batch_idx, obj_start_idx, :]
        obj_end_kw = kw[batch_idx, obj_end_idx, :]
        # obj_kw.shape=[batch_size, relation_max_len, inner_dim * 2]
        obj_kw = torch.cat([obj_start_kw, obj_end_kw], dim=-1)

        # logits.shape=[batch_size, relation_max_len]
        logits = (sub_qw * obj_kw).sum(dim=-1)

        if labels_mask is not None:
            logits += (1.0 - labels_mask) * -1e12  # padding部分值为很小的负数

        return logits / self.inner_dim ** 0.5
