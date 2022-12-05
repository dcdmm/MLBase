import torch


class Model(torch.nn.Module):
    """下游训练任务模型"""

    def __init__(self, pretrained_model):
        super().__init__()
        self.fc = torch.nn.Linear(768, 11)  # 11分类任务
        self.pretrained = pretrained_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.pretrained(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = self.fc(out.pooler_output)
        out = out.softmax(dim=1)  # 模型预测值
        return out