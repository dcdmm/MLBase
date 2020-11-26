from abc import ABC
import torch
import torch.nn as nn


class SkipGram_loss(nn.Module, ABC):
    """自定义损失函数"""

    def __init__(self, reduction='none'):
        self.reduction = reduction
        super(SkipGram_loss, self).__init__()

    def forward(self, input_, target_, mask_=None):
        input_, target_, mask_ = input_.float(), target_.float(), mask_.float()
        res = nn.functional.binary_cross_entropy_with_logits(input_, target_, weight=mask_, reduction=self.reduction)
        res = res.mean(dim=1)  # 计算每个批次的损失
        res = res * mask_.shape[1] / mask_.to(torch.float32).sum(dim=1)  # 去除填充对res的影响
        return res


if __name__ == '__main__':
    loss = SkipGram_loss()
    pred = torch.tensor([[1.5, 0.3, -1, 2],
                         [1.1, -0.6, 2.2, 0.4]])
    label = torch.tensor([[1, 0, 0, 0],
                          [1, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1],
                         [1, 1, 1, 0]])  # 掩码变量

    print(loss(pred, label, mask))
