from abc import ABC
import torch
import torch.nn as nn


class My_loss(nn.Module, ABC):
    def __init__(self, reduction='sum'):
        super(My_loss, self).__init__()
        self.reduction = reduction

    def forward(self, input_, target_):
        if self.reduction == 'sum':
            loss = torch.sum(torch.abs(input_ - target_))
            loss = loss.item()
        elif self.reduction == 'none':
            loss = torch.abs(input_ - target_)
        else:
            loss = None
        return loss  # ★★★★★反向传播要求返回值loss必须为标量


if __name__ == '__main__':
    x = torch.tensor([1, 2, 3, 4])
    y = torch.tensor([2, 3, 4, 5])

    loss_sum = My_loss()
    print(loss_sum(x, y))

    loss_none = My_loss(reduction='none')
    print(loss_none(x, y))
