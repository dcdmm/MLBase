import torch.nn as nn
import torch


class FocalLoss(nn.Module):
    """多分类focal loss函数的实现"""

    def __init__(self, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # 可调节因子gamma
        self.weight = weight  # 类别权重
        self.reduction = reduction  # 默认reduction='mean',参考nn.BCELos

    def forward(self,
                input,  # 参考nn.CrossEntropyLoss forward函数input
                target):  # 参考nn.CrossEntropyLoss forward函数target
        cross_ent_layer = nn.CrossEntropyLoss(reduction=self.reduction, weight=self.weight)
        cross_ent_loss = cross_ent_layer(input, target)
        pt = torch.exp(-cross_ent_loss)
        # pt越大(分类的难易程度越高),权重(1 - pt) ** self.gamma越小,对总损失F_loss越小
        # pt越小(分类的难易程度越低),权重(1 - pt) ** self.gamma越大,对总损失F_loss越大
        easy_hard_weight = (1 - pt) ** self.gamma
        F_loss = easy_hard_weight * cross_ent_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


if __name__ == '__main__':
    x_input = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y_target = torch.tensor([0, 2, 1, 2])
    layer = FocalLoss()
    print(layer(x_input, y_target))
