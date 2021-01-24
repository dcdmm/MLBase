from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module, ABC):
    """残差块单元的简单实现"""

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)

        return F.relu(Y + X)


if __name__ == '__main__':
    img = torch.randn((4, 3, 6, 6))
    blk = Residual(3, 3)
    print(blk(img).shape)

    blk1 = Residual(3, 6, use_1x1conv=True, stride=2)  # 也可以增加输出的通道数同时减半输出的高和宽
    print(blk1(img).shape)
