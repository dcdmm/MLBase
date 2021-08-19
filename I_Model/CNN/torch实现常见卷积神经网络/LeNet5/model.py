from abc import ABC

import torch
import torch.nn as nn


class LeNet5(nn.Module, ABC):
    """LeNet5网络结构"""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.featrues = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 卷积核大小为5*5
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 步幅为2,池化窗口大小为2*2
            nn.Conv2d(6, 16, 5),  # 卷积核大小为5*5
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # 步幅为2,池化窗口大小为2*2

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10))

    def forward(self, x):
        """前向传播"""
        x = self.featrues(x)
        x = x.reshape(x.size()[0], -1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    test_img = torch.randn((1, 1, 32, 32))  # 1张图片
    net = LeNet5()
    print(net(test_img))
