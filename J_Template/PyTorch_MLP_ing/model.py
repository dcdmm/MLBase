import torch.nn as nn
from abc import ABC


class MLP(nn.Module, ABC):
    def __init__(self):
        super(MLP, self).__init__()
        self.features0 = nn.Linear(49, 10)
        self.dropout0 = nn.Dropout(0.25)
        self.relu0 = nn.ReLU()
        self.features1 = nn.Linear(10, 5)
        self.dropout1 = nn.Dropout(0.1)
        self.features2 = nn.Linear(5, 1)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.features0(x)
        x = self.dropout0(x)
        x = self.relu0(x)
        x = self.features1(x)
        x = self.dropout1(x)
        x = self.features2(x)
        x = self.sigmoid1(x)  # 激活函数的选择
        return x
