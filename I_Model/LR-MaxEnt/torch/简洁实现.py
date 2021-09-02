from abc import ABC
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(10)

sample_nums = 100
mean = 1.7
std = 1
bias = 1

# 数据集1
x0 = torch.normal(mean=mean, std=std, size=(sample_nums, 2)) + bias
y0 = torch.zeros(sample_nums)

# 数据集2
x1 = torch.normal(mean=-mean, std=std, size=(sample_nums, 2)) + bias
y1 = torch.ones(sample_nums)

train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)


class LR(nn.Module, ABC):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """模型的前向传播"""
        x = self.features(x)
        x = self.sigmoid(x)  # 逻辑斯蒂回归使用的为sigmoid函数
        return x


lr_net = LR()
loss_fn = nn.BCELoss()  # 交叉熵损失函数
lr = 0.01  # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

for iteration in range(1000):
    optimizer.zero_grad()  # 梯度清零
    y_pred = lr_net(train_x)  # 前向传播
    loss = loss_fn(y_pred.squeeze(), train_y)
    loss.backward()  # 反向传播
    optimizer.step()

    if iteration % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()
        acc = correct.item() / train_y.shape[0]  # 计算分类准确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1
        plt.plot(plot_x, plot_y)

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()
        plt.show()

        if acc > 0.99:  # 分类准确率大于0.99时退出循环
            break
