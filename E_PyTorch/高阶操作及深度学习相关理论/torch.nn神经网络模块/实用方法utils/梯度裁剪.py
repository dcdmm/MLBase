from abc import ABC
import torch
import torch.nn as nn

x0, y0 = torch.normal(mean=1.7, std=1, size=(100, 2)) + 1, torch.zeros(100)  # 数据集1
x1, y1 = torch.normal(mean=-1.7, std=1, size=(100, 2)) + 1, torch.ones(100)  # 数据集2
train_x, train_y = torch.cat((x0, x1), 0), torch.cat((y0, y1), 0)


class LR(nn.Module, ABC):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


lr_net = LR()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(lr_net.parameters(), lr=0.01, momentum=0.9)

for iteration in range(500):
    optimizer.zero_grad()  # 梯度清零
    y_pred = lr_net(train_x)
    loss = loss_fn(y_pred.squeeze(), train_y)
    loss.backward()

    # 梯度裁剪示例1
    # nn.utils.clip_grad_norm_(lr_net.parameters(),
    #                          max_norm=3,  # 梯度的最大范数
    #                          norm_type=2)  # p范数的类型,inf表示无穷范数

    # 梯度裁剪示例2
    nn.utils.clip_grad_value_(lr_net.parameters(),
                              clip_value=5)  # 梯度的范围[-clip_value, clip_value]

    optimizer.step()

    if iteration % 20 == 0:
        print("iteration：{iteration},    loss: {loss}".format(iteration=iteration, loss=loss))
