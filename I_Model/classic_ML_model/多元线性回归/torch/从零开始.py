import torch
import matplotlib.pyplot as plt

torch.manual_seed(10)  # CPU下随机数种子设置

lr = 0.05

# 创建训练数据
x = torch.rand(20, 1) * 10
y = 2 * x + (5 + torch.randn_like(x))

# 初始化线性回归参数
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for iteration in range(1000):

    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)
    loss = (0.5 * (y - y_pred) ** 2).mean()  # MSE loss

    loss.backward()
    b.data.sub_(lr * b.grad)  # 更新b
    w.data.sub_(lr * w.grad)

    w.grad.zero_()  # 手动使梯度清零
    b.grad.zero_()

    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss < 1:
            break
