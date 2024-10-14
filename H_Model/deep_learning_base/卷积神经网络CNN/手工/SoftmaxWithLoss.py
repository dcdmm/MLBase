import numpy as np


def softmax(x):
    """softmax函数"""
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 广播机制
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策

    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y,
                        t):  # 设定t编码方式为独热编码
    """交叉熵函数"""
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class SoftmaxWithLoss:
    """Softmax-with-Loss层"""

    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None

    def forward(self, x,
                t):  # 设定t编码方式为独热编码
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1  # numpy整数索引
            dx = dx / batch_size

        return dx


if __name__ == '__main__':
    t_k = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_k = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    softmax_loss = SoftmaxWithLoss()
    softmax_loss_forward = softmax_loss.forward(y_k, t_k)
    print(softmax_loss_forward)
    print(softmax_loss.backward())
