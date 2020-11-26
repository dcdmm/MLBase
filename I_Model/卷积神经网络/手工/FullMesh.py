import numpy as np


class FullMesh:
    """全连接层的实现"""

    def __init__(self, W, b):
        self.W = W  # 权重矩阵
        self.b = b  # 偏置
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """全连接层前向传播"""
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b  # 这里采用out = x @ W + b(即行向量形式)

        return out

    def backward(self, dout):
        """全连接层反向传播"""
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 恢复成池化层输出形状

        return dx
