import numpy as np
from im2col import im2col
from col2im import col2im


class Pooling:
    """池化层的实现"""

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h  # 池化窗口的高
        self.pool_w = pool_w  # 池化窗口的宽
        self.stride = stride  # 步幅(这里简化为高宽步幅相等)
        self.pad = pad  # 填充(这里简化为高宽填充相等)
        self.x = None
        self.arg_max = None

    def forward(self, x):
        """池化层的前向传播"""
        N, C, H, W = x.shape
        out_h, out_w, col = im2col(x, self.pool_h, self.pool_w, self.stride,
                                   self.pad)  # col.shape=(N*out_h*out_w, c*self.pool_h*self.pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w)  # col.shape=(N*out_h*out_w*c, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)  # 每一行的最大值(Max池化)
        out = out.reshape((N, out_h, out_w, C)).transpose(0, 3, 1, 2)  # out.shape=(N, C, out_h, out_w)

        self.x = x
        self.arg_max = arg_max  # Max池化

        return out

    def backward(self, dout):
        """池化层的反向传播"""
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


if __name__ == '__main__':
    X = np.arange(16).reshape((1, 1, 4, 4))
    print(X)
    pool = Pooling(3, 3, 2, 1)
    out_put = pool.forward(X)
    print(out_put)
