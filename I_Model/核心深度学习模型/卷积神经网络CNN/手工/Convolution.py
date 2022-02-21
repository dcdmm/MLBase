import numpy as np
from im2col import im2col
from col2im import col2im


class Convolution:
    """卷积层的实现"""

    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # 滤波器
        self.b = b  # 偏置
        self.stride = stride  # 步幅(这里简化为高宽步幅相等)
        self.pad = pad  # 填充(这里简化为高宽填充相等)
        self.x = None  # 输入数据
        self.col = None  # x的合适的二维展开
        self.col_W = None  # 滤波器合适的二维展开
        self.dW = None
        self.db = None

    def forward(self, x):
        """卷积层的前向传播"""
        N, C, H, W = x.shape  # batch_num, channel, height, width
        FN, C, FH, FW = self.W.shape  # 滤波器数量, channel, height, width;滤波器通道数必须与输入数据通道数相等
        out_h, out_w, col = im2col(x, FH, FW, self.stride,
                                   self.pad)  # (输出高,输出宽,x的二维展开);col.shape=(N*out_h*out_w, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T  # 滤波器合适的二维展开(C*FH*FW, FN)
        out = np.dot(col, col_W) + self.b  # out.shape=(N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # out.shape=(N, FN, out_h, out_w)
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """卷积层的反向传播"""
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose((1, 0)).reshape((FN, C, FH, FW))

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


if __name__ == '__main__':
    image = np.arange(720).reshape((10, 2, 6, 6))
    kernel = np.arange(90).reshape((5, 2, 3, 3))
    bias = np.ones(5)
    con = Convolution(kernel, bias)
    output = con.forward(image)
    print(output.shape)
