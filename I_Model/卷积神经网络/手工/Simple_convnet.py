import numpy as np
from collections import OrderedDict  # 有序字典
from Convolution import Convolution  # 相对路径导入,也可以使用绝对路径进行导入
from SoftmaxWithLoss import SoftmaxWithLoss
from FullMesh import FullMesh
from Activations import Relu
from Pooling import Pooling


class SimpleConvNet:
    """
    CNN网络构成:卷积层 --> relu层 --> 池化层 --> 全连接层 --> relu层 --> 全连接层 --> softmax and cross entropy
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param=None,
                 pool_param=None,
                 hidden_size=100,  # 两个全连接中隐藏层的大小
                 output_size=10,  # 输出大小
                 weight_init_std=0.01):
        if conv_param is None:
            conv_param = {'filter_num': 30,  # 卷积层滤波器数量
                          'filter_size': 5,  # 滤波器大小(这里简化为滤波器高宽相等)
                          'pad': 0,  # 卷积层填充(这里简化为高宽填充相等)
                          'stride': 1}  # 卷积层步幅(这里简化为高宽步幅相等)
        if pool_param is None:
            pool_param = {'pool_h': 2,  # 池化窗口的高
                          'pool_w': 2,  # 池化窗口的宽
                          'pool_pid': 0,  # 池化层填充(这里简化为高宽填充相等)
                          'pool_stride': 1}  # 池化层步幅(这里简化为高宽步幅相等)
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        pool_h = pool_param['pool_h']
        pool_w = pool_param['pool_w']
        pool_pad = pool_param['pool_pid']
        pool_stride = pool_param['pool_stride']

        conv_output_size = int(np.floor((input_dim[1] - filter_size + 2*filter_pad + filter_stride) / filter_stride))  # 卷积层每个通道的形状
        pool_out_h = int(np.floor((conv_output_size + 2 * pool_pad - pool_h + pool_stride) / pool_stride))  # 池化层输出每个通道的高
        pool_out_w = int(np.floor((conv_output_size + 2 * pool_pad - pool_w + pool_stride) / pool_stride))  # 池化层输出每个通道的宽
        pool_output_size = filter_num * pool_out_h * pool_out_w  # 第一个全连接层的输入数

        # 参数初始化
        self.params = {'W1': weight_init_std *
                       np.random.randn(filter_num, input_dim[0], filter_size, filter_size),  # 卷积层滤波器
                       'b1': np.zeros(filter_num),  # 卷积层偏置
                       'W2': weight_init_std *
                       np.random.randn(pool_output_size, hidden_size),  # 第一个全连接层权重
                       'b2': np.zeros(hidden_size),  # 第一个全连接层偏置
                       'W3': weight_init_std *
                       np.random.randn(hidden_size, output_size),  # 第二个全连接层权重
                       'b3': np.zeros(output_size)}  # 第二个全连接层偏置

        # 网络层生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=pool_h, pool_w=pool_w,
                                       stride=pool_stride, pad=pool_pad)
        self.layers['FullMesh1'] = FullMesh(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['FullMesh2'] = FullMesh(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """CNN关于x的预测"""
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """CNN关于x的预测与t的交叉熵"""
        y = self.predict(x)

        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """计算x关于t准确率"""
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        acc = acc / x.shape[0]

        return acc

    def gradient(self, x, t):
        """计算每一层的梯度"""
        self.loss(x, t)
        dout = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()  # 反向传播
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}  # 保存梯度信息
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['FullMesh1'].dW, self.layers['FullMesh1'].db
        grads['W3'], grads['b3'] = self.layers['FullMesh2'].dW, self.layers['FullMesh2'].db

        return grads


if __name__ == "__main__":
    ob = SimpleConvNet()
    x = np.arange(7840).reshape((10, 1, 28, 28))
    pre_x = ob.predict(x)
    print(pre_x.shape)
