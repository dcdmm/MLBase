import numpy as np


class Relu:
    """Relu层"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    """Sigmoid层"""

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


if __name__ == '__main__':
    arr_test = np.linspace(-5, 5, 100)
    relu = Relu()
    sigmoid = Sigmoid()
    relu_forward = relu.forward(arr_test)
    sigmoid_forward = sigmoid.forward(arr_test)

    import matplotlib.pyplot as plt
    plt.plot(arr_test, relu_forward, color='red')
    plt.plot(arr_test, sigmoid_forward, color='b')
    plt.show()
