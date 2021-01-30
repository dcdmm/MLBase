import numpy as np


class Model_original:
    """感知机学习算法的原始形式"""

    def __init__(self):
        self.b = 0  # 初始b0
        self.l_rate = 0.1  # 学习率
        self.w = None

    def sign(self, x, w, b):
        y = np.dot(x, w) + b

        return y

    def fit(self, X_train, y_train):
        self.w = np.ones(len(X_train[0]), dtype=np.float32) - 1
        is_wrong = False
        while not is_wrong:  # 直到没有误分类点才结束循环(默认是线性可分的)
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)  # 梯度下降法(多维)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True


class Model_dual:
    """感知机学习算法的对偶形式"""

    def __init__(self, X_trian, y_trian, l_rate=1):
        self.X_trian = X_trian
        self.y_trian = y_trian
        self.lenght = len(X_trian)
        self.alpha = np.zeros(len(X_trian))
        self.b = 0
        self.l_rate = l_rate

    def gram(self):
        """计算Gram矩阵"""
        matrix = np.zeros((self.lenght, self.lenght))
        for i in range(self.lenght):
            for j in range(self.lenght):
                matrix[i, j] = self.X_trian[i] @ self.X_trian[j]

        return matrix

    def sign(self, x):
        result = (self.alpha * self.y_trian) @ x + self.b

        return result

    def fit(self):
        is_wrong = False
        gram_matrix = self.gram()
        while not is_wrong:  # 默认是线性可分的
            wrong_count = 0
            for k in range(self.lenght):
                if self.y_trian[k] * self.sign(gram_matrix[:, k]) <= 0:
                    self.alpha[k] = self.alpha[k] + self.l_rate
                    self.b = self.b + self.y_trian[k] * self.l_rate
                    wrong_count += 1

            if wrong_count == 0:
                is_wrong = True
