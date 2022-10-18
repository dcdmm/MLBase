import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        self.coef_ = None  # 解释变量系数
        self.intercept_ = None  # 截距项
        self._w = None  # 权重向量

    def fit(self, X, y, eta=0.01, n_iters=100000,
            epsilon=1e-10):  # 停止迭代条件
        length = len(y)  # 训练数据样本个数

        def J(wight):
            """模型在权重向量为wight下的损失函数"""
            part_one = np.log(1 + np.exp(X @ wight))
            part_two = (X @ wight) * y

            return np.sum(part_one - part_two) / length

        def dJ():
            """模型梯度向量"""
            part1 = np.exp(X @ self._w)
            part2 = 1 + np.exp(X @ self._w)
            return X.T @ (part1 / part2 - y)

        def gradient_descent():
            """使用梯度下降法训练逻辑斯蒂回归模型"""
            for _ in range(n_iters):
                gradient = dJ()
                last_w = self._w  # 纪录上一次self._w
                self._w = self._w - eta * gradient  # 更新self._w
                if abs(J(self._w) - J(last_w)) < epsilon:
                    break

        X = np.hstack([X, np.ones((len(X), 1))])  # 添加截距项(w, b)--->\beta
        self._w = np.zeros(X.shape[1])  # 权重向量初始为零向量
        gradient_descent()
        self.intercept_ = self._w[-1]  # self._w最后一项为截距项
        self.coef_ = self._w[0:-1]

        return J(self._w)  # 返回迭代完成后模型的损失值

    def predict_proba(self, X_predict):
        """待预测数据集X_predict的y=1概率"""

        def sigmoid(t):
            """计算p(y=1|x)"""
            return np.exp(t) / (1. + np.exp(t))

        X_predict = np.hstack([X_predict, np.ones((len(X_predict), 1))])
        return sigmoid(X_predict.dot(self._w))

    def predict(self, X_predict):
        """待预测数据集X_predict的预测标签"""
        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')  # 概率大于0.5则为1,否则为0

    def score(self, X_t, y_t):
        """根据测试数据集X_t和y_t确定当前模型的准确度"""
        y_predict = self.predict(X_t)
        return accuracy_score(y_t, y_predict)
