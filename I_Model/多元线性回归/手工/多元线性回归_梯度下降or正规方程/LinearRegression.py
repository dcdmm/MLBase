from sklearn.metrics import r2_score
import numpy as np


class LinearRegression:
    """形如Y=a1 + a2x2 + a2x3 + ...+ akxk的多元线性回归"""

    def __init__(self):
        """模型初始化"""
        self.coef_ = None  # 解释变量系数
        self.intercept_ = None  # 截距项
        self._theta = None  # 模型参数([截距项,解释变量系数])

    def fit_normal(self, X_train, y_train):
        """正规方程求解"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, lr=0.01, n_iters=1e4):
        """批量梯度下降法求解(BGD)"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def value(theta, X_b, y):
            """计算损失函数的值"""
            return ((y - (X_b @ theta)).T @ (y - (X_b @ theta))) / (2 * len(y))

        def d_value(theta, X_b, y):
            """计算损失函数的梯度矩阵"""
            return X_b.T.dot(X_b.dot(theta) - y) / len(y)  # 梯度矩阵

        def gradient_descent(X_b, y, initial_theta, lr, n_iters=1e4, epsilon=1e-8):
            """
             批量梯度下降法(BGD)迭代求解模型参数
            :param X_b: 数据矩阵(包含截距项)
            :param y: 样本真实值向量
            :param initial_theta: 初始值
            :param lr: 学习率
            :param n_iters: 最高迭代轮数
            :param epsilon: 迭代结束时必须满足的精度
            :return:
            """

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = d_value(theta, X_b, y)
                last_theta = theta
                theta = theta - lr * gradient  # 迭代值更新过程
                if abs(value(theta, X_b, y) - value(last_theta, X_b, y)) < epsilon:  # 比较2次迭代值的差异,达到精度要求时跳出循环
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(
            X_b, y_train, initial_theta, lr, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, X_train, y_train,
                n_iters=50,  # 训练轮数
                t0=5, t1=50):
        """随机梯度下降法(SGD)迭代求解模型参数"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            """单个样本的损失函数"""
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta,  # 初始值
                n_iters=5, t0=5, t1=50):

            def learning_rate(t):
                """学习率衰减"""
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            for i_iter in range(n_iters):
                indexes = np.random.permutation(m)  # 训练样本重新随机排序
                X_b_new = X_b[indexes, :]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(i_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  # 模型包含截距项
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回X_predict的预测结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    @staticmethod
    def score(y_true, y_predict):
        """计算线性回归模型的可决系数(R平方)"""
        return 1 - r2_score(y_true, y_predict) / np.var(y_true)
