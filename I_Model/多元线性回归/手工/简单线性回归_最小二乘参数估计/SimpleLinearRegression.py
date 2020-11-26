import numpy as np
from sklearn.metrics import mean_squared_error


class SimpleLinearRegression:
    """形如y=a + bx的简单线性回归"""

    def __init__(self):
        """模型初始化"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training test_text."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)  # 最小二乘参数估计
        self.b_ = y_mean - self.a_ * x_mean  # 截距项

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict,返回x_predict的预测结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training test_text."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x_single,返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_

    @staticmethod
    def score(y_true, y_predict):
        """计算线性回归模型的均方误差"""
        return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)

    def __repr__(self):
        return "SimpleLinearRegression()"
