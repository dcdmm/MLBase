import numpy as np
from collections import Counter


class KNNClassifier:

    def __init__(self, n_neighbors=3, p=2):
        """初始化kNN分类器"""
        assert n_neighbors >= 1, "k must be valid"
        self.k = n_neighbors  # 临近点个数
        self.p = p  # Minkowski距离选择(默认p=2,欧氏距离)
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"  # ★★★X_predict必须为矩阵

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x,返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        distances = [np.linalg.norm(x_train - x, ord=self.p) for x_train in self._X_train]  # 利用范数表示结果
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]  # 选取最近的k个点的label
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]  # 使用出现次数最多的label作为待预测数据的label

    def score(self, X_test, y_test):
        """检验模型分类的正确率"""
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X.reshape(-1, len(X)))  # 转换为矩阵形式再进行预测
            if label == y:
                right_count += 1

        return right_count / len(X_test)
