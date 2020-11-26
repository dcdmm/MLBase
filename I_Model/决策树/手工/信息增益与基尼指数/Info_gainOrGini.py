import numpy as np
from collections import Counter


class Info_gainOrGini:

    def __init__(self, max_depth, criterion='entropy'):
        self.max_depth = max_depth  # 树的最大深度
        self.criterion = criterion  # The function to measure the quality of a split
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train  # X_train为连续型数据

        def split_data(X, y, d, value):
            """在维度d上根据阙值value划分数据集合X,y"""
            index_a = (X[:, d] <= value)
            index_b = (X[:, d] > value)

            return X[index_a], X[index_b], y[index_a], y[index_b]

        def entropy(y):
            """计算y的熵"""
            counter = Counter(y)
            pro_vector = np.array(list(counter.values())) / len(y)  # 不同类别所占的比例
            res = - pro_vector @ np.log(pro_vector)

            return res

        def gini(y):
            """计算y的基尼指数"""
            counter = Counter(y)
            pro_vector = np.array(list(counter.values())) / len(y)  # 不同类别所占的比例
            res = 1 - pro_vector @ pro_vector

            return res

        def try_split_data(X, y):
            """找出y上最好的划分"""
            best_ent_g = float('inf')
            best_d = -1  # 此次划分所选的特征
            best_v = -1  # 此特征上最优的划分点

            for d in range(X.shape[1]):
                sorted_index = np.argsort(X[:, d])
                for i in range(1, len(X)):
                    if X[sorted_index[i], d] != X[sorted_index[i - 1], d]:
                        v = (X[sorted_index[i], d] + X[sorted_index[i - 1], d]) / 2  # 连续值处理
                        X_l, X_r, y_l, y_r = split_data(X, y, d, v)
                        p_l, p_r = len(X_l) / len(X), len(X_r) / len(X)
                        if self.criterion == 'entropy':
                            e = p_l * entropy(y_l) + p_r * entropy(y_r)  # 条件熵
                            if e < best_ent_g:
                                best_ent_g, best_d, best_v = e, d, v
                        else:
                            g = p_l * gini(y_l) + p_r * gini(y_r)  # 基尼指数
                            if g < best_ent_g:
                                best_ent_g, best_d, best_v = g, d, v

            return best_ent_g, best_d, best_v

        if self.criterion == 'entropy':
            print('criterion=entropy')
            while self.max_depth:
                ent = entropy(y_train)
                best_ent_g, best_d, best_v = try_split_data(self.X_train, self.y_train)
                print("info_gain =", ent - best_ent_g)  # 使用信息增益进行决策树的划分特征选择
                print("best_d =", best_d)
                print("best_v =", best_v, end='\n\n')

                X_l, X_r, y_l, y_r = split_data(self.X_train, self.y_train, best_d, best_v)
                if entropy(y_l) > entropy(y_r):  # 进行决策树的下一次划分
                    self.X_train = X_l
                    self.y_train = y_l
                else:
                    self.X_train = X_r
                    self.y_train = y_r

                self.max_depth -= 1
        else:
            print('criterion =gini')
            while self.max_depth:
                best_ent_g, best_d, best_v = try_split_data(self.X_train, self.y_train)
                print("best_g =", best_ent_g)  # 使用基尼指数进行决策树的划分特征选择
                print("best_d =", best_d)
                print("best_v =", best_v, end='\n\n')

                X_l, X_r, y_l, y_r = split_data(self.X_train, self.y_train, best_d, best_v)
                if gini(y_l) > gini(y_r):
                    self.X_train = X_l
                    self.y_train = y_l
                else:
                    self.X_train = X_r
                    self.y_train = y_r

                self.max_depth -= 1
