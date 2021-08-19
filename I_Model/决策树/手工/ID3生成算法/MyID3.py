
import numpy as np
from collections import Counter


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root  # 是否为叶节点
        self.label = label  # 叶节点所属的类
        self.feature = feature
        # 分割训练数据集的特征
        self.feature_name = feature_name
        self.tree = {}
        self.result = {
            'root': self.root,
            'label:': self.label,
            'feature': self.feature,
            'feature_name': self.feature_name,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class MyID3:
    def __init__(self, epsilon=0.01):
        # 决策树(信息增益)阙值
        self.epsilon = epsilon
        self._tree = {}

    @staticmethod
    def calc_ent(datasets):
        """计算熵"""
        counter = Counter(
            datasets[:, -1])  # 不同类的个数
        pro_vector = np.array(list(
            counter.values())) / len(datasets)  # 不同类所占的比例
        res = - \
            pro_vector @ np.log2(
                pro_vector)

        return res

    def cond_ent(self, datasets, axis=0):
        """计算条件熵"""
        conter = Counter(
            datasets[:, axis])  # 特征datasets[:, axis]不同取值的个数
        data_length = len(
            datasets)
        # 特征datasets[:, axis]不同取值所占的比例
        pro_vector = np.array(list(
            conter.values())) / data_length
        hd_vector = list()
        for i in conter.keys():
            hd_vector.append(self.calc_ent(
                datasets[np.argwhere(datasets == i)[:, 0]]))
        result = pro_vector @ hd_vector

        return result

    @staticmethod
    def info_gain(ent, cond_ent):
        """信息增益"""
        return ent - cond_ent

    def info_gain_train(self, datasets):
        """找出信息增益最大的特征"""
        count = len(
            datasets[0]) - 1
        ent = self.calc_ent(
            datasets)  # 经验熵H(D)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(
                datasets, axis=c))  # 不同特征划分的信息增益g(D,A)
            best_feature.append(
                (c, c_info_gain))
        best_ = max(
            best_feature, key=lambda x: x[-1])

        return best_

    def train(self,
              train_data):  # DataFrame格式
        """"""
        y_train, features = train_data.iloc[:, -
                                            1], train_data.columns[: -1]
        # 1,若D中所有实例属于同一类Ck,则T为单结点树,并将类Ck作为该结点的类标记,返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])

        # 2,若A为空,则T为单结点树,将D中实例数最大的类Ck作为该节点的类标记,返回T
        if len(features) == 0:
            return Node(
                root=True,
                label=y_train.value_counts().sort_values(
                    ascending=False).index[0])

        # 3,计算A中各特征对D的信息增益,选择信息增益最大的特征Ag
        max_feature, max_info_gain = self.info_gain_train(
            np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树,并将D中是实例数最大的类Ck作为该节点的类标记,返回T
        if max_info_gain < self.epsilon:
            return Node(
                root=True,
                label=y_train.value_counts().sort_values(
                    ascending=False).index[0])

        # 5,构建Ag的子结点,由结点及其子结点构造的树T
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts(
        ).index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop(
                [max_feature_name], axis=1)

            # 6,递归生成树
            sub_tree = self.train(
                sub_train_df)
            node_tree.add_node(
                f, sub_tree)

        return node_tree

    def fit(self, train_data):
        self._tree = self.train(
            train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)
