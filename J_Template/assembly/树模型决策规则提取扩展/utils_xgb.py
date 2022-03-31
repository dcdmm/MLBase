import copy
from collections import Counter


def one_split_path(tree_id, df_trees_no_leaf, split_path):
    """
    根据节点ID获得`跟节点--->该节点`的划分路径

    Parameters
    ---------
    tree_id : str
        节点ID
    df_trees_no_leaf : pandas DataFrame
       xgboost导出的决策结构中为非叶子节点的部分
    split_path : list
        用于储存单个`跟节点--->该节点`的划分路径
    """
    if tree_id == '0-0':
        return
    y = df_trees_no_leaf['Yes'] == tree_id
    n = df_trees_no_leaf['No'] == tree_id
    # 如果由yes路径划分得
    if y.sum() == 1:
        # 节点信息
        temp_yes = [{'Feature': df_trees_no_leaf[y]['Feature'].values[0]},
                    {'Less_than': 'Yes'},
                    {'Split': df_trees_no_leaf[y]['Split'].values[0]},
                    {'Gain': df_trees_no_leaf[y]['Gain'].values[0]}]
        split_path.append(temp_yes)
        one_split_path(df_trees_no_leaf[y]['ID'].values[0], df_trees_no_leaf, split_path)
    else:
        temp_no = [{'Feature': df_trees_no_leaf[n]['Feature'].values[0]},
                   {'Less_than': 'No'},
                   {'Split': df_trees_no_leaf[n]['Split'].values[0]},
                   {'Gain': df_trees_no_leaf[n]['Gain'].values[0]}]
        split_path.append(temp_no)
        one_split_path(df_trees_no_leaf[n]['ID'].values[0], df_trees_no_leaf, split_path)


def find_all_split_path(df_trees):
    """
    将xgboost导出的决策结构解析为所有叶子节点`跟节点--->该叶子节点`的划分路径

    Parameters
    ---------
    df_trees : pandas DataFrame
        xgboost导出的决策结构

    Returns
    -------
    all_split_path : list
        所有叶子节点`跟节点--->该叶子节点`的划分路径
    """
    all_split_path = []
    # xgboost导出的决策结构中为叶子节点的部分
    df_trees_leaf = df_trees[df_trees['Feature'] == 'Leaf']
    # xgboost导出的决策结构中为非叶子节点的部分
    df_trees_no_leaf = df_trees[df_trees['Feature'] != 'Leaf']
    for row in df_trees_leaf.iterrows():
        # 叶子节点信息
        split_path = [{'Gain': row[1]['Gain']}]
        leaf_id = row[1]['ID']
        one_split_path(leaf_id, df_trees_no_leaf, split_path)
        split_path.reverse()
        all_split_path.append(split_path)
    return all_split_path


def find_class(data, all_split_path):
    """
    根据划分路径确定该划分的类别信息

    Parameters
    ---------
    data : numpy array
        数据集
    all_split_path : list
        划分路径

    Returns
    -------
    distribution : list
        划分路径的类别分布
    split_class : list
        划分路径最有可能属于的类
    """
    distribution = []
    split_class = []
    for split_path in all_split_path:
        s = copy.deepcopy(data)
        # 划分路径最后一个元素为叶子节点信息
        for i in split_path[:-1]:
            feature = int(list(i[0].values())[0][1:])
            less_than = list(i[1].values())[0]
            split = list(i[2].values())[0]
            if less_than == 'No':
                # 'No'表示`>=`,'Yes'表示`<`
                s = s[s[:, feature] >= split]
            else:
                s = s[s[:, feature] < split]
        # 数据集最后一列表示为标签
        res = Counter(s[:, -1])
        distribution.append(dict(res))
        split_class.append(res.most_common(1)[0][0])
    return distribution, split_class
