def one_split_path(tree_id, df_trees_no_leaf, split_path):
    """
    根据叶子节点ID获得该叶子节点的划分路径

    Parameters
    ---------
    tree_id : str
        节点ID
    df_trees_no_leaf : pandas DataFrame
       xgboost导出的决策结构中为叶子节点的部分
    split_path : list
        用于储存单个叶子节点的划分路径
    """
    if tree_id == '0-0':
        return
    y = df_trees_no_leaf['Yes'] == tree_id
    n = df_trees_no_leaf['No'] == tree_id
    # 如果由yes路径划分得
    if y.sum() == 1:
        # 单次划分信息
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
    将xgboost导出的决策结构解析为所有叶子节点的划分路径

    Parameters
    ---------
    df_trees : pandas DataFrame
        xgboost导出的决策结构

    Returns
    -------
    all_split_path : list
        所有叶子节点的划分路径
    """
    all_split_path = []
    # xgboost导出的决策结构中为叶子节点的部分
    df_trees_leaf = df_trees[df_trees['Feature'] == 'Leaf']
    # xgboost导出的决策结构中不为叶子节点的部分
    df_trees_no_leaf = df_trees[df_trees['Feature'] != 'Leaf']
    for row in df_trees_leaf.iterrows():
        # 叶子节点信息
        split_path = [{'Gain': row[1]['Gain']}]
        leaf_id = row[1]['ID']
        one_split_path(leaf_id, df_trees_no_leaf, split_path)
        split_path.reverse()
        all_split_path.append(split_path)
    return all_split_path
