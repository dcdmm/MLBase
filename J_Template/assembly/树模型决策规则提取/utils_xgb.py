def one_split_path(id, df_trees_no_leaf, split_path):
    if id == '0-0':
        return
    y = df_trees_no_leaf['Yes'] == id
    n = df_trees_no_leaf['No'] == id
    if y.sum() == 1:
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
    all_path_path = []
    df_trees_leaf = df_trees[df_trees['Feature'] == 'Leaf']
    df_trees_no_leaf = df_trees[df_trees['Feature'] != 'Leaf']
    for row in df_trees_leaf.iterrows():
        split_path = []
        split_path.append({'Gain': row[1]['Gain']})
        leaf_id = row[1]['ID']
        one_split_path(leaf_id, df_trees_no_leaf, split_path)
        split_path.reverse()
        all_path_path.append(split_path)
    return all_path_path
