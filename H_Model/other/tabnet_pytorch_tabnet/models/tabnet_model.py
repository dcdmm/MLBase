import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor


def MyTabnet(X_train_data, y_train_data, X_test_data, kfold, tabnet_params, fit_params):
    """
    Tabnet模型封装(回归为例;具体任务对应修改)
    Parameters
    ---------
    X_train_data : numpy array (n_sample, n_feature)
        训练数据集
    y_train_data : numpy array (n_sample, )
        训练数据集标签
    X_test_data : numpy array (n_sample, n_feature)
        测试数据集
    kfold :
        k折交叉验证对象(也可先生成交叉验证数据)
    tabnet_params : dict
        tabnet模型params参数
    fit_params : dict
        tabnet模型fit方法参数

    Returns
    -------
    train_predictions : array (n_sample, 1)
        训练数据集预测结果
    test_predictions : array (n_sample, )
        测试数据集预测结果
    model_list : list
        训练模型组成的列表
    """
    train_predictions = np.zeros((X_train_data.shape[0], 1))
    test_predictions = np.zeros(X_test_data.shape[0])
    model_list = list()  # k折交叉验证模型结果
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(X_train_data)):
        print(f'Training fold {fold + 1}')
        clf = TabNetRegressor(**tabnet_params)

        fit_params["X_train"] = X_train_data[trn_ind]
        fit_params["y_train"] = y_train_data[trn_ind].reshape(-1, 1)
        fit_params["eval_set"] = [(X_train_data[val_ind], y_train_data[val_ind].reshape(-1, 1))]

        clf.fit(**fit_params)
        model_list.append(clf)

        train_predictions[val_ind] = clf.predict(X_train_data[val_ind])
        test_predictions += clf.predict(X_test_data).flatten() / kfold.n_splits

    return test_predictions, train_predictions, model_list
