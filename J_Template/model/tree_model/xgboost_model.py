import xgboost as xgb
import numpy as np


def MyXgboost(X_train_data, y_train_data, X_test_data, kfold,
              params, early_stopping_rounds=None, verbose_eval=True, feval=None, fweight=None):
    """
    原生xgboost模型封装(具体任务微调)
    Parameters
    ---------
    X_train_data : numpy array (n_sample, n_feature)
        训练数据集
    y_train_data : numpy array (n_sample, 1)
        训练数据集标签
    X_test_data : numpy array (n_sample, n_feature)
        测试数据集
    kfold :
        k折交叉验证对象(也可先生成交叉验证数据)
    params : dict
        xgboost模型train方法params参数
    early_stopping_rounds:
        xgboost模型train方法early_stopping_rounds参数
    verbose_eval :
        xgboost模型train方法verbose_eval参数
    feval :
        xgboost模型train方法feval参数
    fweight : 函数(返回训练数据集的权重)
        返回值为xgboost模型DMatrix方法weight参数

    Returns
    -------
    train_predictions : array
        训练数据集预测结果
    test_predictions : array
        测试数据集预测结果
    model_list : list
        训练模型组成的列表
    """
    num_class = params.get('num_class')  # 多分类问题的判别
    train_predictions = np.zeros(
        X_train_data.shape[0] if num_class is None else [X_train_data.shape[0], num_class])  # 训练数据集预测结果
    test_predictions = np.zeros(
        X_test_data.shape[0] if num_class is None else [X_test_data.shape[0], num_class])  # 测试数据集预测结果
    model_list = list()  # k折交叉验证模型结果
    X_test_data = xgb.DMatrix(X_test_data)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(X_train_data)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = X_train_data[trn_ind], X_train_data[val_ind]
        y_train, y_val = y_train_data[trn_ind], y_train_data[val_ind]

        train_weights = None if fweight is None else fweight(y_train)
        val_weights = None if fweight is None else fweight(y_val)

        train_dataset = xgb.DMatrix(x_train, y_train, weight=train_weights)
        x_val_dataset = xgb.DMatrix(x_val)
        val_dataset = xgb.DMatrix(x_val, y_val, weight=val_weights)

        eval_set = [(train_dataset, "train_"), (val_dataset, "val_")]
        model = xgb.train(params=params,
                          dtrain=train_dataset,
                          evals=eval_set,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval,
                          feval=feval)
        model_list.append(model)
        train_predictions[val_ind] = model.predict(x_val_dataset)
        test_predictions += model.predict(X_test_data) / kfold.n_splits

    return train_predictions, test_predictions, model_list
