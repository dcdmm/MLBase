import lightgbm as lgb
import numpy as np


def MyLightGBM(X_train_data, y_train_data, X_test_data, kfold,
               params, callbacks=None, feval=None, fweight=None,
               categorical_feature="auto"):
    """
    原生lightgbm模型封装(具体任务对应修改)

    Parameters
    ---------
    X_train_data : numpy array
        shape=(n_sample, n_feature)
        训练数据集
    y_train_data : numpy array
        shape=(n_sample, )
        训练数据集标签
    X_test_data : numpy array
        shape=(n_sample, n_feature)
        测试数据集
    kfold :
        k折交叉验证对象(也可先生成交叉验证文件)
    params : dict
        lightgbm模型train方法params参数
    callbacks : []
        lightgbm模型callbacks参数
    feval :
        lightgbm模型train方法feval参数
    fweight : 函数(返回训练数据集的权重)
        返回值为lightgbm模型Dataset方法weight参数
    categorical_feature : list(分类特征的索引) or 'auto'
        lightgbm模型Dataset方法categorical_feature参数

    Returns
    -------
    train_predictions : array
        多分类:shape=(n_sample, n_class)  二分类或回归:shape=(n_sample, )
        训练数据集预测结果
    test_predictions : array
        多分类:shape=(n_sample, n_class)  二分类或回归:shape=(n_sample, )
        测试数据集预测结果
    model_list : list
        训练模型组成的列表
    """
    num_class = params.get('num_class')  # 多分类问题的判别
    train_predictions = np.zeros(
        X_train_data.shape[0] if num_class is None else [X_train_data.shape[0], num_class])  # 训练数据集预测结果
    test_predictions = np.zeros(
        X_test_data.shape[0] if num_class is None else [X_test_data.shape[0], num_class])  # 测试数据集预测结果

    # 警告的避免
    if 'num_boost_round' in params:
        num_boost_round = params.pop('num_boost_round')
    else:
        num_boost_round = 1000

    model_list = list()  # k折交叉验证模型结果
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(X_train_data)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = X_train_data[trn_ind], X_train_data[val_ind]
        y_train, y_val = y_train_data[trn_ind], y_train_data[val_ind]

        train_weights = None if fweight is None else fweight(y_train)
        val_weights = None if fweight is None else fweight(y_val)

        train_dataset = lgb.Dataset(x_train, y_train, weight=train_weights, categorical_feature=categorical_feature)
        val_dataset = lgb.Dataset(x_val, y_val, weight=val_weights, categorical_feature=categorical_feature)

        model = lgb.train(params=params,
                          train_set=train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          num_boost_round=num_boost_round,
                          callbacks=callbacks,
                          feval=feval)
        model_list.append(model)
        train_predictions[val_ind] = model.predict(x_val)
        test_predictions += model.predict(X_test_data) / kfold.n_splits

    return train_predictions, test_predictions, model_list
