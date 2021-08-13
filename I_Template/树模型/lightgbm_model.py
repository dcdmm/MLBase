import lightgbm as lgb
import numpy as np


def MyLightGBM(X_train_data, y_train_data, X_test_data, kfold,
               params, early_stopping_rounds=None, verbose_eval=True, feval=None, fweight=None,
               categorical_feature="auto"):
    num_class = params.get('num_class')  # 多分类问题的判别
    train_predictions = np.zeros(
        X_train_data.shape[0] if num_class is None else [X_train_data.shape[0], num_class])  # 训练数据集预测结果
    test_predictions = np.zeros(
        X_test_data.shape[0] if num_class is None else [X_test_data.shape[0], num_class])  # 测试数据集预测结果
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
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval,
                          feval=feval)
        model_list.append(model)
        train_predictions[val_ind] = model.predict(x_val)
        test_predictions += model.predict(X_test_data) / kfold.n_splits

    return train_predictions, test_predictions, model_list
