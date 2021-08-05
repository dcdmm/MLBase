def MyXgbLgbCat(X_train, y_train, model, kf, model_params, fit_params):
    models = []
    score_lst = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        print(f'KFold: {fold}')
        x_tr, x_va = X_train[tr_idx], X_train[va_idx]
        if model.__name__.__contains__("XGB"):
            y_tr, y_va = y_train[tr_idx].reshape(-1, 1), y_train[va_idx].reshape(-1, 1)
        else:
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        modelapp = model(**model_params)
        modelapp.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)], **fit_params)
        score = modelapp.score(x_va, y_va)
        print("{0}_score:{1}".format(model.__name__, score))
        score_lst.append(score)
        models.append(modelapp)

    return models, score_lst
