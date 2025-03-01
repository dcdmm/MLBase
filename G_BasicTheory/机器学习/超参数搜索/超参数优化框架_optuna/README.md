
参考链接:https://optuna.readthedocs.io/en/stable/

## Pythonic Search Space

* optuna.trial.Trial.suggest_categorical() for categorical parameters

* optuna.trial.Trial.suggest_int() for integer parameters

* optuna.trial.Trial.suggest_float() for floating point parameters

With optional arguments of step and log, we can discretize or take the logarithm of integer and floating point parameters.

```python
import optuna


def objective(trial):
    # Categorical parameter
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

    # Integer parameter
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Integer parameter (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # Integer parameter (discretized)
    num_units = trial.suggest_int("num_units", 10, 100, step=5)

    # Floating point parameter
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Floating point parameter (discretized)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
```

## Defining Parameter Spaces

In Optuna, we define search spaces using familiar Python syntax including conditionals and loops.

Also, you can use branches or loops depending on the parameter values.

* Branches:

```python
import sklearn.ensemble
import sklearn.svm


def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)
```

* Loops:
```python
import torch
import torch.nn as nn


def create_model(trial, in_size):
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, 10))

    return nn.Sequential(*layers)
```


## Sampling Algorithms

Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values, leading to an optimal search space which giving off parameters leading to better objective values. More detailed explanation of how samplers suggest parameters is in optuna.samplers.BaseSampler.

Optuna provides the following sampling algorithms:

* Tree-structured Parzen Estimator algorithm implemented in optuna.samplers.TPESampler

* CMA-ES based algorithm implemented in optuna.samplers.CmaEsSampler

* Grid Search implemented in optuna.samplers.GridSampler

* Random Search implemented in optuna.samplers.RandomSampler

The default sampler is optuna.samplers.TPESampler.


## Pruning Algorithms

Pruners automatically stop unpromising trials at the early stages of the training (a.k.a., automated early-stopping).

Optuna provides the following pruning algorithms:

* Asynchronous Successive Halving algorithm implemented in optuna.pruners.SuccessiveHalvingPruner

* Hyperband algorithm implemented in optuna.pruners.HyperbandPruner

* Median pruning algorithm implemented in optuna.pruners.MedianPruner

* Threshold pruning algorithm implemented in optuna.pruners.ThresholdPruner

We use optuna.pruners.MedianPruner in most examples, though basically it is outperformed by optuna.pruners.SuccessiveHalvingPruner and optuna.pruners.HyperbandPruner as in this benchmark result.