# first load ds
# then split again for validation
# then construct catboost ensemble
# then fit
# output predictions on test set

from methods import CatboostMethod
import numpy as np
import sklearn.model_selection
import torch
from metric import amex_metric_np
import optuna
from optuna.integration import CatBoostPruningCallback
import argparse

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    depth = trial.suggest_int("depth", 3, 11)
    l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 3, 30)
    #use_class_weights = trial.suggest_int("use_class_weights", 0, 1)
    class_weights = [
        np.count_nonzero(train_y == 0) / len(train_y),
        np.count_nonzero(train_y == 1) / len(train_y),
    ]
    print(f'Determined class weights {class_weights}')
    catboost_options = {
        "iterations": 400,
        "l2_leaf_reg": l2_leaf_reg,
        "depth": depth,
        "learning_rate": learning_rate,
        "verbose": True,
        "task_type": "GPU" if torch.cuda.is_available() else None,
        "eval_metric": "Accuracy",
        #"class_weights": class_weights if use_class_weights == 1 else None,
        "auto_class_weights": "Balanced",
        "random_seed": 42, # Make more reproducible
        "use_best_model": True,
        "od_type": "Iter",
        "od_wait": 50,
    }
    catboost_fit_options = {
        "callbacks": [CatBoostPruningCallback(trial, "Accuracy")],
    }

    method = CatboostMethod(config=catboost_options)
    method.train(X_train_floats, X_val_floats, X_train_cat, X_val_cat, y_train, y_val, fit_kwargs=catboost_fit_options)

    preds = method.eval(X_val_floats, X_val_cat)
    metric = amex_metric_np(y_val, preds)
    return metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)

    args = parser.parse_args()

    data = np.load(args.data)
    train_floats = data['train_floats']
    train_cat = data['train_cat']
    train_y = data['train_y']
    test_floats = data['test_floats']
    test_cat = data['test_cat']
    test_y = data['test_y']

    X_train_floats, X_val_floats, X_train_cat, X_val_cat, y_train, y_val = sklearn.model_selection.train_test_split(
        train_floats, train_cat, train_y
    )

    study = optuna.create_study(direction="maximize", storage='sqlite:///optuna.db', load_if_exists=True, study_name="amex_predict")
    study.optimize(objective, n_trials=3)
    best_params = study.best_params
    print(best_params)