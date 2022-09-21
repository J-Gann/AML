# 220920 Results on Hyperparameter Optimization
#[I 2022-09-20 18:55:41,396] Trial 265 finished with value: 0.7666563710701346 and parameters: {'learning_rate': 0.09949402494302435, 'depth': 10, 'l2_leaf_reg': 14}. Best is trial 35 with value: 0.787402148343339
#{'depth': 10, 'l2_leaf_reg': 10, 'learning_rate': 0.04111238193128377}

from methods import CatboostMethod
import numpy as np
import sklearn.model_selection
from metric import amex_metric_np
import argparse
import pathlib
import pandas as pd

import logging
log = logging.getLogger(__name__)
FORMAT = "%(asctime)s %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logging.root.setLevel(logging.DEBUG)


def train(X_train_floats, X_train_cat, y_train, X_val_floats, X_val_cat, y_val):
    catboost_options = {
        "iterations": 400,
        "l2_leaf_reg": 10,
        "depth": 10,
        "learning_rate": 0.04111238193128377,
        "verbose": True,
        "eval_metric": "Accuracy",
        "auto_class_weights": "Balanced",
        "use_best_model": True,
        "od_type": "Iter",
        "od_wait": 50,
    }
    method = CatboostMethod(config=catboost_options)
    method.train(X_train_floats, X_val_floats, X_train_cat, X_val_cat, y_train, y_val)

    preds = method.eval(X_val_floats, X_val_cat)
    metric = amex_metric_np(y_val, preds)
    log.info(f'Classifier trained with metric: {metric}')
    return method

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True, help='Where to save Model Checkpoints etc.')

    args = parser.parse_args()

    data = np.load(args.data)
    output_path = pathlib.Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True,)

    train_floats = data['train_floats']
    train_cat = data['train_cat']
    train_y = data['train_y']
    test_floats = data['test_floats']
    test_cat = data['test_cat']
    test_y = data['test_y']


    kf = sklearn.model_selection.KFold(
        n_splits=5, shuffle=True, random_state=42,
    )
    kf_index = 0
    ensemble = []
    for train_index, val_index in kf.split(train_floats):
        model_path = (output_path / f"model_{kf_index}.gbm")
        if model_path.exists():
            log.info(f'Model path {model_path} exists, loading instead')
            method = CatboostMethod()
            method.load(model_path)
        else:
            log.info(f'Training Model Nr. {kf_index}')
            X_train_f = train_floats[train_index]
            X_train_c = train_cat[train_index]
            X_val_f = train_floats[val_index]
            X_val_c = train_cat[val_index]
            y_train = train_y[train_index]
            y_val = train_y[val_index]
            method = train(X_train_f, X_train_c, y_train, X_val_f, X_val_c, y_val)
            method.save(model_path)
        ensemble.append(method)
        kf_index += 1

    print('Model training/loading done. Predict..')
    all_preds = []
    for method in ensemble:
        preds = method.eval(test_floats, test_cat)
        all_preds.append(preds)
    all_preds = np.array(all_preds)
    pd.DataFrame(all_preds.mean(axis=0)).to_csv(output_path / "preds.csv")
    import ipdb; ipdb.set_trace()