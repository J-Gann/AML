from dataclasses import dataclass
import logging

import sklearn.model_selection
import numpy as np

from data import Dataset
from preprocess import Preprocessor
from methods import Method
from metric import amex_metric_np

log = logging.getLogger(__name__)


@dataclass
class Experiment:
    name: str
    dataset: Dataset
    preprocessor: Preprocessor
    method: Method
    method_config: dict 
    method_fit_kwargs: dict
    use_cache: bool = True
    evaluate_on_validation: bool = True
    kfold_ensemble_num: int = 5 # Set to something lower for quick dev.
    kfold_splits: int = 5
    kfold_random_state: int = 42


def run_experiment(exp: Experiment):
    log.info(f"Starting experiment {exp.name}")
    exp.dataset.load()

    # TODO KFold validation and ensembling here.
    (
        X_prepped_floats,
        X_prepped_cat,
        y_prepped,
    ), hash = exp.preprocessor.preprocess_cached(
        exp.dataset.df_train, exp.dataset.df_train_labels, use_cache=exp.use_cache
    )

    kf = sklearn.model_selection.KFold(
        n_splits=exp.kfold_splits, shuffle=True, random_state=exp.kfold_random_state
    )
    kf_n = 0
    all_metrics = []
    # TODO add trained methods to some sort of list for ensembled prediction later
    for train_index, val_index in kf.split(X_prepped_floats):
        X_train_f = X_prepped_floats[train_index]
        X_train_c = X_prepped_cat[train_index]
        X_val_f = X_prepped_floats[val_index]
        X_val_c = X_prepped_cat[val_index]
        y_train = y_prepped[train_index]
        y_val = y_prepped[val_index]

        method = exp.method(config=exp.method_config)
        method.train_cached(
            hash,
            X_train_f,
            X_val_f,
            X_train_c,
            X_val_c,
            y_train,
            y_val,
            use_cache=exp.use_cache,
            fit_kwargs=exp.method_fit_kwargs,
        )
        if exp.evaluate_on_validation:
            log.debug("Prediction on validation split")
            predictions = method.eval(X_val_f, X_val_c)
            metric = amex_metric_np(y_val, predictions)
            all_metrics.append(metric)
            log.debug(f"Kaggle metric on validation split: {metric}")
        kf_n += 1
        if kf_n >= exp.kfold_ensemble_num and kf_n != exp.kfold_splits:
            log.info("Stopping ensemble training early due to kfold_ensemble_num")
            break

    return np.mean(np.array(all_metrics))

    # TODO ensembled prediction

    #(
    #    X_train_f,
    #    X_val_f,
    #    X_train_c,
    #    X_val_c,
    #    y_train,
    #    y_val,
    #) = sklearn.model_selection.train_test_split(
    #    X_prepped_floats, X_prepped_cat, y_prepped
    #)

    #method = exp.method(config=exp.method_config)
    #method.train_cached(
    #    hash,
    #    X_train_f,
    #    X_val_f,
    #    X_train_c,
    #    X_val_c,
    #    y_train,
    #    y_val,
    #    use_cache=exp.use_cache,
    #)

    #if exp.evaluate_on_validation:
    #    log.debug("Prediction on validation split")
    #    predictions = method.eval(X_val_f, X_val_c)
    #    metric = amex_metric_np(y_val, predictions)
    #    log.debug(f"Kaggle metric on validation split: {metric}")

    # TODO produce final evaluations here (for submitting to kaggle), if requested
    # if exp.dataset.df_test is not None:
    #     X_test_prepped = exp.preprocessor.preprocess_cached(exp.dataset.df_test)
    #     exp.method.eval(X_test_prepped)
    # else:
    #     log.info("No test data supplied, not doing inference.")
