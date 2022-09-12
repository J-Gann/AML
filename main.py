import logging

from experiment import Experiment, run_experiment
from preprocess import AmexPreprocessor
from methods import CatboostMethod, StubMethod
from data import Dataset

import optuna
from optuna.integration import CatBoostPruningCallback
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

import torch

def objective(trial):
    ds = Dataset("data/train_data_reduced.csv", "data/train_labels_reduced.csv")
    impute_strategy = trial.suggest_categorical("impute_strategy", ["mean", "median", "most_frequent"])
    iterations = trial.suggest_int("iterations", 20, 150)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    depth = trial.suggest_int("depth", 3, 11)
    boosting_type = trial.suggest_categorical("boosting_type", ["Ordered", "Plain"])
    corr_drop_percentage = trial.suggest_float("correlation_drop_percentage", 0.01, 0.99)
    #objective = trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
    exp = Experiment(
        "exp",
        ds,
        AmexPreprocessor(ds, config={
            "float_imputer": "simple",
            "float_simple_imputer_strategy": impute_strategy,
            "float_simple_imputer_fill_value": 0, # Only used if strategy=constant
            "float_scale": True, # Whether to use standardscaler
            "correlation_drop_percentage": corr_drop_percentage,
        }),
        CatboostMethod,
        method_config={
            "iterations": iterations,
            "l2_leaf_reg": 7,
            "depth": depth,
            "learning_rate": learning_rate,
            "verbose": True,
            "task_type": "GPU" if torch.cuda.is_available() else None,
            "eval_metric": "Accuracy",
            "boosting_type": boosting_type,
            #"objective": objective
        },
        method_fit_kwargs={
            "callbacks": [CatBoostPruningCallback(trial, "Accuracy")]
        },
        kfold_ensemble_num=1, # Only train 1 classifier for now, quick dev
        use_cache=False,
    )
    return run_experiment(exp)

FORMAT = "%(asctime)s %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logging.root.setLevel(logging.DEBUG)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", storage='sqlite:///optuna.db', load_if_exists=True, study_name="amex2")
    study.optimize(objective, n_trials=2)
    best_params = study.best_params
    print(best_params)

    logging.root.setLevel(logging.WARNING)

    fig = plot_optimization_history(study)
    fig.write_html("plot_optimization_history.html")

    fig = plot_parallel_coordinate(study)
    fig.write_html("plot_parallel_coordinate.html")

    fig = plot_contour(study)
    fig.write_html("plot_contour.html")

    fig = plot_param_importances(study)
    fig.write_html("plot_param_importances.html")

    fig = plot_edf(study)
    fig.write_html("plot_edf.html")