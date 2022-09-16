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

log = logging.getLogger(__name__)

def objective(trial):
    #ds = Dataset("data/train_data_tinytest.csv", "data/train_labels_tinytest.csv")
    ds = Dataset("data/train_data_reduced.csv", "data/train_labels_reduced.csv")
    impute_strategy = trial.suggest_categorical("impute_strategy", ["mean", "median", "most_frequent"])
    iterations = trial.suggest_int("iterations", 20, 150)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    depth = trial.suggest_int("depth", 3, 11)
    boosting_type = trial.suggest_categorical("boosting_type", ["Ordered", "Plain"])
    corr_drop_percentage = trial.suggest_float("correlation_drop_percentage", 0.80, 0.99)
    float_denoise = trial.suggest_int("denoise", 0, 1)
    float_denoise_bool = bool(float_denoise)
    impute_per_customer = trial.suggest_int("impute_per_customer", 0, 1)
    impute_per_customer_bool = bool(impute_per_customer)
    exp = Experiment(
        "exp",
        ds,
        AmexPreprocessor(ds, config={
            "float_impute_strategy": impute_strategy,
            "float_scale": False, # Whether to use standardscaler
            "float_denoise": float_denoise_bool, # Whether to use denoising trick
            "correlation_drop_percentage": corr_drop_percentage,
            "correlation_drop_percentage": 0.95,  # Delete Features with higher than this correlation coefficient
            "impute_per_customer": impute_per_customer_bool,  # Whether to impute per customer
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
            "random_seed": 42 # Make more reproducible
            #"objective": objective
        },
        method_fit_kwargs={
            "callbacks": [CatBoostPruningCallback(trial, "Accuracy")]
        },
        kfold_ensemble_num=1, # Only train 1 classifier for now, quick dev
        kfold_random_state=15,
        use_cache=False,
    )
    return run_experiment(exp)

FORMAT = "%(asctime)s %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logging.root.setLevel(logging.DEBUG)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", storage='sqlite:///optuna.db', load_if_exists=True, study_name="amex2")
    study.optimize(objective, n_trials=300)
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