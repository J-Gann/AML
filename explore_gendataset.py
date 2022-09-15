import logging

from experiment import Experiment, run_experiment
from preprocess import AmexPreprocessor
from methods import PersistDatasetMethod
from data import Dataset

import torch

log = logging.getLogger(__name__)

"""Use best parameters from hyperparameter optimization to generate imputed and preprocessed dataset."""
def gen_dataset():
    ds = Dataset("data/train_data_reduced.csv", "data/train_labels_reduced.csv")
    exp = Experiment(
        "exp",
        ds,
        AmexPreprocessor(ds, config={
            "float_impute_strategy": "mean",
            "float_scale": False, # Whether to use standardscaler
            "float_denoise": True, # Whether to use denoising trick
            "correlation_drop_percentage": 0.94,  # Delete Features with higher than this correlation coefficient
            "impute_per_customer": True,  # Whether to impute per customer
        }),
        PersistDatasetMethod,
        method_config={
            "target_dir": "output",
        },
        method_fit_kwargs={},
        kfold_ensemble_num=1, # Only train 1 classifier for now, quick dev
        kfold_random_state=15,
        use_cache=False,
    )
    return run_experiment(exp)

FORMAT = "%(asctime)s %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logging.root.setLevel(logging.DEBUG)

if __name__ == "__main__":
    gen_dataset()