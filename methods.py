import numpy as np
import logging
import pathlib
import joblib

from catboost import CatBoostClassifier, Pool, FeaturesData

log = logging.getLogger(__name__)


class Method:
    cache_dir = "cache"

    def __init__(self, config=None) -> None:
        pass

    def train(self, X_train_f, X_val_f, X_train_c, X_val_c, y_train, y_val, fit_kwargs={}):
        raise NotImplementedError()

    def train_cached(self, parent_hash, *train_args, use_cache=True, fit_kwargs={}):
        if not use_cache:
            return self.train(*train_args)
        train_hash = joblib.hash((parent_hash, train_args))
        cache_loc, _ = self.get_cache_loc(train_hash)
        if cache_loc.exists():
            log.debug(f"Loading model from {cache_loc}")
            return self.load(cache_loc)
        log.debug("Training Model")
        self.train(*train_args, fit_kwargs=fit_kwargs)
        cache_loc.parent.mkdir(parents=True, exist_ok=True)
        log.debug(f"Saving Model parameters to {cache_loc}")
        self.save(cache_loc)

    def load(self, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def get_hash(self, parent_hash):
        return parent_hash

    def get_cache_loc(self, parent_hash):
        hash = self.get_hash(parent_hash)
        return (
            pathlib.Path(f"{self.cache_dir}/{self.__class__.__name__}_{hash}.joblib"),
            hash,
        )

    def eval(self, X_floats, X_cat):
        raise NotImplementedError()


class StubMethod(Method):
    def __init__(self) -> None:
        super().__init__()

    def train(self, X_train_f, X_val_f, X_train_c, X_val_c, y_train, y_val):
        log.debug(f"Training using {self.__class__.__name__}")
        pass

    def eval(self, X_floats, X_cat):
        return np.zeros(len(X_floats))


class CatboostMethod(Method):
    def __init__(
        self,
        config={
            "iterations": 50,
            "l2_leaf_reg": 7,
            "depth": 7,
            "learning_rate": 0.1,
            "verbose": True,
        },
    ) -> None:
        super().__init__()
        self.config = config
        self.model = CatBoostClassifier(**config)

    def get_hash(self, parent_hash):
        return joblib.hash((parent_hash, self.config))

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)

    def train(self, X_train_f, X_val_f, X_train_c, X_val_c, y_train, y_val, fit_kwargs={}):
        log.debug(f"Training using {self.__class__.__name__}")
        features_data = FeaturesData(
            num_feature_data=X_train_f.astype(np.float32),
            cat_feature_data=X_train_c.astype(str).astype(object),
        )
        pool = Pool(features_data, y_train.tolist())
        features_data_val = FeaturesData(
            num_feature_data=X_val_f.astype(np.float32),
            cat_feature_data=X_val_c.astype(str).astype(object),
        )
        pool_val = Pool(features_data_val, y_val.tolist())
        self.model.fit(
            pool, 
            eval_set=pool_val, 
            use_best_model=False,
            **fit_kwargs)

    def eval(self, X_floats, X_cat):
        X = FeaturesData(
            num_feature_data=X_floats.astype(np.float32),
            cat_feature_data=X_cat.astype(str).astype(object),
        )
        return self.model.predict_proba(X)[:, 1]
