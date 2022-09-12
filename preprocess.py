import logging
import joblib
import pathlib
import numpy as np
import pandas as pd

from data import Dataset
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

def remove_correlated_features(df, drop_percentage):
    log.debug("Computing correlation matrix")
    corr = df.corr()
    corr_relevant = corr.where(np.triu(np.ones(corr.shape), k=0)==1)[corr != 1][corr >= drop_percentage]
    delete_cols = []
    for row in corr_relevant.index:
        for col in corr_relevant.index:
            if pd.notna(corr_relevant.loc[row, col]):
                log.debug(f'Rows {row}, {col} very correlated: {corr_relevant.loc[row, col]}')
                delete_cols.append(col)
    for col in set(delete_cols):
        df = df.drop(col, axis=1)
    return df, set(delete_cols)

class Preprocessor:
    def __init__(self, dataset: Dataset, cache_dir="cache") -> None:
        self.dataset = dataset
        self.cache_dir = cache_dir

    def preprocess(self, X, y=None):
        """
        Returns NxM matrix X_cat with categorical features
                NxM' matrix X_floats with floating point features
                IF y is supplied, also a Nx1 vector of labels.
        """
        raise NotImplementedError()

    def get_hash(self, X, y=None):
        """
        Returns hash unique for this instance of preprocessing.
        Hash should be sensitive to changes to:
        * X and y parameters
        * Actual preprocessing hyperparameters and configuration.
        """
        return joblib.hash((X, y))

    def get_cache_loc(self, X, y=None):
        hash = self.get_hash(X, y)
        return pathlib.Path(
            f"{self.cache_dir}/{self.__class__.__name__}_{hash}.joblib"
        ), hash

    def preprocess_cached(self, X, y=None, use_cache=True):
        if use_cache:
            cache_loc, hash = self.get_cache_loc(X, y)
            if cache_loc.exists():
                log.debug(f"Loading cache from {cache_loc}")
                return joblib.load(cache_loc), hash
        log.debug(f"Starting preprocessing with {__class__}")
        data = self.preprocess(X, y)
        log.debug("Preprocessing done")
        hash = None
        if use_cache:
            cache_loc, hash = self.get_cache_loc(X, y)
            if not cache_loc.exists():
                log.debug(f"Caching data at {cache_loc}")
                cache_loc.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(data, cache_loc)
        return data, hash


class AmexPreprocessor(Preprocessor):
    def __init__(
        self,
        dataset,
        config={
            "float_imputer": "simple",
            "float_simple_imputer_strategy": "mean",
            "float_simple_imputer_fill_value": 0, # Only used if strategy=constant
            "float_scale": True, # Whether to use standardscaler
            "correlation_drop_percentage": 0.95 # Delete Features with higher than this correlation coefficient
        },
        cache_dir="cache",
        # more config here for bayes opt
    ):
        super().__init__(dataset, cache_dir)
        self.float_imputer = SimpleImputer(
            strategy=config["float_simple_imputer_strategy"],
            fill_value=config["float_simple_imputer_fill_value"],
        )
        self.float_iterative_imputer = IterativeImputer(
            max_iter=2,
            verbose=1,
        )
        self.float_scaler = StandardScaler()
        self.config = config

    def float_impute(self, X_floats):
        if self.config["float_imputer"] == "simple":
            return self.float_imputer.fit_transform(X_floats)
        elif self.config["float_imputer"] == "iterative":
            return self.float_iterative_imputer.fit_transform(X_floats)
        else:
            raise NotImplementedError(f"No such float imputer method: {self.config['float_imputer']}")

    def preprocess(self, X, y=None):
        X, _ = remove_correlated_features(X, self.config["correlation_drop_percentage"])


        # Stack Dataframe so that each customer is represented by one row, make columns wider
        X_reset = X.reset_index()
        X_reset['group_index'] = X.groupby('customer_ID').cumcount()
        X_reindex = X_reset.set_index(['customer_ID', 'group_index'])
        X_unstack = X_reindex.unstack()
        X_unstack = X_unstack.drop('index', level=0, axis=1)
        X_unstack = X_unstack.drop('S_2', level=0, axis=1)
        X_unstack.columns = [' '.join(map(str, col)).strip() for col in X_unstack.columns.values]

        # Convert categorical features in wide dataframe to integers
        cat_vars_wide = list(filter(lambda x: any([y in x for y in self.dataset.cat_vars]), X_unstack.columns))
        float_vars_wide = [x for x in X_unstack.columns if x not in cat_vars_wide]
        cat_dtypes = X_unstack.dtypes[cat_vars_wide]
        X_unstack = X_unstack.astype({k: 'Int32' for k in cat_dtypes[cat_dtypes==np.float64].keys()})
        X_unstack[cat_vars_wide] = X_unstack[cat_vars_wide].replace(np.nan, -1) # Put NaN in own category, represented by -1 value
        cat_dtypes = X_unstack.dtypes[cat_vars_wide]
        X_unstack = X_unstack.astype({k: np.int32 for k in cat_dtypes[cat_dtypes=='Int32'].keys()})

        # Additionally convert cateorical features that are strings to integers, so we have only numbers (needed for tabnet only)
        string_cols = X_unstack.select_dtypes(include='object').columns
        X_unstack = X_unstack.astype({k: 'category' for k in string_cols})
        cols = X_unstack.select_dtypes(include='category').columns
        for col in cols:
            X_unstack[col] = X_unstack[col].cat.codes

        # Remove highly correlated features
        # TODO maybe. Or rather perhaps condense time series somehow
        # X_unstack, deleted_cols = remove_correlated_features(X_unstack, self.config["correlation_drop_percentage"])

        # Convert to numpy arrays and return
        X_floats = X_unstack[set(float_vars_wide)].to_numpy()
        X_floats = self.float_impute(X_floats)
        # Scale
        if self.config["float_scale"]:
            X_floats = self.float_scaler.fit_transform(X_floats)
        X_cat = X_unstack[set(cat_vars_wide)].to_numpy()

        
        y_np = y.target.to_numpy() if y is not None else None
        return X_floats, X_cat, y_np