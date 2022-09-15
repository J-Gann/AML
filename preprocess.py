import logging
import joblib
import pathlib
import numpy as np
import pandas as pd

from data import Dataset
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
tqdm.pandas()

log = logging.getLogger(__name__)


def remove_correlated_features(df, drop_percentage):
    log.debug("Computing correlation matrix")
    corr = df.corr()
    corr_relevant = corr.where(np.triu(np.ones(corr.shape), k=0) == 1)[corr != 1][
        corr >= drop_percentage
    ]

    unstacked = corr_relevant.unstack()  # MxM Matrix to hierarchical index
    cols_to_delete = unstacked[~unstacked.isnull()].reset_index()["level_1"].unique()
    df = df.drop(cols_to_delete, axis=1)
    log.debug(f"Removing {len(cols_to_delete)} highly correlated features.")

    return df, cols_to_delete

def apply_strategy(pandas_thing, strategy):
    if strategy == "mean":
        return pandas_thing.mean()
    elif strategy == "median":
        return pandas_thing.median()
    elif strategy == "most_frequent":
        return pandas_thing.mode()
    else:
        raise RuntimeError(f"No such strategy: {strategy}")

def impute_per_group(groupby_df, strategy):
    fill_values = apply_strategy(groupby_df, strategy).to_dict()
    return groupby_df.fillna(fill_values)

def amex_impute(X, *, cat_vars, strategy, per_customer=True):
    log.info("Starting Imputation routine")
    # Fill NaN with strategy, per customer
    X_float_cols = [
        col
        for col in X.columns
        if col not in cat_vars and col not in ("Unnamed: 0", "customer_ID", "S_2")
    ]
    X_cat_cols = [
        col
        for col in X.columns
        if col in cat_vars and col not in ("Unnamed: 0", "customer_ID", "S_2")
    ]
    if per_customer:
        log.debug("Imputing per customer")
        X[X_float_cols] =  X.groupby("customer_ID")[X_float_cols].progress_apply(lambda df: impute_per_group(df, strategy)).reset_index(drop=True)
    # At this stage, some NaN will remain - as some customer columns are only NaN.
    # In that case, fill with global aggregate value.
    # As a last resort (NOTHING is non-NaN) - fills with zeros.
    X[X_float_cols] = X[X_float_cols].transform(
        lambda x: x.fillna(apply_strategy(x, strategy)).fillna(0)
    )
    # No NaN floats after this point.

    if per_customer:
        # Next, fill categorical variables - again on a per-customer basis
        X[X_cat_cols] = X.groupby("customer_ID")[X_cat_cols].apply(lambda df: impute_per_group(df, "most_frequent")).reset_index(drop=True)
    # Any NaNs that remain are there because they are NaN on every customer row.
    # Convert to categorical and then replace with the category code - will also fill remaining NaN with -1
    X[X_cat_cols] = X[X_cat_cols].astype("category")
    X[X_cat_cols] = X[X_cat_cols].transform(lambda s: s.cat.codes)

    log.debug("Imputation done.")
    if not X.isnull().any().any():
        import ipdb;ipdb.set_trace()
    #TODO disable
    #assert not X.isnull().any().any(), "After Imputation, null values remain."
    return X

def _add_missing_rows(df):
    if len(df) != 13:
        proto_series = df.iloc[0].copy()
        proto_series.values[:] = np.nan
        proto_series.customer_ID = df.iloc[0].customer_ID
        proto_series.S_2 = "filled"
        while len(df) != 13:
            df = df.append(proto_series.copy())
        return df
    return df

def amex_fill_missing_records(X):
    log.debug("Filling missing records")
    X = X.groupby('customer_ID').progress_apply(_add_missing_rows).reset_index(drop=True)
    return X


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
        return (
            pathlib.Path(f"{self.cache_dir}/{self.__class__.__name__}_{hash}.joblib"),
            hash,
        )

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
            "float_impute_strategy": "mean", # mean, median, most_frequent.
            "float_scale": True,  # Whether to use standardscaler
            "correlation_drop_percentage": 0.95,  # Delete Features with higher than this correlation coefficient
            "impute_per_customer": True,  # Whether to impute per customer
            "float_denoise": True, # Whether to apply denoising
        },
        cache_dir="cache",
        # more config here for bayes opt
    ):
        super().__init__(dataset, cache_dir)
        self.float_scaler = StandardScaler()
        self.config = config

    def preprocess(self, X, y=None):
        # TODO is it smart to attempt to remove correlated cat variables?
        X, _ = remove_correlated_features(X, self.config["correlation_drop_percentage"])
        X = amex_fill_missing_records(X) # TODO should we make it a hyperparam?
        X = amex_impute(
            X,
            cat_vars=self.dataset.cat_vars,
            strategy=self.config["float_impute_strategy"],
            per_customer=self.config["impute_per_customer"],
        )
        if self.config["float_denoise"]:
            log.debug("Applying denoising trick")
            # Denoising
            X_float_cols = [
                col
                for col in X.columns
                if col not in self.dataset.cat_vars and col not in ("Unnamed: 0", "customer_ID", "S_2")
            ]
            X[X_float_cols] = np.floor(X[X_float_cols]*100)/100

        # Stack Dataframe so that each customer is represented by one row, make columns wider
        X_reset = X.reset_index()
        X_reset["group_index"] = X.groupby("customer_ID").cumcount()

        X_reindex = X_reset.set_index(["customer_ID", "group_index"])
        X_unstack = X_reindex.unstack()
        X_unstack = X_unstack.drop("index", level=0, axis=1)
        X_unstack = X_unstack.drop("S_2", level=0, axis=1)
        X_unstack.columns = [
            " ".join(map(str, col)).strip() for col in X_unstack.columns.values
        ]

        # Convert categorical features in wide dataframe to integers
        # cat_vars_wide = list(
        #     filter(
        #         lambda x: any([y in x for y in self.dataset.cat_vars]),
        #         X_unstack.columns,
        #     )
        # )
        # cat_dtypes = X_unstack.dtypes[cat_vars_wide]
        # X_unstack = X_unstack.astype(
        #     {k: "Int32" for k in cat_dtypes[cat_dtypes == np.float64].keys()}
        # )
        # X_unstack[cat_vars_wide] = X_unstack[cat_vars_wide].replace(
        #     np.nan, -1
        # )  # Put NaN in own category, represented by -1 value
        # cat_dtypes = X_unstack.dtypes[cat_vars_wide]
        # X_unstack = X_unstack.astype(
        #     {k: np.int32 for k in cat_dtypes[cat_dtypes == "Int32"].keys()}
        # )

        # # Additionally convert cateorical features that are strings to integers, so we have only numbers (needed for tabnet only)
        # string_cols = X_unstack.select_dtypes(include="object").columns
        # X_unstack = X_unstack.astype({k: "category" for k in string_cols})
        # cols = X_unstack.select_dtypes(include="category").columns
        # for col in cols:
        #     X_unstack[col] = X_unstack[col].cat.codes

        # Remove highly correlated features
        # TODO maybe. Or rather perhaps condense time series somehow
        # X_unstack, deleted_cols = remove_correlated_features(X_unstack, self.config["correlation_drop_percentage"])

        # Convert to numpy arrays and return
        cat_vars_wide = list(
            filter(
                lambda x: any([y in x for y in self.dataset.cat_vars]),
                X_unstack.columns,
            )
        )
        float_vars_wide = [x for x in X_unstack.columns if x not in cat_vars_wide]
        X_floats = X_unstack[set(float_vars_wide)].to_numpy()
        # Scale
        if self.config["float_scale"]:
            X_floats = self.float_scaler.fit_transform(X_floats)
        X_cat = X_unstack[set(cat_vars_wide)].to_numpy()

        y_np = y.target.to_numpy() if y is not None else None
        return X_floats, X_cat, y_np
