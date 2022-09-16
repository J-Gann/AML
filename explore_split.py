import logging

from experiment import Experiment, run_experiment
from preprocess import AmexPreprocessor
from methods import PersistDatasetMethod
from data import Dataset

import torch

log = logging.getLogger(__name__)

def gen_dataset():
    ds = Dataset("data/train_data_reduced.csv", "data/train_labels_reduced.csv")
    ds.load()
    ds.df_train['customer_ID_numeric'] = ds.df_train.customer_ID.astype('category')
    ds.df_train['customer_ID_numeric'] = ds.df_train.customer_ID_numeric.cat.codes
    ds.df_train_labels['customer_ID_numeric'] = ds.df_train_labels.customer_ID.astype('category')
    ds.df_train_labels['customer_ID_numeric'] = ds.df_train_labels.customer_ID_numeric.cat.codes
    num_splits = 24
    total_length = ds.df_train.customer_ID_numeric.max()
    split_width = total_length // num_splits
    log.info(f"Using {num_splits} splits over {total_length} customers.")
    split_points = []
    for i in range(num_splits):
        if i == num_splits-1:
            split_points.append((i*split_width, total_length+1))
        else:
            split_points.append((i*split_width, (i+1)*split_width))

    print(split_points)
    for idx, split in enumerate(split_points):
        print(f'Dumping {idx}')
        df_train_split = ds.df_train[(ds.df_train['customer_ID_numeric'] >= split[0]) & (ds.df_train['customer_ID_numeric'] < split[1])]
        df_train_split = df_train_split.drop('customer_ID_numeric', axis=1)
        df_train_split = df_train_split.drop('Unnamed: 0', axis=1)
        df_train_labels_split = ds.df_train_labels[(ds.df_train_labels['customer_ID_numeric'] >= split[0]) & (ds.df_train_labels['customer_ID_numeric'] < split[1])]
        df_train_labels_split = df_train_labels_split.drop('customer_ID_numeric', axis=1)
        df_train_labels_split = df_train_labels_split.drop('Unnamed: 0', axis=1)
        df_train_split.to_csv(f"output/df_train_split.{idx}.csv")
        df_train_labels_split.to_csv(f"output/df_train_labels_split.{idx}.csv")

FORMAT = "%(asctime)s %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logging.root.setLevel(logging.DEBUG)

if __name__ == "__main__":
    gen_dataset()