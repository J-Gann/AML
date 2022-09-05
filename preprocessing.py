import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import math

dataset = pd.read_csv('test_data/test_data.csv')

labels = pd.read_csv('train_data/train_labels.csv')

timeint = pd.to_datetime(dataset['S_2']).astype(int)

timeint = (timeint-timeint.min())/(timeint.max()-timeint.min())

dataset_ = dataset.drop(['D_63', 'D_64', 'S_2'], axis=1)

dummies = pd.get_dummies(dataset[['D_63', 'D_64']])

dataset_ = pd.concat([dataset_, timeint, dummies], axis = 1)

dataset_ = dataset_.fillna(0)

dataset_ = dataset_.groupby(["customer_ID"], as_index=False).agg(list)

def fillSeries(row):
    new_row = []
    size = len(row[1])
    if size < 13:
        for idx in range(len(row)):
            if idx == 0: new_row.append(row[idx])
            else: new_row.append([row[idx][0] for cnt in range(13 - size)] + row[idx])
    else:
        new_row = row
    return new_row


dataset_ = dataset_.apply(fillSeries, axis=1)

dataset_ = dataset_.set_index('customer_ID').join(labels.set_index('customer_ID'))

dataset_ = dataset_.reset_index(level=0)

dataset_.to_pickle("./transformed_test_dataset")