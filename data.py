import pandas as pd


class Dataset:
    def __init__(
        self,
        path_train,
        path_train_labels,
        path_test=None,
        cat_vars=[
            "B_30",
            "B_38",
            "D_114",
            "D_116",
            "D_117",
            "D_120",
            "D_126",
            "D_63",
            "D_64",
            "D_66",
            "D_68",
        ],
    ) -> None:
        self.path_train = path_train
        self.path_train_labels = path_train_labels
        self.path_test = path_test
        self.cat_vars = cat_vars
        self.df_train = None
        self.df_train_labels = None
        self.df_test = None

    def load_train(self):
        if self.df_train is None:
            self.df_train = pd.read_csv(self.path_train)
            self.df_train_labels = pd.read_csv(self.path_train_labels)

    def load_test(self):
        if self.df_test is None:
            self.df_test = pd.read_csv(self.path_test)

    def load(self):
        self.load_train()
        if self.path_test is not None:
            self.load_test()
