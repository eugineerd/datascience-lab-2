import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from typing import List, Tuple
from sklearn.model_selection import train_test_split

import src.cfg as cfg


def default_train_split(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return train_test_split(X, y, test_size=0.2, random_state=cfg.RS)  # type: ignore


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(path, "rb") as f:
        df = pickle.load(f)
    X = df.drop(["SalePrice"], axis=1)
    y = df["SalePrice"]
    return X, y


def get_model_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame):
    return {
        "r2 score": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }
