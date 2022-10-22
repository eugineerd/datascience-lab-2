import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import recall_score
from typing import List, Tuple
from sklearn.model_selection import train_test_split

import src.cfg as cfg


def default_train_split(
    X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return train_test_split(X, y, test_size=0.2, random_state=cfg.RS)  # type: ignore


def load_train_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open("data/processed/train.pkl", "rb") as f:
        df = pickle.load(f)
    X = df.drop(["SalePrice"], axis=1)
    y = df["SalePrice"]
    return X, y
