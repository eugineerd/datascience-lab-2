import catboost
import pandas as pd
from src.models.common import default_train_split, load_train_dataset
from typing import Dict, List
import src.cfg as cfg


def make_catboost_regressor(
    X: pd.DataFrame, y: pd.DataFrame
) -> catboost.CatBoostRegressor:
    X_train, X_val, y_train, y_val = default_train_split(X, y)

    cat_features = list(X.columns[X.dtypes == "object"])
    cb = catboost.CatBoostRegressor(
        random_state=cfg.RS,
        iterations=3000,
        max_depth=4,
        verbose=0,
        cat_features=cat_features,
    )
    cb.fit(
        X=X_train,
        y=y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    return cb
