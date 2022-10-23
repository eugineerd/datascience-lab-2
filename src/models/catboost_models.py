import catboost
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.models.common import default_train_split
from typing import Any, Dict, List, Optional, Tuple
import src.cfg as cfg

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def make_catboost_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_tr: Optional[ColumnTransformer] = None,
) -> Pipeline:
    X_train, X_val, y_train, y_val = default_train_split(X, y)
    if feature_tr != None:
        cat_features = None
    else:
        cat_features = list(X.columns[X.dtypes == "object"])

    cb = catboost.CatBoostRegressor(
        random_state=cfg.RS,
        iterations=3000,
        max_depth=4,
        verbose=0,
        cat_features=cat_features,
    )

    pipe_steps: List[Tuple[str, Any]] = [("reg", cb)]
    pipe = Pipeline(steps=pipe_steps)
    if feature_tr != None:
        feature_tr.fit(X)
        pipe_steps.insert(0, ("col_tr", feature_tr))
        X_val = feature_tr.transform(X_val)
        X_train = feature_tr.transform(X_train)
        cb.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    else:
        pipe.fit(
            X=X_train,
            y=y_train,
            reg__eval_set=(X_val, y_val),
            reg__use_best_model=True,
        )

    return pipe
