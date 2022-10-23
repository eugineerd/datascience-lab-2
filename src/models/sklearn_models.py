import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from src.models.common import default_train_split
import src.cfg as cfg


def make_extra_trees_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_tr: ColumnTransformer,
) -> Pipeline:
    X_train, X_val, y_train, y_val = default_train_split(X, y)
    pipe = Pipeline(
        steps=[
            ("col_tr", feature_tr),
            (
                "reg",
                ExtraTreesRegressor(
                    verbose=0, random_state=cfg.RS, n_estimators=100, n_jobs=-1
                ),
            ),
        ]
    )
    pipe.fit(X=X_train, y=y_train)

    return pipe
