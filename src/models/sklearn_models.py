import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from src.models.common import default_train_split
import src.cfg as cfg


def make_extra_trees_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    feature_tr: ColumnTransformer,
) -> Pipeline:
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

    param_grid = {
        # "reg__n_estimators": (80, 100, 120, 300),
        # "reg__max_features": ("auto", "log2", "sqrt", 1.0),
        "reg__bootstrap": (True, False),
        "reg__min_samples_leaf": (1, 2),
    }
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, refit=True)
    gs.fit(X, y)

    return gs.best_estimator_  # type: ignore
