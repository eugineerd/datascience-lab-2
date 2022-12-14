from sklearn.compose import ColumnTransformer
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_feature_transformer(X: pd.DataFrame, y: pd.DataFrame) -> ColumnTransformer:
    ord_mapping = [
        {
            "col": "LotShape",
            "mapping": {
                "Reg": 3,
                "IR1": 2,
                "IR2": 1,
                "IR3": 0,
            },
        },
        {
            "col": "Utilities",
            "mapping": {
                "AllPub": 3,
                "NoSewr": 2,
                "NoSeWa": 1,
                "ELO": 0,
            },
        },
        {
            "col": "LandSlope",
            "mapping": {
                "Gtl": 0,
                "Mod": 1,
                "Sev": 2,
            },
        },
        {
            "col": "ExterQual",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
            },
        },
        {
            "col": "ExterCond",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
            },
        },
        {
            "col": "BsmtQual",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
                "NA": -1,
            },
        },
        {
            "col": "BsmtCond",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
                "NA": -1,
            },
        },
        {
            "col": "BsmtExposure",
            "mapping": {
                "Gd": 3,
                "Av": 2,
                "Mn": 1,
                "No": 0,
                "NA": -1,
            },
        },
        {
            "col": "BsmtFinType1",
            "mapping": {
                "GLQ": 5,
                "ALQ": 4,
                "BLQ": 3,
                "Rec": 2,
                "LwQ": 1,
                "Unf": 0,
                "NA": -1,
            },
        },
        {
            "col": "BsmtFinType2",
            "mapping": {
                "GLQ": 5,
                "ALQ": 4,
                "BLQ": 3,
                "Rec": 2,
                "LwQ": 1,
                "Unf": 0,
                "NA": -1,
            },
        },
        {
            "col": "HeatingQC",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
            },
        },
        {"col": "CentralAir", "mapping": {"Y": 1, "N": 0}},
        {
            "col": "Electrical",
            "mapping": {
                "SBrkr": 4,
                "FuseA": 3,
                "FuseF": 2,
                "FuseP": 1,
                "Mix": -1,
            },
        },
        {
            "col": "KitchenQual",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
            },
        },
        {
            "col": "Functional",
            "mapping": {
                "Typ": 7,
                "Min1": 6,
                "Min2": 5,
                "Mod": 4,
                "Maj1": 3,
                "Maj2": 2,
                "Sev": 1,
                "Sal": 0,
            },
        },
        {
            "col": "FireplaceQu",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
                "NA": -1,
            },
        },
        {
            "col": "GarageFinish",
            "mapping": {
                "Fin": 2,
                "RFn": 1,
                "Unf": 0,
                "NA": -1,
            },
        },
        {
            "col": "GarageQual",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
                "NA": -1,
            },
        },
        {
            "col": "GarageCond",
            "mapping": {
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "Po": 0,
                "NA": -1,
            },
        },
        {
            "col": "PavedDrive",
            "mapping": {
                "Y": 2,
                "P": 1,
                "N": 0,
            },
        },
        {
            "col": "Fence",
            "mapping": {
                "GdPrv": 3,
                "MnPrv": 2,
                "GdWo": 1,
                "MnWw": 0,
                "NA": -1,
            },
        },
    ]
    ord_features = set(x["col"] for x in ord_mapping)
    cat_features = set(X.columns[X.dtypes == "object"]).difference(ord_features)
    real_features = set(X.columns).difference((ord_features.union(cat_features)))

    col_tr = ColumnTransformer(
        transformers=[
            ("real", StandardScaler(), list(real_features)),
            (
                "ord",
                OrdinalEncoder(handle_unknown="value", handle_missing="error", mapping=ord_mapping),  # type: ignore
                list(ord_features),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="value", handle_missing="error"),
                list(cat_features),
            ),
        ],
        remainder="drop",
    )
    col_tr.fit(X, y)
    return col_tr
