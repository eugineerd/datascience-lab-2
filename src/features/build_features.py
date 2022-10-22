# NOT ACTUALLY USED!
import pandas as pd


# df.GarageYrBlt = garage_year_built_to_cat(df).astype("object")
def garage_year_built_to_cat(df: pd.DataFrame) -> pd.Series:
    start_years = map(str, range(1900, 2000, 10))
    end_years = map(str, range(1910, 2001, 10))
    garage_labels = (
        ["NA"] + list(map("-".join, zip(start_years, end_years))) + ["2000+"]
    )
    garage_bins = [-1] + list(range(1900, 2001, 10)) + [9999]
    return pd.cut(
        df.GarageYrBlt,
        bins=garage_bins,
        include_lowest=True,
        labels=garage_labels,
    )


def concat_garage_cols(df: pd.DataFrame) -> pd.DataFrame:
    garage_cols = [
        "GarageType",
        "GarageYrBlt",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
    ]
    df["GarageComb"] = df[garage_cols].agg(" ".join, axis=1)
    df = df.drop(garage_cols, axis=1)
    return df


def concat_basement_cols(df: pd.DataFrame) -> pd.DataFrame:
    basement_cols = [
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]
    df["BsmtComb"] = df[basement_cols].agg(" ".join, axis=1)
    df = df.drop(basement_cols, axis=1)
    return df
