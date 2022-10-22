import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("Id")
    df = df.drop("PoolQC", axis=1)  # not enough samples
    df = df.dropna(subset=["Electrical"])
    df["MasVnrArea"] = df["MasVnrType"].replace(
        "None", pd.NA
    )  # Might as well treat None and NA the same

    # Not needed anymore
    # df = df.fillna(
    #     {
    #         "Alley": "NA",
    #         "MasVnrType": "None",
    #         "MasVnrArea": 0,
    #         # Basement
    #         "BsmtQual": "NA",
    #         "BsmtCond": "NA",
    #         "BsmtExposure": "NA",
    #         "BsmtFinType1": "NA",
    #         "BsmtFinType2": "NA",
    #         # Fireplace
    #         "FireplaceQu": "NA",
    #         # Garage
    #         "GarageType": "NA",
    #         "GarageYrBlt": 0,
    #         "GarageFinish": "NA",
    #         "GarageCars": 0,
    #         "GarageArea": 0,
    #         "GarageQual": "NA",
    #         "GarageCond": "NA",
    #         "Fence": "NA",
    #         "MiscFeature": "NA",
    #     }
    # )
    df = fill_rest(df)
    return df


def fill_rest(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].notna().all():
            continue
        if df[col].dtype == "object":
            # mode = df[col].mode()[0]
            # df[col] = df[col].fillna(mode)
            df[col] = df[col].fillna("NA")
        else:
            # mean = df[col].mean()
            # df[col] = df[col].fillna(mean)
            df[col] = df[col].fillna(0)
    return df
