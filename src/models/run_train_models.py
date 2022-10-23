import logging
import click
import pickle
import os
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.models.sklearn_models import make_extra_trees_pipeline
from .catboost_models import make_catboost_pipeline
from .common import load_train_dataset
from src.features.feature_transformer import get_feature_transformer


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)
    logger.info("Training models")
    if not os.path.exists(output_filepath):
        logger.info(f"'{output_filepath}' not found, creating")
        os.mkdir(output_filepath)

    logging.info("Loading training dataset")
    X, y = load_train_dataset(input_filepath)
    col_tr = get_feature_transformer(X, y)

    logging.info("Training CatBoostRegressor with native classification")
    cb_native = make_catboost_pipeline(X, y)
    logging.info("Saving CatBoostRegressor with native classification")
    with open(os.path.join(output_filepath, "catboost_native.pkl"), "wb") as f:
        pickle.dump(cb_native, f)

    logging.info("Training CatBoostRegressor with transformed features")
    cb_tr = make_catboost_pipeline(X, y, feature_tr=col_tr)
    logging.info("Saving CatBoostRegressor with transformed features")
    with open(os.path.join(output_filepath, "catboost_tr.pkl"), "wb") as f:
        pickle.dump(cb_tr, f)

    logging.info("Training ExtraTreesRegressor")
    et = make_extra_trees_pipeline(X, y, col_tr)
    logging.info("Saving ExtraTreesRegressor")
    with open(os.path.join(output_filepath, "extra_trees.pkl"), "wb") as f:
        pickle.dump(et, f)

    logging.info("Complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
