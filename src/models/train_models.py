import logging
import click
import pickle
import os
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from .catboost_models import make_catboost_regressor
from .common import load_train_dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)
    logger.info("Training models")

    logging.info("Loading training dataset")
    X, y = load_train_dataset(input_filepath)
    logging.info("Training CatBoostRegressor")
    cb = make_catboost_regressor(X, y)
    logging.info("Saving CatBoostRegressor")
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    with open(os.path.join(output_filepath, "catboost.pkl"), "wb") as f:
        pickle.dump(cb, f)
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
