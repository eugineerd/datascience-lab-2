# -*- coding: utf-8 -*-
from typing import Optional
import click
import logging
import os
import sys
import pandas as pd
import pickle
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.data.preprocess import preprocess


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_train_filepath", type=click.Path())
@click.argument("output_validate_filepath", type=click.Path())
# @click.option("--input_answers", type=click.Path(exists=True), default=None)
def main(
    input_filepath: str,
    output_train_filepath: str,
    output_validate_filepath: str,
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logging.info("Loading dataset")
    df = pd.read_csv(input_filepath)
    logging.info("Processing dataset")
    df = preprocess(df)
    df_train, df_val = train_test_split(df, test_size=0.2)
    logging.info("Saving dataset")
    with open(output_train_filepath, "wb") as f:
        pickle.dump(df_train, f)
    with open(output_validate_filepath, "wb") as f:
        pickle.dump(df_val, f)
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
