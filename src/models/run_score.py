import click
import logging
import os
import pandas as pd
import pickle
import json
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from .common import get_model_metrics, load_dataset, default_train_split


@click.command()
@click.argument("predictions_filepath", type=click.Path(exists=True))
@click.argument("dataset_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--split", default=True)
def main(
    predictions_filepath: str,
    dataset_filepath: str,
    output_filepath: str,
    split: bool,
):
    logger = logging.getLogger(__name__)
    logger.info("Scoring models...")
    X, y = load_dataset(dataset_filepath)
    if split:
        _, _, _, y_test = default_train_split(X, y)
    else:
        y_test = y

    if not os.path.exists(output_filepath):
        logger.info(f"'{output_filepath}' not found, creating")
        os.mkdir(output_filepath)

    for predictions_path in Path(predictions_filepath).glob("*.json"):
        logging.info(f"Testing {predictions_path.name}")
        with open(predictions_path, "r") as f:
            y_pred = json.load(f)
        metrics = get_model_metrics(y_test, y_pred)
        with open(Path(output_filepath).joinpath(predictions_path.name), "w") as f:
            json.dump(metrics, f)

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
