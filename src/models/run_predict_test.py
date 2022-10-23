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
from .common import get_model_metrics, load_train_dataset, default_train_split


@click.command()
@click.argument("models_dir", type=click.Path(exists=True))
@click.argument("train_dataset_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(models_dir: str, train_dataset_filepath: str, output_filepath: str):
    logger = logging.getLogger(__name__)
    logger.info("Predicting test models...")

    if not os.path.exists(output_filepath):
        logger.info(f"'{output_filepath}' not found, creating")
        os.mkdir(output_filepath)

    logging.info("Loading dataset")
    X, y = load_train_dataset(train_dataset_filepath)
    _, X_test, _, _ = default_train_split(X, y)

    for model_path in Path(models_dir).glob("*.pkl"):
        model_name = model_path.name.rsplit(".", 1)[0]
        logging.info(f"Testing {model_name}")
        with open(model_path, "rb") as f:
            reg: Pipeline = pickle.load(f)
        y_pred = list(reg.predict(X_test))
        with open(Path(output_filepath).joinpath(f"{model_name}.json"), "w") as f:
            json.dump(y_pred, f)

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
