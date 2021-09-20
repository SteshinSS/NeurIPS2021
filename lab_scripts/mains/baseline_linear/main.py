import logging
import pickle
import sys
from pathlib import Path
from typing import Dict
import argparse

import anndata as ad
import yaml  # type: ignore

sys.path.append(str(Path.cwd()))

from lab_scripts.data import dataloader
from lab_scripts.models.baselines import linear_regression
from lab_scripts.utils import utils

log = logging.getLogger("linear_regression")
default_config_path = "configs/mp/baseline_linreg.yaml"
default_checkpoint_path = "checkpoints/mp/baseline_linreg/"


def predict_submission(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
) -> ad.AnnData:
    log.info("Start linear regression prediction...")

    log.info(f"Config path: {default_config_path}")
    # Actually, we don't need it for prediction, because sklearn saves models with parameters
    config = yaml.safe_load(default_config_path)

    # Choose model checkpoint depending on input modalities
    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    checkpoint_path = default_checkpoint_path + task_type + '.ckpt'
    log.info(f"Checkpoint path: {checkpoint_path}")

    logging.info("Loading model...")
    with open(checkpoint_path, "rb") as f:
        regressor = pickle.load(f)

    return linear_regression.predict(
        input_train_mod1,
        input_train_mod2,
        input_test_mod1,
        regressor,
    )


def train(
    config: dict,
):
    files = dataloader.load_data(config["data"])

    input_train_mod1 = files["input_train_mod1"]
    mod1 = utils.get_mod(input_train_mod1)
    input_train_mod2 = files["input_train_mod2"]
    mod2 = utils.get_mod(input_train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    log.info("Data is loaded")

    regressor = linear_regression.train(input_train_mod1, input_train_mod2, config)

    checkpoint_path = config.get(
        "checkpoint_path", default_checkpoint_path + task_type + ".ckpt"
    )
    with open(checkpoint_path, "wb") as f:
        pickle.dump(regressor, f)
        log.info(f"Model is saved at {checkpoint_path}")


def evaluate(config: dict):
    files = dataloader.load_data(config["data"])
    input_train_mod1 = files['input_train_mod1']
    input_train_mod2 = files['input_train_mod2']
    input_test_mod1 = files['input_test_mod1']
    true_mod2 = files['input_test_mod2']

    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    log.info('Data is loaded')

    checkpoint_path = config.get('checkpoint_path', default_checkpoint_path + task_type + '.ckpt')
    with open(checkpoint_path, 'rb') as f:
        regressor = pickle.load(f)
        log.info(f'Model is loaded from checkpoint {checkpoint_path}')
    
    predictions = linear_regression.predict(input_train_mod1, input_train_mod2, input_test_mod1)



def predict(config: dict):
    files = dataloader.load_data(config["data"])
    input_train_mod1 = files['input_train_mod1']
    input_train_mod2 = files['input_train_mod2']
    input_test_mod1 = files['input_test_mod1']
    true_mod2 = files['input_test_mod2']

    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    log.info('Data is loaded')

    checkpoint_path = config.get('checkpoint_path', default_checkpoint_path + task_type + '.ckpt')
    with open(checkpoint_path, 'rb') as f:
        regressor = pickle.load(f)
        log.info(f'Model is loaded from checkpoint {checkpoint_path}')
    
    predictions = linear_regression.predict(input_train_mod1, input_train_mod2, input_test_mod1)
    log.info('Predictions are made.')
    return predictions


def get_parser():
    parser = argparse.ArgumentParser(
        description="Linear Regression Baseline for MP task"
    )
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=argparse.FileType("r"))

    parser_evaluate = subparsers.add_parser("evaluate")
    parser_evaluate.add_argument("config", type=argparse.FileType("r"))

    parser_predict = subparsers.add_parser("predict")
    parser_predict.add_argument("config", type=argparse.FileType("r"))

    return parser


def cli():
    parser = get_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    if args.action == "train":
        train(config)
    elif args.action == "evaluate":
        evaluate(config)
    elif args.action == "predict":
        predict(config)
    else:
        print("Enter command [train, evaluate, predict]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
