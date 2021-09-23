"""Linear Regression Baseline for Modality Prediction task

Run ElasticNet and do SVD decomposition for dimension reduction."""
import argparse
import logging
import pickle
import sys
from pathlib import Path
from scipy.sparse import csr_matrix

import anndata as ad
import yaml  # type: ignore

sys.path.append(str(Path.cwd()))

from lab_scripts.data import dataloader
from lab_scripts.metrics import mp
from lab_scripts.models.baselines import linear_regression
from lab_scripts.utils import utils

log = logging.getLogger("linear_regression")
base_config_path = "configs/mp/baseline_linreg/"
base_checkpoint_path = "checkpoints/mp/baseline_linreg/"


def _get_svd_default_checkpoint(mod: str, n_components: int):
    filename = f"svd_{mod}{n_components}.ckpt"
    return base_checkpoint_path + filename


def get_svd(mod_config: dict, mod: str, resources_dir: str = ""):
    """Loads SVD object"""
    reduce_config = mod_config["reduce_dim"]

    n_components = reduce_config["n_components"]
    svd_checkpoint_path = reduce_config.get(
        "checkpoint_path",
        _get_svd_default_checkpoint(mod, n_components),
    )
    svd_checkpoint_path = resources_dir + svd_checkpoint_path
    with open(svd_checkpoint_path, "rb") as f:
        svd = pickle.load(f)
    return svd


def predict_submission(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
    resources_dir: str,
) -> ad.AnnData:
    log.info("Start linear regression prediction...")

    # Choose model checkpoint depending on input modalities
    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    log.info("Data is loaded")

    # Open config file
    config_path = resources_dir + base_config_path + task_type + ".yaml"
    log.info(f"Config path: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Preprocess data
    mod1_config = config.get("mod1", {})
    if "reduce_dim" in mod1_config:
        svd_mod1 = get_svd(mod1_config, mod1, resources_dir)
        test_mod1_X = linear_regression.apply_svd(input_test_mod1, svd_mod1)
    else:
        test_mod1_X = utils.convert_to_dense(input_test_mod1).X
    log.info("Data is preprocessed")

    checkpoint_path = resources_dir + base_checkpoint_path + task_type + ".ckpt"
    log.info(f"Checkpoint path: {checkpoint_path}")
    logging.info("Loading model...")
    with open(checkpoint_path, "rb") as f:  # type: ignore
        regressor = pickle.load(f)  # type: ignore

    predictions = linear_regression.predict(
        test_mod1_X,
        regressor,
    )

    mod2_config = config.get("mod2", {})
    if "reduce_dim" in mod2_config:
        svd_mod2 = get_svd(mod2_config, mod2, resources_dir)
        predictions = linear_regression.invert_svd(predictions, svd_mod2)

    predictions = csr_matrix(predictions)

    result = ad.AnnData(
        X=predictions,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,
        uns={"dataset_id": input_train_mod1.uns["dataset_id"]},
    )
    return result


def evaluate(config: dict):
    """Calculates train and test metrics."""
    # Load data
    files = dataloader.load_data(config["data"])
    train_mod1 = files["train_mod1"]
    train_mod2 = files["train_mod2"]
    test_mod1 = files["test_mod1"]
    test_mod2 = files["test_mod2"]

    mod1 = utils.get_mod(train_mod1)
    mod2 = utils.get_mod(train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    log.info("Data is loaded")

    # Preprocess data
    mod1_config = config.get("mod1", {})
    if "reduce_dim" in mod1_config:
        svd_mod1 = get_svd(mod1_config, mod1)
        train_mod1_X = linear_regression.apply_svd(train_mod1, svd_mod1)
        test_mod1_X = linear_regression.apply_svd(test_mod1, svd_mod1)
    else:
        train_mod1_X = utils.convert_to_dense(train_mod1).X
        test_mod1_X = utils.convert_to_dense(test_mod1).X
    log.info("Data is preprocessed")

    # Load model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )
    with open(checkpoint_path, "rb") as f:
        regressor = pickle.load(f)
        log.info(f"Model is loaded from checkpoint {checkpoint_path}")

    # Evaluate on train
    train_predictions = linear_regression.predict(train_mod1_X, regressor)
    mod2_config = config.get("mod2", {})
    if "reduce_dim" in mod2_config:
        svd_mod2 = get_svd(mod2_config, mod2)
        train_predictions = linear_regression.invert_svd(train_predictions, svd_mod2)
    print(f"Train target metric: {mp.calculate_target(train_predictions, train_mod2)}")

    # Evaluate on test
    test_predictions = linear_regression.predict(test_mod1_X, regressor)
    if "reduce_dim" in mod2_config:
        svd_mod2 = get_svd(mod2_config, mod2)
        test_predictions = linear_regression.invert_svd(test_predictions, svd_mod2)
    print(f"Test target metric: {mp.calculate_target(test_predictions, test_mod2)}")


def train_svd(dataset: ad.AnnData, mod_config: dict):
    """Trains dimension reduction"""
    reduce_config = mod_config["reduce_dim"]

    mod_type = utils.get_mod(dataset)
    n_components = reduce_config["n_components"]
    svd_checkpoint_path = reduce_config.get(
        "checkpoint_path",
        _get_svd_default_checkpoint(mod_type, n_components),
    )
    log.info(f"Training SVD for {mod_type} into {n_components} dimensions...")
    svd = linear_regression.train_svd(dataset.X, n_components)
    with open(svd_checkpoint_path, "wb") as f:
        pickle.dump(svd, f)
        log.info(f"SVD for {mod_type} is saved at {svd_checkpoint_path}")


def train(
    config: dict,
):
    """Trains model and preprocessing pipeline"""
    # Load data
    files = dataloader.load_data(config["data"])
    train_mod1 = files["train_mod1"]
    mod1 = utils.get_mod(train_mod1)
    train_mod2 = files["train_mod2"]
    mod2 = utils.get_mod(train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    log.info("Data is loaded")

    # Preprocess data (and train dimension reduction if needed)
    mod1_config = config.get("mod1", {})
    if "reduce_dim" in mod1_config:
        train_svd(train_mod1, mod1_config)
        svd_mod1 = get_svd(mod1_config, mod1)
        X_mod1 = linear_regression.apply_svd(train_mod1, svd_mod1)
    else:
        X_mod1 = utils.convert_to_dense(train_mod1).X

    mod2_config = config.get("mod2", {})
    if "reduce_dim" in mod2_config:
        train_svd(train_mod2, mod2_config)
        svd_mod2 = get_svd(mod2_config, mod2)
        X_mod2 = linear_regression.apply_svd(train_mod2, svd_mod2)
    else:
        X_mod2 = utils.convert_to_dense(train_mod2).X
    log.info("Data is preprocessed")

    regressor = linear_regression.train_regressor(X_mod1, X_mod2, config)

    # Save model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )
    with open(checkpoint_path, "wb") as f:
        pickle.dump(regressor, f)
        log.info(f"Model is saved at {checkpoint_path}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Linear Regression Baseline for MP task"
    )
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=argparse.FileType("r"))

    parser_evaluate = subparsers.add_parser("evaluate")
    parser_evaluate.add_argument("config", type=argparse.FileType("r"))

    return parser


def cli():
    parser = get_parser()
    args = parser.parse_args()
    config = yaml.safe_load(args.config)
    if args.action == "train":
        train(config)
    elif args.action == "evaluate":
        evaluate(config)
    else:
        print("Enter command [train, evaluate]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
