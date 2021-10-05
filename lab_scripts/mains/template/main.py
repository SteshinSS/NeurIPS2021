"""Template main for the Modality Prediction task. Copy it and edit."""
import argparse
import logging

import anndata as ad
import yaml  # type: ignore
from scipy.sparse import csr_matrix

from lab_scripts.data import dataloader
from lab_scripts.metrics import mp
from lab_scripts.utils import utils

# from lab_scripts.models import my_model

log = logging.getLogger("template_logger")
base_config_path = "configs/mp/template/"
base_checkpoint_path = "checkpoints/mp/template/"


def predict_submission(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
    resources_dir: str = "",
) -> ad.AnnData:
    log.info("Start template prediction...")

    # Load data
    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    log.info("Data is loaded")

    # Here are our resources
    config_path = resources_dir + base_config_path
    checkpoint_path = resources_dir + base_checkpoint_path

    # Select data type
    task_type = utils.get_task_type(mod1, mod2)
    predictions = None
    if task_type == "gex_to_atac":
        # predictions = predict_gex_to_atac(...)
        pass
    elif task_type == "atac_to_gex":
        # predictions = predict_atac_to_gex(...)
        pass
    elif task_type == "gex_to_adt":
        # predictions = predict_gex_to_adt(...)
        pass
    elif task_type == "adt_to_gex":
        # predictions = predict_adt_to_gex(...)
        pass
    else:
        raise ValueError(f"Inappropriate dataset types: {task_type}")

    # Convert matrix into csr_matrix (is needed for submission)
    predictions = csr_matrix(predictions)

    # Create AnnData object
    result = ad.AnnData(
        X=predictions,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,
        uns={"dataset_id": input_train_mod1.uns["dataset_id"]},
    )
    return result


def evaluate(config: dict):
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
    ...
    log.info("Data is preprocessed")

    # Load model
    # Example checkpoint path
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )
    ...
    log.info(f"Model is loaded from {checkpoint_path}")

    train_predictions = None
    # train_predictions = my_model.predict(...)
    print(f"Train target metric: {mp.calculate_target(train_predictions, train_mod2)}")

    test_predictions = None
    # test_predictions = my_model.predict(...)
    print(f"Test target metric: {mp.calculate_target(test_predictions, test_mod2)}")


def train(config: dict):
    # Load data
    files = dataloader.load_data(config["data"])
    train_mod1 = files["train_mod1"]
    mod1 = utils.get_mod(train_mod1)
    train_mod2 = files["train_mod2"]
    mod2 = utils.get_mod(train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    log.info("Data is loaded")

    # Preprocess data
    ...
    log.info("Data is preprocessed")

    # Train model
    # model = my_model_module.model(config)
    # model.train()

    # Save model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )
    ...
    log.info(f"Model is saved to {checkpoint_path}")


def get_parser():
    """Creates parser.

    Remove lines with adding config, if you don't need it.
    """
    parser = argparse.ArgumentParser(description="Template for MP task")
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=argparse.FileType("r"))
    parser_evaluate = subparsers.add_parser("evaluate")
    parser_evaluate.add_argument("config", type=argparse.FileType("r"))
    return parser


def cli():
    """Runs Command-Line Interface."""
    parser = get_parser()
    args = parser.parse_args()

    # Read yaml config into dict
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
