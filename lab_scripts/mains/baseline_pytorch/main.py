"""Pytorch baseline for the Modality Prediction task."""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

# Add lab_scripts folder to path. This line should be before imports
sys.path.append(str(Path.cwd()))

from lab_scripts.data import dataloader
from lab_scripts.mains import baseline_linear
from lab_scripts.metrics import mp
from lab_scripts.models.baselines import neuralnet
from lab_scripts.utils import utils

log = logging.getLogger("baseline_pytorch")
base_config_path = "configs/mp/baseline_pytorch/"
base_checkpoint_path = "checkpoints/mp/baseline_pytorch/"


def gex_to_adt(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
    resources_dir: str = "",
) -> np.ndarray:
    # Ensure this is right task type
    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    assert task_type == "gex_to_adt"

    # Load config
    config_path = resources_dir + base_config_path + task_type + ".yaml"
    log.info(f"Config path: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Preprocess data
    scaler_mod1 = load_scaler(mod1, task_type, config, resources_dir)
    test_mod1_X, _ = neuralnet.preprocess_dataset(input_test_mod1, scaler_mod1)

    # Here is tricky code: the dataloader returns pair (gex, adt). Since we need only
    # first modality for prediction, I will initialize dataloader with (gex, gex).
    dataloader = neuralnet.get_dataloader(
        test_mod1_X, test_mod1_X, batch_size=128, shuffle=False
    )
    log.info("Data is preprocessed")

    # Set configs
    config["input_features"] = input_train_mod1.X.shape[1]
    config["output_features"] = input_train_mod2.X.shape[1]
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )
    checkpoint_path = resources_dir + checkpoint_path

    # Load model
    model = neuralnet.BaselineModel.load_from_checkpoint(checkpoint_path, config=config)
    log.info(f"Model is loaded from {checkpoint_path}")
    model.eval()

    trainer = pl.Trainer(gpus=use_gpu)

    predictions = trainer.predict(model, dataloader)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()  # type: ignore
    scaler_mod2 = load_scaler(mod2, task_type, config, resources_dir)
    predictions = scaler_mod2.inverse_transform(predictions)
    log.info("Prediction is made")
    return predictions  # type: ignore


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
    task_type = utils.get_task_type(mod1, mod2)
    log.info("Data is loaded")

    # Select data type
    task_type = utils.get_task_type(mod1, mod2)
    if task_type != "gex_to_adt":
        # Run linear regression for all except for gex to adt
        return baseline_linear.predict_submission(
            input_train_mod1, input_train_mod2, input_test_mod1, resources_dir
        )

    predictions = gex_to_adt(
        input_train_mod1, input_train_mod2, input_test_mod1, resources_dir
    )

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


def load_scaler(mod: str, task_type: str, config: dict, resource_dir: str = ""):
    """Load StandardScaler from file

    Args:
        mod (str): modality
        task_type (str):
        config (dict):

    Returns:
        StandardScaler:
    """
    mod_config = config.get(mod, {})
    checkpoint_path = mod_config.get(
        "checkpoint_path",
        base_checkpoint_path + mod + "_scaler_" + task_type + ".ckpt",
    )
    checkpoint_path = resource_dir + checkpoint_path
    with open(checkpoint_path, "rb") as f:
        scaler = pickle.load(f)
        log.info(f"{mod} StandardScaler is loaded from {checkpoint_path}")
        return scaler


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
    assert task_type == "gex_to_adt"
    log.info("Data is loaded")

    # Preprocess data
    scaler_mod1 = load_scaler(mod1, task_type, config)
    train_mod1_X, scaler_mod1 = neuralnet.preprocess_dataset(train_mod1, scaler_mod1)
    scaler_mod2 = load_scaler(mod2, task_type, config)
    train_mod2_X, scaler_mod2 = neuralnet.preprocess_dataset(train_mod2, scaler_mod2)
    train_dataloader = neuralnet.get_dataloader(
        train_mod1_X, train_mod2_X, batch_size=128, shuffle=False
    )
    test_mod1_X, _ = neuralnet.preprocess_dataset(test_mod1, scaler_mod1)
    test_mod2_X, _ = neuralnet.preprocess_dataset(test_mod2, scaler_mod2)
    test_dataloader = neuralnet.get_dataloader(
        test_mod1_X, test_mod2_X, batch_size=128, shuffle=False
    )
    log.info("Data is preprocessed")

    # Set configs
    config["input_features"] = train_mod1_X.shape[1]
    config["output_features"] = train_mod2_X.shape[1]
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )

    # Load model
    model = neuralnet.BaselineModel.load_from_checkpoint(checkpoint_path, config=config)
    log.info(f"Model is loaded from {checkpoint_path}")
    model.eval()

    trainer = pl.Trainer(gpus=use_gpu, logger=False)

    train_predictions = trainer.predict(model, train_dataloader)
    train_predictions = torch.cat(train_predictions, dim=0).cpu().numpy()  # type: ignore
    train_predictions = scaler_mod2.inverse_transform(train_predictions)
    print(f"Train target metric: {mp.calculate_target(train_predictions, train_mod2)}")

    test_predictions = trainer.predict(model, test_dataloader)
    test_predictions = torch.cat(test_predictions, dim=0).cpu().numpy()  # type: ignore
    test_predictions = scaler_mod2.inverse_transform(test_predictions)
    print(f"Test target metric: {mp.calculate_target(test_predictions, test_mod2)}")


def save_scaler(scaler: StandardScaler, mod: str, task_type: str, config: dict):
    """Save StandardScaler into file

    Args:
        scaler (StandardScaler):
        mod (str): modality type
        task_type (str): task type
        config (dict):
    """
    mod_config = config.get(mod, {})
    checkpoint_path = mod_config.get(
        "checkpoint_path",
        base_checkpoint_path + mod + "_scaler_" + task_type + ".ckpt",
    )
    with open(checkpoint_path, "wb") as f:
        pickle.dump(scaler, f)
        log.info(f"{mod} StandardScaler is saved to {checkpoint_path}")


def train(config: dict):
    # Load data
    files = dataloader.load_data(config["data"])
    train_mod1 = files["train_mod1"]
    train_mod2 = files["train_mod2"]
    test_mod1 = files["test_mod1"]
    test_mod2 = files["test_mod2"]
    mod1 = utils.get_mod(train_mod1)
    mod2 = utils.get_mod(train_mod2)
    task_type = utils.get_task_type(mod1, mod2)
    assert task_type == "gex_to_adt"
    log.info("Data is loaded")

    # Preprocess data
    train_mod1_X, scaler_mod1 = neuralnet.preprocess_dataset(train_mod1)
    train_mod2_X, scaler_mod2 = neuralnet.preprocess_dataset(train_mod2)
    train_dataloader = neuralnet.get_dataloader(
        train_mod1_X, train_mod2_X, batch_size=128, shuffle=True
    )

    test_mod1_X, _ = neuralnet.preprocess_dataset(test_mod1, scaler_mod1)
    test_mod2_X, _ = neuralnet.preprocess_dataset(test_mod2, scaler_mod2)
    test_dataloader = neuralnet.get_dataloader(
        test_mod1_X, test_mod2_X, batch_size=128, shuffle=False
    )
    log.info("Data is preprocessed")

    # Set configs
    config["input_features"] = train_mod1_X.shape[1]
    config["output_features"] = train_mod2_X.shape[1]
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )

    # Open model
    model = neuralnet.BaselineModel(config)

    # Configure logger
    pl_logger = None
    if config["wandb"]:
        pl_logger = WandbLogger(
            project="nips2021",
            log_model="all",  # type: ignore
            config=config,
            tags=["baseline"],
            config_exclude_keys=["wandb"],
        )
        pl_logger.watch(model)

    # Train model
    trainer = pl.Trainer(gpus=use_gpu, checkpoint_callback=False, logger=pl_logger)
    trainer.fit(model, train_dataloader, test_dataloader)

    # Save model
    trainer.save_checkpoint(checkpoint_path)
    log.info(f"Model is saved to {checkpoint_path}")
    save_scaler(scaler_mod1, mod1, task_type, config)
    save_scaler(scaler_mod2, mod2, task_type, config)


def get_parser():
    parser = argparse.ArgumentParser(description="Pytorch Baseline for MP task")
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=argparse.FileType("r"))
    parser_train.add_argument("--wandb", action="store_true", default=False)
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
        config["wandb"] = args.wandb
        train(config)
    elif args.action == "evaluate":
        evaluate(config)
    else:
        print("Enter command [train, evaluate]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
