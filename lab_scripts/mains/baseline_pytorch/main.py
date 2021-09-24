"""Template main for the Modality Prediction task. Copy it and edit."""
import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import yaml  # type: ignore
from scipy.sparse import csr_matrix

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
    pass


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
    train_mod1_X = utils.convert_to_dense(train_mod1).X
    scaler_mod1 = StandardScaler()
    train_mod1_X = scaler_mod1.fit_transform(train_mod1_X)

    train_mod2_X = utils.convert_to_dense(train_mod2).X
    scaler_mod2 = StandardScaler()
    train_mod2_X = scaler_mod2.fit_transform(train_mod2_X)
    train_dataset = neuralnet.BaselineDataloader(train_mod1_X, train_mod2_X)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_mod1_X = utils.convert_to_dense(test_mod1).X
    test_mod1_X = scaler_mod1.transform(test_mod1_X)
    test_mod2_X = utils.convert_to_dense(test_mod2).X
    test_mod2_X = scaler_mod2.transform(test_mod2_X)
    test_dataset = neuralnet.BaselineDataloader(test_mod1_X, test_mod2_X)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
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

    trainer = pl.Trainer(gpus=use_gpu)

    train_predictions = trainer.predict(model, train_dataloader)
    train_predictions = torch.cat(train_predictions, dim=0).cpu().numpy()
    train_predictions = scaler_mod2.inverse_transform(train_predictions)
    print(f"Train target metric: {mp.calculate_target(train_predictions, train_mod2)}")

    test_predictions = trainer.predict(model, test_dataloader)
    test_predictions = torch.cat(test_predictions, dim=0).cpu().numpy()
    test_predictions = scaler_mod2.inverse_transform(test_predictions)
    print(f"Test target metric: {mp.calculate_target(test_predictions, test_mod2)}")


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
    train_mod1_X = utils.convert_to_dense(train_mod1).X
    scaler_mod1 = StandardScaler()
    train_mod1_X = scaler_mod1.fit_transform(train_mod1_X)

    train_mod2_X = utils.convert_to_dense(train_mod2).X
    scaler_mod2 = StandardScaler()
    train_mod2_X = scaler_mod2.fit_transform(train_mod2_X)
    train_dataset = neuralnet.BaselineDataloader(train_mod1_X, train_mod2_X)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_mod1_X = utils.convert_to_dense(test_mod1).X
    test_mod1_X = scaler_mod1.transform(test_mod1_X)
    test_mod2_X = utils.convert_to_dense(test_mod2).X
    test_mod2_X = scaler_mod2.transform(test_mod2_X)
    test_dataset = neuralnet.BaselineDataloader(test_mod1_X, test_mod2_X)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
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

    # Train model
    model = neuralnet.BaselineModel(config)
    trainer = pl.Trainer(gpus=use_gpu, checkpoint_callback=False)
    trainer.fit(model, train_dataloader, test_dataloader)

    # Save model
    trainer.save_checkpoint(checkpoint_path)
    log.info(f"Model is saved to {checkpoint_path}")


def get_parser():
    """Creates parser.

    Remove lines with adding config, if you don't need it.
    """
    parser = argparse.ArgumentParser(description="Pytorch Baseline for MP task")
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
