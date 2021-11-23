import argparse
import logging

log = logging.getLogger("je")
logging.basicConfig(level=logging.INFO)

import anndata as ad
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from lab_scripts.data import dataloader
from lab_scripts.mains.je import preprocessing
from lab_scripts.mains.je.model import JEAutoencoder, TargetCallback
from lab_scripts.mains.je.preprocessing import (base_checkpoint_path,
                                                base_config_path)
from lab_scripts.utils import utils
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix


def predict_submission(
    input_mod1: ad.AnnData,
    input_mod2: ad.AnnData,
    resources_dir: str = "",
) -> ad.AnnData:
    log.info("Start JE prediction...")

    # Load data
    mod1 = utils.get_mod(input_mod1)
    mod2 = utils.get_mod(input_mod2)
    log.info("Data is loaded")

    # Select data type
    task_type = utils.get_task_type(mod1, mod2)
    predictions = None
    if task_type in ["gex_to_atac", "atac_to_gex"]:
        pass
    elif task_type in ["gex_to_adt", "adt_to_gex"]:
        predictions = train_cite(input_mod1, input_mod2, resources_dir)
    else:
        raise ValueError(f"Inappropriate dataset types: {task_type}")

    # Convert matrix into csr_matrix (is needed for submission)
    predictions = csr_matrix(predictions)

    # Create AnnData object
    result = ad.AnnData(
        X=predictions,
        uns={"dataset_id": input_mod1.uns["dataset_id"]},
    )
    return result


def train_cite(input_mod1, input_mod2, resources_dir):
    config_path = resources_dir + base_config_path + "cite_pre" + ".yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    model_config = config["model"]
    dataset = {
        "train_mod1": input_mod1,
        "train_mod2": input_mod2,
    }
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset
    )
    model_config = preprocessing.update_model_config(config, preprocessed_data)
    if data_config['batch_correct']:
        train_dataloaders = [preprocessed_data["train_shuffled_dataloader"]]
        train_dataloaders.extend(preprocessed_data["correction_dataloaders"])
    else:
        train_dataloaders = preprocessed_data["train_shuffled_dataloader"]
    log.info("Data is preprocessed")


    # Train model
    model = JEAutoencoder(model_config)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5000,
        logger=None,
        callbacks=[],
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config["gradient_clip"] if not model_config['use_critic'] else 0.0,
    )

    trainer.fit(model, train_dataloaders=train_dataloaders)

    model.eval()
    predictions = []  # type: ignore
    with torch.no_grad():
        for i, batch in enumerate(preprocessed_data["train_unshuffled_dataloader"]):
            first, second = batch
            predictions.append(model(first, second))
    predictions = torch.cat(predictions, dim=0)  # type: ignore
    return predictions.numpy()  # type: ignore



def get_logger(config):
    pl_logger = None
    if config["wandb"]:
        task_type = config['data']["task_type"]
        pl_logger = WandbLogger(
            project="je_" + task_type,
            log_model=False,  # type: ignore
            config=config,
            tags=[],
            config_exclude_keys=["wandb"],
        )
        # pl_logger.experiment.define_metric(name="test_top0.05", summary="max")
    return pl_logger


def get_callbacks(
    preprocessed_data: dict, dataset: dict, model_config: dict, logger=None
):
    callbacks = []

    val_callback = TargetCallback(
        dataset["train_solution"],
        preprocessed_data["train_unshuffled_dataloader"],
        frequency=10,
    )
    callbacks.append(val_callback)

    if logger is not None:
        learning_rate_monitor = LearningRateMonitor(
            logging_interval="step",
        )
        callbacks.append(learning_rate_monitor)
    return callbacks


def train(config: dict):
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_custom_je_data(
        data_config["task_type"],
        data_config["train_batches"]
    )
    log.info("Data is loaded")

    # Preprocess data
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset
    )
    model_config = preprocessing.update_model_config(config, preprocessed_data)
    if data_config['batch_correct']:
        train_dataloaders = [preprocessed_data["train_shuffled_dataloader"]]
        train_dataloaders.extend(preprocessed_data["correction_dataloaders"])
    else:
        train_dataloaders = preprocessed_data["train_shuffled_dataloader"]
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, dataset, model_config, pl_logger)

    # Train model
    model = JEAutoencoder(model_config)
    if pl_logger:
        pl_logger.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5000,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config["gradient_clip"] if not model_config['use_critic'] else 0.0,
    )
    trainer.fit(model, train_dataloaders=train_dataloaders)


def get_parser():
    """Creates parser.

    Remove lines with adding config, if you don't need it.
    """
    parser = argparse.ArgumentParser(description="JE.")
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=argparse.FileType("r"))
    parser_train.add_argument("--wandb", action="store_true", default=False)
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
    else:
        print("Enter command [train, evaluate, tune]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    utils.set_deafult_seed()
    cli()
