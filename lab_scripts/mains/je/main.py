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
from lab_scripts.mains.je.model import JEAutoencoder
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
    log.info("Start MM prediction...")

    # Load data
    mod1 = utils.get_mod(input_mod1)
    mod2 = utils.get_mod(input_mod2)
    log.info("Data is loaded")

    # Select data type
    task_type = utils.get_task_type(mod1, mod2)
    predictions = None
    if task_type == "gex_to_atac":
        # predictions = predict_gex_to_atac(...)
        pass
    elif task_type == "atac_to_gex":
        # predictions = predict_atac_to_gex(...)
        pass
    elif task_type in ["gex_to_adt", "adt_to_gex"]:
        predictions = predict_cite(input_mod1, input_mod2, resources_dir)
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


def predict_cite(input_mod1, input_mod2, resources_dir):
    config_path = resources_dir + base_config_path + "cite_pre" + ".yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    model_config = config["model"]
    dataset = {
        "test_mod1": input_mod1,
        "test_mod2": input_mod2,
    }
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=False
    )
    model_config = preprocessing.update_model_config(model_config, preprocessed_data)
    checkpoint_path = (
        resources_dir + base_checkpoint_path + data_config["task_type"] + ".ckpt"
    )
    model = JEAutoencoder.load_from_checkpoint(checkpoint_path, config=model_config)
    model.eval()
    predictions = []  # type: ignore
    with torch.no_grad():
        for i, batch in enumerate(preprocessed_data["test_dataloader"]):
            first, second = batch
            predictions.append(model(first, second))
    predictions = torch.cat(predictions, dim=0)  # type: ignore
    return predictions.numpy()  # type: ignore


def evaluate(config: dict):
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_custom_je_data(
        data_config["task_type"],
        data_config["train_batches"],
        data_config["test_batches"],
        val_size=0,
    )
    log.info("Data is loaded")

    # Preprocess data
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=False
    )
    model_config = preprocessing.update_model_config(model_config, preprocessed_data)
    log.info("Data is preprocessed")

    # Load model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + data_config["task_type"] + ".ckpt"
    )
    model = JEAutoencoder.load_from_checkpoint(checkpoint_path, config=model_config)
    model.eval()
    predictions = []  # type: ignore
    with torch.no_grad():
        for i, batch in enumerate(preprocessed_data["test_dataloader"]):
            first, second = batch
            predictions.append(model(first, second))
    predictions = torch.cat(predictions, dim=0)  # type: ignore
    prediction = je_metrics.create_anndata(dataset["test_solution"], predictions.numpy())  # type: ignore
    all_metrics = je_metrics.calculate_metrics(prediction, dataset["test_solution"])
    print(all_metrics)


def get_logger(config):
    pl_logger = None
    if config["wandb"]:
        pl_logger = WandbLogger(
            project="je",
            log_model=False,  # type: ignore
            config=config,
            tags=[config['data']["task_type"]],
            config_exclude_keys=["wandb"],
        )
        # pl_logger.experiment.define_metric(name="test_top0.05", summary="max")
    return pl_logger


def get_callbacks(
    preprocessed_data: dict, dataset: dict, model_config: dict, logger=None
):
    callbacks = []

    val_callback = TargetCallback(
        dataset["test_solution"],
        preprocessed_data["test_dataloader"],
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
        data_config["train_batches"],
        data_config["test_batches"],
        val_size=data_config["val_size"],
    )
    log.info("Data is loaded")

    # Preprocess data
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=True
    )
    model_config = preprocessing.update_model_config(model_config, preprocessed_data)
    train_dataloaders = [preprocessed_data["train_shuffled_dataloader"]]
    train_dataloaders.extend(preprocessed_data["correction_dataloaders"])
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

    # Save model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + data_config["task_type"] + ".ckpt"
    )
    trainer.save_checkpoint(checkpoint_path)
    log.info(f"Model is saved to {checkpoint_path}")


def get_parser():
    """Creates parser.

    Remove lines with adding config, if you don't need it.
    """
    parser = argparse.ArgumentParser(description="JE.")
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=argparse.FileType("r"))
    parser_train.add_argument("--wandb", action="store_true", default=False)
    parser_evaluate = subparsers.add_parser("evaluate")
    parser_evaluate.add_argument("config", type=argparse.FileType("r"))
    parser_tune = subparsers.add_parser("tune")
    parser_tune.add_argument("config", type=argparse.FileType("r"))
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
    elif args.action == "tune":
        from lab_scripts.mains.je import tune

        tune.tune_hp(config)
    else:
        print("Enter command [train, evaluate, tune]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    utils.set_deafult_seed()
    from lab_scripts.mains.je.callback import TargetCallback
    from lab_scripts.metrics import je as je_metrics

    cli()
