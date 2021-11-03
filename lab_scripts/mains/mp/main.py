import argparse
import logging

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from lab_scripts.data import dataloader
from lab_scripts.mains.mp import model as mp_model
from lab_scripts.mains.mp import preprocessing, tune
from lab_scripts.mains.mp.preprocessing import (base_checkpoint_path,
                                                base_config_path)
from lab_scripts.utils import utils
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix

log = logging.getLogger("mp")


def predict_submission(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
    resources_dir: str = "",
) -> ad.AnnData:
    log.info("Start MP prediction...")

    # Load data
    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    log.info("Data is loaded")

    # Select data type
    task_type = utils.get_task_type(mod1, mod2)
    
    predictions = predict(
        input_train_mod1,
        input_train_mod2,
        input_test_mod1,
        task_type,
        resources_dir,
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


def predict(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
    task_type: str,
    resources_dir,
):
    config_path = resources_dir + base_config_path + task_type + ".yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    data_config = config["data"]
    dataset = {
        "train_mod1": input_train_mod1,
        "train_mod2": input_train_mod2,
        "test_mod1": input_test_mod1,
    }
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, mode="test", resources_dir=resources_dir
    )

    test_dataloader = preprocessed_data["test_dataloader"]
    second_test_inverse = preprocessed_data["second_test_inverse"]

    # Add input feature size
    model_config = preprocessing.update_model_config(config, preprocessed_data)
    log.info("Data is preprocessed")

    # Load model
    checkpoint_path = (
        resources_dir + base_checkpoint_path + data_config["task_type"] + ".ckpt"
    )
    model = mp_model.Predictor.load_from_checkpoint(
        checkpoint_path, config=model_config
    )
    log.info(f"Model is loaded from {checkpoint_path}")

    model.eval()
    second_pred = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            prediction = model.predict_step(batch, i)
            second_pred.append(prediction.cpu())
    second_pred = torch.cat(second_pred, dim=0)  # type: ignore
    second_pred = second_test_inverse(second_pred)
    return second_pred  # type: ignore


def get_logger(config):
    pl_logger = None
    if config["wandb"]:
        pl_logger = WandbLogger(
            project="mp",
            log_model=False,  # type: ignore
            config=config,
            tags=[config["data"]["task_type"]],
            config_exclude_keys=["wandb"],
        )
        pl_logger.experiment.define_metric(name="train_m", summary="min")
        pl_logger.experiment.define_metric(name="test_m", summary="min")
    return pl_logger


def get_callbacks(
    preprocessed_data: dict, dataset: dict, model_config: dict, logger=None
):
    small_idx = preprocessed_data["small_idx"]
    train_callback = mp_model.TargetCallback(
        preprocessed_data["small_train_dataloader"],
        preprocessed_data["second_train_inverse"],
        dataset["train_mod2"][small_idx],
        prefix="train",
    )
    callbacks = [train_callback]

    if "val_dataloader" in preprocessed_data:
        val_callback = mp_model.TargetCallback(
            preprocessed_data["val_dataloader"],
            preprocessed_data["second_val_inverse"],
            dataset["val_mod2"],
            prefix="val",
        )
        callbacks.append(val_callback)

    test_callback = mp_model.TargetCallback(
        preprocessed_data["test_dataloader"],
        preprocessed_data["second_test_inverse"],
        dataset["test_mod2"],
        prefix="test",
    )
    callbacks.append(test_callback)

    if model_config["do_tsne"]:
        tsne_callback = mp_model.BatchEffectCallback(
            train_dataset=preprocessed_data["train_unshuffled_dataloader"],
            test_dataset=preprocessed_data["test_dataloader"],
            frequency=model_config["tsne_frequency"],
        )
        callbacks.append(tsne_callback)

    if logger is not None:
        learning_rate_monitor = LearningRateMonitor(
            logging_interval="step",
        )
        callbacks.append(learning_rate_monitor)
    return callbacks


def train(config: dict):
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_custom_mp_data(
        task_type=data_config["task_type"],
        train_batches=data_config["train_batches"],
        test_batches=data_config["test_batches"],
        val_size=data_config["val_size"],
    )
    log.info("Data is loaded")

    # Preprocess data
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, mode="train"
    )
    model_config = preprocessing.update_model_config(config, preprocessed_data)
    if model_config["total_correction_batches"] > 0:
        train_dataloaders = [preprocessed_data["train_shuffled_dataloader"]]
        train_dataloaders.extend(preprocessed_data["correction_dataloaders"])
    else:
        train_dataloaders = preprocessed_data["train_shuffled_dataloader"]
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, dataset, model_config, pl_logger)

    # Train model
    model = mp_model.Predictor(model_config)
    if pl_logger:
        pl_logger.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5000,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config["gradient_clip"]
        if not model_config["use_critic"]
        else 0.0,
    )
    trainer.fit(model, train_dataloaders=train_dataloaders)

    # Save model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + data_config["task_type"] + ".ckpt"
    )
    trainer.save_checkpoint(checkpoint_path)
    log.info(f"Model is saved to {checkpoint_path}")

    if model.use_vi_dropout:
        weights_path = base_checkpoint_path + "genes.csv"
        weights = model.vi_dropout.weight.detach().cpu().numpy()
        np.savetxt(weights_path, weights, delimiter=",")
        log.info(f"Genes weights are saved to {weights_path}")


def get_parser():
    """Creates parser.

    Remove lines with adding config, if you don't need it.
    """
    parser = argparse.ArgumentParser(description="Modality Prediction")
    subparsers = parser.add_subparsers(dest="action")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("config", type=argparse.FileType("r"))
    parser_train.add_argument("--wandb", action="store_true", default=False)
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
    elif args.action == "tune":
        tune.tune_hp(config)
    else:
        print("Enter command [train, evaluate, tune]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    utils.set_deafult_seed()
    cli()
