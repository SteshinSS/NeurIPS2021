import argparse
import logging

import anndata as ad
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from lab_scripts.data import dataloader
from lab_scripts.mains.mm import preprocessing, tune, common
from lab_scripts.mains.mm.preprocessing import base_checkpoint_path, base_config_path
from lab_scripts.models import clip
from lab_scripts.utils import utils
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix

log = logging.getLogger("mm")


def get_logger(config):
    pl_logger = None
    if config["wandb"]:
        pl_logger = WandbLogger(
            project="mm",
            log_model=False,  # type: ignore
            config=config,
            tags=["baseline"],
            config_exclude_keys=["wandb"],
        )
        pl_logger.experiment.define_metric(name="test_top0.05", summary="max")
        pl_logger.experiment.define_metric(name="test_top0.01", summary="max")
        pl_logger.experiment.define_metric(name="test_top0.1", summary="max")
    return pl_logger


def get_callbacks(preprocessed_data: dict, model_config: dict, logger=None):
    callbacks = []
    small_train_callback = clip.TargetCallback(
        preprocessed_data["small_train_dataloader"],
        model_config["predict_temperature"],
        "train",
    )
    callbacks.append(small_train_callback)

    if "val_dataloader" in preprocessed_data:
        val_callback = clip.TargetCallback(
            preprocessed_data["val_dataloader"], model_config["predict_temperature"], "val", log_top=[0.05, 0.01]
        )
        callbacks.append(val_callback)

    log_preds = logger is not None
    test_callback = clip.TargetCallback(
        preprocessed_data["test_dataloader"], model_config["predict_temperature"], "test", log_top=[0.1, 0.05, 0.01], log_preds=log_preds
    )
    callbacks.append(test_callback)

    if logger is not None:
        learning_rate_monitor = LearningRateMonitor(
            logging_interval="step",
        )
        callbacks.append(learning_rate_monitor)
    return callbacks


def predict_submission(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
    resources_dir: str = "",
) -> ad.AnnData:
    log.info("Start x_autoencoder prediction...")
    raise NotImplementedError()

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
    pass
    # Load data


def train(config: dict):
    # Solution of strange bug.
    # See https://github.com/pytorch/pytorch/issues/57794#issuecomment-834976384
    torch.cuda.set_device(0)
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_custom_mm_data(
        data_config['task_type'],
        data_config['train_batches'],
        data_config['test_batches'],
        filter_genes_params=(data_config["gene_fraction"], "data/genes.csv"),
        val_size=data_config["val_size"],
    )
    log.info("Data is loaded")

    # Preprocess data
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=True
    )
    train_dataloaders = preprocessed_data["train_shuffled_dataloader"]
    model_config = common.update_model_config(model_config, preprocessed_data)
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, model_config, pl_logger)

    # Train model
    model = clip.Clip(model_config)
    if pl_logger:
        pl_logger.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5000,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config["gradient_clip"],
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
    parser = argparse.ArgumentParser(description="X autoencoder.")
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
        tune.tune_hp(config)
    else:
        print("Enter command [train, evaluate, tune]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    utils.set_deafult_seed()
    cli()
