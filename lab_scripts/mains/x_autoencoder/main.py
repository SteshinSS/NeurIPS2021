import argparse
import logging
import pickle
from typing import Callable, Optional

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from lab_scripts.data import dataloader
from lab_scripts.data.integration import processor
from lab_scripts.metrics import mp
from lab_scripts.models import x_autoencoder
from lab_scripts.utils import utils
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

log = logging.getLogger("x_autoencoder")
base_config_path = "configs/mp/x_autoencoder/"
base_checkpoint_path = "checkpoints/mp/x_autoencoder/"


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


def get_processor_path(mod_config: dict, task_type: str, mod: str):
    return mod_config.get(
        "checkpoint_path", base_checkpoint_path + task_type + "_" + mod + ".ckpt"
    )


def evaluate(config: dict):
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_data(data_config["dataset_name"])
    first_mod = utils.get_mod(dataset["train_mod1"])
    second_mod = utils.get_mod(dataset["train_mod2"])
    task_type = utils.get_task_type(first_mod, second_mod)
    log.info("Data is loaded")

    # Preprocess data
    first_config = data_config.get("mod1", {})
    first_processor_checkpoint_path = get_processor_path(
        first_config, task_type, first_mod
    )
    with open(first_processor_checkpoint_path, "rb") as f:
        first_processor = pickle.load(f)
    train_first_X, _ = first_processor.transform(dataset["train_mod1"])
    test_first_X, _ = first_processor.transform(dataset["test_mod1"])

    second_config = data_config.get("mod2", {})
    second_processor_checkpoint_path = get_processor_path(
        second_config, task_type, second_mod
    )
    with open(second_processor_checkpoint_path, "rb") as f:
        second_processor = pickle.load(f)
    train_second_X, train_second_inverse = second_processor.transform(dataset["train_mod2"])
    test_second_X, test_second_inverse = second_processor.transform(dataset["test_mod2"])

    # Add input feature size
    model_config = config["model"]
    model_config["first_dims"].insert(0, train_first_X.shape[1])
    model_config["second_dims"].insert(0, train_second_X.shape[1])

    train_dataset = processor.TwoOmicsDataset(train_first_X, train_second_X)
    train_dataloader = DataLoader(train_dataset, batch_size=model_config["batch_size"])
    test_dataset = processor.TwoOmicsDataset(test_first_X, test_second_X)
    test_dataloader = DataLoader(test_dataset, batch_size=model_config["batch_size"])
    log.info("Data is preprocessed")

    # Load model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )
    model = x_autoencoder.X_autoencoder.load_from_checkpoint(
        checkpoint_path, config=model_config
    )
    log.info(f"Model is loaded from {checkpoint_path}")
    model.eval()
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore

    trainer = pl.Trainer(gpus=use_gpu, logger=False)

    # Run predictions
    train_predictions = trainer.predict(model, train_dataloader)
    train_predictions = torch.cat(train_predictions, dim=0).cpu() # type: ignore
    train_predictions = train_second_inverse(train_predictions)
    print(
        f"Train target metric: {mp.calculate_target(train_predictions, dataset['train_mod2'])}"
    )

    test_predictions = trainer.predict(model, test_dataloader)
    test_predictions = torch.cat(test_predictions, dim=0).cpu() # type: ignore
    test_predictions = test_second_inverse(test_predictions)
    print(
        f"Test target metric: {mp.calculate_target(test_predictions, dataset['test_mod2'])}"
    )


def train(config: dict):
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_data(data_config["dataset_name"])
    first_mod = utils.get_mod(dataset["train_mod1"])
    second_mod = utils.get_mod(dataset["train_mod2"])
    task_type = utils.get_task_type(first_mod, second_mod)
    log.info("Data is loaded")

    # Preprocess data
    first_mod_config = data_config['mod1']
    first_processor = processor.Processor(first_mod_config, first_mod)
    train_first_X, _ = first_processor.fit_transform(dataset["train_mod1"])
    test_first_X, _ = first_processor.transform(dataset["test_mod1"])

    second_mod_config = data_config['mod2']
    second_processor = processor.Processor(second_mod_config, second_mod)
    train_second_X, _ = second_processor.fit_transform(dataset["train_mod2"])
    test_second_X, _ = second_processor.transform(dataset["test_mod2"])

    # Add input feature size
    model_config = config["model"]
    model_config["first_dims"].insert(0, train_first_X.shape[1])
    model_config["second_dims"].insert(0, train_second_X.shape[1])

    train_dataset = processor.TwoOmicsDataset(train_first_X, train_second_X)
    train_dataloader = DataLoader(
        train_dataset, batch_size=model_config["batch_size"], shuffle=True
    )
    test_dataset = processor.TwoOmicsDataset(test_first_X, test_second_X)
    test_dataloader = DataLoader(test_dataset, batch_size=model_config["batch_size"])
    log.info("Data is preprocessed")

    # Configure logger
    pl_logger = None
    if config["wandb"]:
        pl_logger = WandbLogger(
            project="nips2021",
            log_model="all",  # type: ignore
            config=config,
            tags=["baseline", "x_autoencoder"],
            config_exclude_keys=["wandb"],
        )

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore

    # Train model
    model = x_autoencoder.X_autoencoder(model_config)
    trainer = pl.Trainer(
        gpus=use_gpu, max_epochs=5000, checkpoint_callback=False, logger=pl_logger,
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )

    # Save model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + task_type + ".ckpt"
    )
    trainer.save_checkpoint(checkpoint_path)
    log.info(f"Model is saved to {checkpoint_path}")

    # Save processors
    first_processor_checkpoint_path = get_processor_path(
        first_mod_config, task_type, first_mod
    )
    with open(first_processor_checkpoint_path, "wb") as f:
        pickle.dump(first_processor, f)
    log.info(f"{first_mod} processor is saved to {first_processor_checkpoint_path}")

    second_processor_checkpoint_path = get_processor_path(
        second_mod_config, task_type, second_mod
    )
    with open(second_processor_checkpoint_path, "wb") as f:
        pickle.dump(second_processor, f)
    log.info(f"{second_mod} processor is saved to {second_processor_checkpoint_path}")


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
    utils.set_deafult_seed()
    cli()