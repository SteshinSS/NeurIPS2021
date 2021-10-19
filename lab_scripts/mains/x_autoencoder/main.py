import argparse
import logging

import anndata as ad
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from lab_scripts.data import dataloader
from lab_scripts.mains.x_autoencoder import preprocessing, tune
from lab_scripts.mains.x_autoencoder.preprocessing import (
    base_checkpoint_path, base_config_path)
from lab_scripts.models import x_autoencoder
from lab_scripts.utils import utils
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix

log = logging.getLogger("x_autoencoder")


def update_model_config(model_config: dict, preprocessed_data: dict):
    model_config["first"]["input_features"] = preprocessed_data["first_input_features"]
    model_config["first"]["target_features"] = preprocessed_data[
        "first_target_features"
    ]
    model_config["second"]["input_features"] = preprocessed_data[
        "second_input_features"
    ]
    model_config["second"]["target_features"] = preprocessed_data[
        "second_target_features"
    ]
    model_config["total_batches"] = torch.unique(
        preprocessed_data["train_batch_idx"]
    ).shape[0]
    model_config["batch_weights"] = preprocessed_data["train_batch_weights"]
    return model_config


def get_logger(config):
    pl_logger = None
    if config["wandb"]:
        pl_logger = WandbLogger(
            project="nips2021",
            log_model=False,  # type: ignore
            config=config,
            tags=["x_autoencoder"],
            config_exclude_keys=["wandb"],
        )

        pl_logger.experiment.define_metric(name="true_1_to_2", summary="min")
        pl_logger.experiment.define_metric(name="true_2_to_1", summary="min")
    return pl_logger


def get_callbacks(preprocessed_data: dict, dataset: dict):
    validation_callback = x_autoencoder.TargetCallback(
        test_dataloader=preprocessed_data["test_dataloader"],
        first_inverse=preprocessed_data["first_test_inverse"],
        first_true_target=dataset["test_mod1"],
        second_inverse=preprocessed_data["second_test_inverse"],
        second_true_target=dataset["test_mod2"],
    )
    small_idx = preprocessed_data["small_idx"]
    validation_train_callback = x_autoencoder.TargetCallback(
        test_dataloader=preprocessed_data["small_dataloader"],
        first_inverse=preprocessed_data["first_train_inverse"],
        first_true_target=dataset["train_mod1"][small_idx],
        second_inverse=preprocessed_data["second_train_inverse"],
        second_true_target=dataset["train_mod2"][small_idx],
        prefix="train",
    )

    learning_rate_monitor = LearningRateMonitor(
        logging_interval="step",
    )
    callbacks = [validation_callback, validation_train_callback, learning_rate_monitor]
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
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_data(data_config["dataset_name"])
    first_mod = utils.get_mod(dataset["train_mod1"])
    second_mod = utils.get_mod(dataset["train_mod2"])
    task_type = utils.get_task_type(first_mod, second_mod)
    log.info("Data is loaded")

    # Preprocess data
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=False
    )

    train_dataloader = preprocessed_data["train_dataloader"]
    first_train_inverse = preprocessed_data["first_train_inverse"]
    second_train_inverse = preprocessed_data["second_train_inverse"]

    test_dataloader = preprocessed_data["test_dataloader"]
    first_test_inverse = preprocessed_data["first_test_inverse"]
    second_test_inverse = preprocessed_data["second_test_inverse"]

    # Add input feature size
    model_config = update_model_config(model_config, preprocessed_data)
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
    def run_dataset(
        dataloader, name, first_target, first_inverse, second_target, second_inverse
    ):
        predictions = trainer.predict(model, dataloader)
        first_to_second = []
        second_to_first = []
        for (first_to_second_batch, second_to_first_batch) in predictions:  # type: ignore
            first_to_second.append(first_to_second_batch.cpu())
            second_to_first.append(second_to_first_batch.cpu())
        print(
            f"{name} {first_mod} to {second_mod} metric: {x_autoencoder.calculate_metric(first_to_second, second_inverse, second_target)}"
        )
        print(
            f"{name} {second_mod} to {first_mod} metric: {x_autoencoder.calculate_metric(second_to_first, first_inverse, first_target)}"
        )

    run_dataset(
        train_dataloader,
        "Train",
        dataset["train_mod1"],
        first_train_inverse,
        dataset["train_mod2"],
        second_train_inverse,
    )
    run_dataset(
        test_dataloader,
        "Test",
        dataset["test_mod1"],
        first_test_inverse,
        dataset["test_mod2"],
        second_test_inverse,
    )


def train(config: dict):
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_data(data_config["dataset_name"])
    log.info("Data is loaded")

    # Preprocess data
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=True
    )
    train_dataloader = preprocessed_data["train_dataloader"]
    model_config = update_model_config(model_config, preprocessed_data)
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, dataset)
    if not pl_logger:
        callbacks.pop()

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore

    # Train model
    model = x_autoencoder.X_autoencoder(model_config)
    if pl_logger:
        pl_logger.watch(model)

    trainer = pl.Trainer(
        gpus=use_gpu,
        max_epochs=5000,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)

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
