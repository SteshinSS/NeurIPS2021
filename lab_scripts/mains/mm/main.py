import argparse
import logging

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from lab_scripts.data import dataloader
from lab_scripts.mains.mm import clip, preprocessing, tune
from lab_scripts.mains.mm.preprocessing import (base_checkpoint_path,
                                                base_config_path)
from lab_scripts.metrics import mm
from lab_scripts.utils import utils
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix
from torch import nn

log = logging.getLogger("mm")


def predict_submission(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_train_sol: ad.AnnData,
    input_test_mod1: ad.AnnData,
    input_test_mod2: ad.AnnData,
    resources_dir: str = "",
) -> ad.AnnData:
    log.info("Start MM prediction...")

    # Load data
    mod1 = utils.get_mod(input_train_mod1)
    mod2 = utils.get_mod(input_train_mod2)
    log.info("Data is loaded")

    # Select data type
    task_type = utils.get_task_type(mod1, mod2)
    predictions = predict(
        input_train_mod1,
        input_train_mod2,
        input_train_sol,
        input_test_mod1,
        input_test_mod2,
        resources_dir,
        task_type,
    )
    # Convert matrix into csr_matrix (is needed for submission)
    predictions = csr_matrix(predictions)

    # Create AnnData object
    result = ad.AnnData(
        X=predictions,
        uns={"dataset_id": input_train_mod1.uns["dataset_id"]},
    )
    return result


def predict(
    input_train_mod1,
    input_train_mod2,
    input_train_sol,
    input_test_mod1,
    input_test_mod2,
    resources_dir,
    task_type,
):
    dataset = {
        "train_mod1": input_train_mod1,
        "train_mod2": input_train_mod2,
        "train_sol": input_train_sol,
        "test_mod1": input_test_mod1,
        "test_mod2": input_test_mod2,
    }

    # Here are our resources
    config_path = resources_dir + base_config_path + task_type + ".yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, mode="test", resources_dir=resources_dir
    )
    model_config = preprocessing.update_model_config(config, preprocessed_data)
    checkpoint_path = resources_dir + base_checkpoint_path + task_type + ".ckpt"
    model = clip.Clip.load_from_checkpoint(checkpoint_path, config=model_config)

    first_pred = []
    second_pred = []
    all_batches = []
    with torch.no_grad():
        for i, batch in enumerate(preprocessed_data["test_dataloader"]):  # type: ignore
            first, second, batch_idx = batch
            first, second, _ = model(first, second)
            first_pred.append(first.cpu())
            second_pred.append(second.cpu())
            all_batches.append(batch_idx.cpu())
    first = torch.cat(first_pred, dim=0)  # type: ignore
    second = torch.cat(second_pred, dim=0)  # type: ignore
    all_batches = torch.cat(all_batches, dim=0)  # type: ignore
    embeddings = first @ second.t()  # type: ignore
    unique_batches = torch.unique(all_batches)
    for batch in unique_batches:
        idx = all_batches == batch
        embeddings[idx][:, ~idx] = -1e9
    final_predictions = embeddings * np.exp(model_config["predict_temperature"])
    final_predictions = torch.softmax(final_predictions, dim=1)
    (_, best_idx) = torch.sort(final_predictions, descending=True)
    for i in range(final_predictions.shape[1]):
        worst_row_idx = best_idx[i][999:]
        final_predictions[i][worst_row_idx] = 0.0
    final_predictions /= final_predictions.sum(axis=1)
    return final_predictions.detach().cpu().numpy()


def evaluate(config: dict):
    torch.cuda.set_device(0)
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_custom_mm_data(
        data_config["task_type"],
        data_config["train_batches"],
        data_config["test_batches"],
        val_size=0,
    )
    log.info("Data is loaded")

    # Preprocess data
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(data_config, dataset, mode="train")
    model_config = preprocessing.update_model_config(config, preprocessed_data)
    log.info("Data is preprocessed")

    # Load model
    checkpoint_path = config.get(
        "checkpoint_path", base_checkpoint_path + data_config["task_type"] + ".ckpt"
    )
    model = clip.Clip.load_from_checkpoint(checkpoint_path, config=model_config)
    model.eval()

    def run_for_dataset(dataset, prefix, temps=None):
        first_pred = []
        second_pred = []
        all_batches = []
        with torch.no_grad():
            for i, batch in enumerate(dataset):  # type: ignore
                first, second, batch_idx = batch
                first, second, _ = model(first, second)
                first_pred.append(first.cpu())
                second_pred.append(second.cpu())
                all_batches.append(batch_idx.cpu())
        first = torch.cat(first_pred, dim=0)  # type: ignore
        second = torch.cat(second_pred, dim=0)  # type: ignore
        all_batches = torch.cat(all_batches, dim=0)  # type: ignore
        embeddings = first @ second.t()  # type: ignore

        print(f"{prefix} metric:")
        if temps:
            for temp in temps:
                print(f"Temp: {temp}", calculate_metric(embeddings, temp, all_batches))
        else:
            best_temp = find_best_temperature(embeddings.cuda())
            print(
                f"Final metric: ", calculate_metric(embeddings, best_temp, all_batches)
            )

    run_for_dataset(preprocessed_data["train_unshuffled_dataloader"], "Train", [7.86])
    run_for_dataset(preprocessed_data["test_dataloader"], "Test")


def calculate_metric(embeddings, temperature, all_batches):
    final_predictions = embeddings * np.exp(temperature)
    unique_batches = torch.unique(all_batches)
    for batch in unique_batches:
        idx = all_batches == batch
        embeddings[idx][:, ~idx] = -1e6
    final_predictions = torch.softmax(final_predictions, dim=1)
    (_, best_idx) = torch.sort(final_predictions, descending=True)
    for i in range(final_predictions.shape[1]):
        worst_row_idx = best_idx[i][999:]
        final_predictions[i][worst_row_idx] = 0.0
    final_predictions /= final_predictions.sum(axis=1)
    return mm.calculate_target(final_predictions)


class TempModule(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.t = nn.Parameter(torch.ones([]) * 1)

    def forward(self):
        loss = torch.softmax(self.embeddings * torch.exp(self.t), dim=0)
        return -loss.diag().sum()


def find_best_temperature(embeddings):
    module = TempModule(embeddings).cuda()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    for i in range(100):
        pred = module()
        optimizer.zero_grad()
        pred.backward()
        optimizer.step()
        print(pred.item())
    print()
    best_temp = module.t.item()
    print(f"Best temp: {best_temp}")
    return best_temp


def get_logger(config):
    pl_logger = None
    if config["wandb"]:
        if config["data"]["task_type"] in ["atac_to_gex", "gex_to_atac"]:
            project = "mm_atac"
        elif config["data"]["task_type"] in ["adt_to_gex", "gex_to_adt"]:
            project = "mm_adt"
        else:
            raise NotImplementedError()
        pl_logger = WandbLogger(
            project=project,
            log_model=False,  # type: ignore
            config=config,
            tags=[],
            config_exclude_keys=["wandb"],
        )
        pl_logger.experiment.define_metric(name="test_top1", summary="max")
        pl_logger.experiment.define_metric(name="test_top5", summary="max")
        pl_logger.experiment.define_metric(name="test_top10", summary="max")
    return pl_logger


def get_callbacks(preprocessed_data: dict, model_config: dict, logger=None):
    callbacks = []
    small_train_callback = clip.TargetCallback(
        preprocessed_data["small_train_dataloader"],
        model_config["predict_temperature"],
        "train",
        log_top=[5, 1],
    )
    callbacks.append(small_train_callback)

    if "val_dataloader" in preprocessed_data:
        val_callback = clip.TargetCallback(
            preprocessed_data["val_dataloader"],
            model_config["predict_temperature"],
            "val",
            log_top=[5, 1],
        )
        callbacks.append(val_callback)

    log_preds = logger is not None
    test_callback = clip.TargetCallback(
        preprocessed_data["test_dataloader"],
        model_config["predict_temperature"],
        "test",
        log_top=[10, 5, 1],
        log_preds=log_preds,
    )
    callbacks.append(test_callback)

    if logger is not None:
        learning_rate_monitor = LearningRateMonitor(
            logging_interval="step",
        )
        callbacks.append(learning_rate_monitor)
    return callbacks


def train(config: dict):
    # Load data
    data_config = config["data"]
    dataset = dataloader.load_custom_mm_data(
        data_config["task_type"],
        data_config["train_batches"],
        data_config["test_batches"],
        val_size=data_config["val_size"],
    )
    log.info("Data is loaded")

    # Preprocess data
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, mode="train"
    )
    train_dataloaders = preprocessed_data["train_shuffled_dataloader"]
    model_config = preprocessing.update_model_config(config, preprocessed_data)
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
