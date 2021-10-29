import logging

import pytorch_lightning as pl
import torch
from lab_scripts.data import dataloader
from lab_scripts.mains.je import common, preprocessing
from lab_scripts.mains.je.preprocessing import (base_checkpoint_path,
                                                base_config_path)
from lab_scripts.models import je as je_model
from lab_scripts.utils import utils
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from ray import tune

log = logging.getLogger("mm")


def get_logger(config):
    pl_logger = WandbLogger(
        project="je",
        log_model=False,  # type: ignore
        config=config,
        tags=["baseline", "tune"],
        config_exclude_keys=["wandb"],
    )
    # pl_logger.experiment.define_metric(name="test_top0.05", summary="max")
    return pl_logger


def get_callbacks(preprocessed_data: dict, model_config: dict, logger=None):
    callbacks = []

    val_callback = je_model.TargetCallback(
        preprocessed_data["test_solution"],
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


def tune_one_config(config, good_config: dict, preprocessed_data):
    utils.change_directory_to_repo()
    good_config.update(config)
    config = good_config

    # Solution of strange bug.
    # See https://github.com/pytorch/pytorch/issues/57794#issuecomment-834976384
    torch.cuda.set_device(0)

    data_config = config["data"]
    model_config = config["model"]

    train_dataloaders = [preprocessed_data["train_shuffled_dataloader"]]
    train_dataloaders.extend(preprocessed_data["correction_dataloaders"])
    model_config = common.update_model_config(model_config, preprocessed_data)
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, model_config, pl_logger)

    # Train model
    model = je_model.JEAutoencoder(model_config)
    if pl_logger:
        pl_logger.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config["gradient_clip"],
    )
    trainer.fit(model, train_dataloaders=train_dataloaders)
    tune.report(loss=0.0)


def tune_hp(config: dict):
    data_config = config["data"]
    dataset = dataloader.load_custom_je_data(
        data_config["task_type"],
        data_config["train_batches"],
        data_config["test_batches"],
        val_size=data_config["val_size"],
    )
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=True
    )

    log.info("Data is preprocessed")
    tune.run(
        tune.with_parameters(
            tune_one_config, good_config=config, preprocessed_data=preprocessed_data
        ),
        metric="loss",
        mode="min",
        config=model_search_space,
        resources_per_trial={"gpu": 1, "cpu": 16},
        num_samples=-1,
    )


model_search_space = {
    'lr': tune.choice([0.1, 0.01, 0.001]),
}  # type: ignore
