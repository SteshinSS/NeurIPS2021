import logging

import pytorch_lightning as pl
import torch
import numpy as np
from lab_scripts.data import dataloader
from lab_scripts.mains.je import preprocessing
from lab_scripts.mains.je import model as je_model
from lab_scripts.mains.je.callback import TargetCallback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger("mm")


def get_logger(config):
    pl_logger = WandbLogger(
        project="je",
        log_model=False,  # type: ignore
        config=config,
        tags=[config['data']['task_type'], "tune"],
        config_exclude_keys=["wandb"],
    )
    # pl_logger.experiment.define_metric(name="test_top0.05", summary="max")
    return pl_logger


def get_callbacks(preprocessed_data: dict, dataset, model_config: dict, logger=None):
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


def tune_one_config(config, preprocessed_data, dataset):
    data_config = config["data"]
    model_config = config["model"]

    train_dataloaders = [preprocessed_data["train_shuffled_dataloader"]]
    train_dataloaders.extend(preprocessed_data["correction_dataloaders"])
    model_config = preprocessing.update_model_config(model_config, preprocessed_data)
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, dataset, model_config, pl_logger)

    # Train model
    model = je_model.JEAutoencoder(model_config)
    if pl_logger:
        pl_logger.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=51,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config["gradient_clip"] if not model_config['use_critic'] else 0.0,
    )
    trainer.fit(model, train_dataloaders=train_dataloaders)


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
    for i in range(10):
        new_config = config.copy()
        new_config['model'] = update_config(new_config['model'], i)
        tune_one_config(new_config, preprocessed_data, dataset)



def update_config(config: dict, i):
    if i == 0:
        return config
    if i == 1:
        config['critic_iterations'] = 3 
    return config

