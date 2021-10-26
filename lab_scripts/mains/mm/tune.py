import logging

import pytorch_lightning as pl
import torch
from lab_scripts.data import dataloader
from lab_scripts.mains.mm import common, preprocessing
from lab_scripts.mains.mm.preprocessing import (base_checkpoint_path,
                                                base_config_path)
from lab_scripts.models import clip
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ray import tune

log = logging.getLogger("mm")


def get_logger(config):
    pl_logger = None
    if config["wandb"]:
        pl_logger = WandbLogger(
            project="mm",
            log_model=False,  # type: ignore
            config=config,
            tags=["baseline", "tune"],
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
            preprocessed_data["val_dataloader"],
            model_config["predict_temperature"],
            "val",
            log_top=[0.05, 0.01],
        )
        callbacks.append(val_callback)

    log_preds = logger is not None
    test_callback = clip.TargetCallback(
        preprocessed_data["test_dataloader"],
        model_config["predict_temperature"],
        "test",
        log_top=[0.1, 0.05, 0.01],
        log_preds=log_preds,
    )
    callbacks.append(test_callback)

    if logger is not None:
        learning_rate_monitor = LearningRateMonitor(
            logging_interval="step",
        )
        callbacks.append(learning_rate_monitor)
    
    callbacks.append(EarlyStopping(monitor='test_top0.01', patience=30, mode='max'))
    return callbacks


def tune_one_config(config: dict, preprocessed_data):
    # Solution of strange bug.
    # See https://github.com/pytorch/pytorch/issues/57794#issuecomment-834976384
    torch.cuda.set_device(0)

    data_config = config["data"]
    model_config = config["model"]

    model_config[''] = None

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


def tune_hp(config: dict):
    data_config = config["data"]
    dataset = dataloader.load_custom_mm_data(
        data_config["task_type"],
        data_config["train_batches"],
        data_config["test_batches"],
        filter_genes_params=(data_config["gene_fraction"], "data/genes.csv"),
        val_size=data_config["val_size"],
    )
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=True
    )
    log.info("Data is preprocessed")

    config["model"].update(model_search_space)
    preprocessed_data["dataset"] = dataset
    tune.run(
        tune.with_parameters(tune_one_config, preprocessed_data=preprocessed_data),
        metric="test",
        mode="test_top0.01",
        config=config,
        resources_per_trial={"gpu": 1, "cpu": 16},
        num_samples=-1,
        local_dir="tune",
    )


model_search_space = {}  # type: ignore
