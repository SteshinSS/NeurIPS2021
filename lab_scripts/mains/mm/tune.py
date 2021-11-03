import logging

import pytorch_lightning as pl
from lab_scripts.data import dataloader
from lab_scripts.mains.mm import preprocessing, clip
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger("mm")
from copy import deepcopy


def get_logger(config):
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
        tags=[config["data"]["task_type"]],
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
    callbacks.append(EarlyStopping(monitor="test_top5", patience=40, mode="max"))
    return callbacks


def tune_one_config(config, preprocessed_data):
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
        max_epochs=1,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config["gradient_clip"],
    )
    trainer.fit(model, train_dataloaders=train_dataloaders)
    pl_logger.experiment.finish(quiet=True)


def tune_hp(config: dict):
    data_config = config["data"]
    dataset = dataloader.load_custom_mm_data(
        data_config["task_type"],
        data_config["train_batches"],
        data_config["test_batches"],
        val_size=data_config["val_size"],
    )
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, mode="train"
    )
    log.info("Data is preprocessed")
    for i in range(1, 10):
        print(i)
        new_config = deepcopy(config)
        new_config = update_config(new_config, i)
        tune_one_config(new_config, preprocessed_data)


def update_config(config, i):
    model_config = config['model']
    if i == 1:
        model_config['first_batchnorm'] = [2]
        model_config['second_batchnorm'] = [2]
    elif i == 2:
        model_config['first_batchnorm'] = [2, 4]
        model_config['second_batchnorm'] = [2, 4]
    elif i == 3:
        model_config['latent_dim'] = 50
    elif i == 4:
        model_config['latent_dim'] = 75
    elif i == 5:
        model_config['latent_dim'] = 100
    elif i == 6:
        model_config['activation'] = 'relu'
    elif i == 7:
        model_config['activation'] = 'selu'
    elif i == 8:
        model_config['train_temperature'] = 1.0
    elif i == 9:
        model_config['train_temperature'] = 2.0
    elif i == 10:
        model_config['train_temperature'] = 4.0
    elif i == 11:
        model_config['train_temperature'] = 5.0
    elif i == 12:
        model_config['l2_lambda'] = 0.01
    elif i == 13:
        model_config['first_dim'] = [5000, 3000, 1000, 500, 300]
        model_config['second_dim'] = [5000, 3000, 1000, 500, 300]
    elif i == 14:
        model_config['first_dim'] = [5000, 1000, 500, 300]
        model_config['second_dim'] = [5000, 1000, 500, 300]

    return config
