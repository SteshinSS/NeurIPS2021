import logging

import pytorch_lightning as pl
import torch
from lab_scripts.data import dataloader
from lab_scripts.mains.mp import preprocessing
from lab_scripts.mains.mp import model as mp
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger("mp")


def get_logger(config):
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

def get_callbacks(preprocessed_data: dict, dataset: dict):
    small_idx = preprocessed_data["small_idx"]
    train_callback = mp.TargetCallback(
        preprocessed_data["small_train_dataloader"],
        preprocessed_data["second_train_inverse"],
        dataset["train_mod2"][small_idx],
        prefix="train",
    )
    callbacks = [train_callback]

    if "val_dataloader" in preprocessed_data:
        val_callback = mp.TargetCallback(
            preprocessed_data["val_dataloader"],
            preprocessed_data["second_val_inverse"],
            dataset["val_mod2"],
            prefix="val",
        )
        callbacks.append(val_callback)

    test_callback = mp.TargetCallback(
        preprocessed_data["test_dataloader"],
        preprocessed_data["second_test_inverse"],
        dataset["test_mod2"],
        prefix="test",
    )
    callbacks.append(test_callback)

    learning_rate_monitor = LearningRateMonitor(
        logging_interval="step",
    )
    callbacks.append(learning_rate_monitor)

    early_stopping = EarlyStopping(monitor="test_m", patience=50, mode="min")
    callbacks.append(early_stopping)
    return callbacks


def tune_one_config(config: dict, preprocessed_data: dict, dataset):
    model_config = preprocessing.update_model_config(config, preprocessed_data)
    if model_config["total_correction_batches"] > 0:
        train_dataloaders = [preprocessed_data["train_shuffled_dataloader"]]
        train_dataloaders.extend(preprocessed_data["correction_dataloaders"])
    else:
        train_dataloaders = preprocessed_data["train_shuffled_dataloader"]
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, dataset)

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore

    # Train model
    model = mp.Predictor(model_config)
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


def tune_hp(config: dict):
    data_config = config["data"]
    dataset = dataloader.load_custom_mp_data(
        task_type=data_config["task_type"],
        train_batches=data_config["train_batches"],
        test_batches=data_config["test_batches"],
        val_size=data_config["val_size"],
    )
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, mode="train"
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


