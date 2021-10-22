import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import early_stopping
import torch
from lab_scripts.data import dataloader
from lab_scripts.mains.mp import preprocessing, common
from lab_scripts.mains.mp.preprocessing import (
    base_checkpoint_path, base_config_path)
from lab_scripts.models import mp
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from ray import tune

log = logging.getLogger("mp")


def get_logger(config):
    pl_logger = WandbLogger(
        project="mp",
        log_model=False,  # type: ignore
        config=config,
        tags=["baseline", 'tune'],
        config_exclude_keys=["wandb"],
    )
    pl_logger.experiment.define_metric(name="train_m", summary="min")
    pl_logger.experiment.define_metric(name="test_m", summary="min")
    return pl_logger


def get_callbacks(preprocessed_data: dict, dataset: dict):
    small_idx = preprocessed_data['small_idx']
    train_val_callback = mp.TargetCallback(
        preprocessed_data['small_dataloader'],
        preprocessed_data['second_train_inverse'],
        dataset['train_mod2'][small_idx],
        prefix='train'
    )

    test_val_callback = mp.TargetCallback(
        preprocessed_data['test_dataloader'],
        preprocessed_data['second_test_inverse'],
        dataset['test_mod2'],
        prefix='test'
    )

    learning_rate_monitor = LearningRateMonitor(
        logging_interval="step",
    )

    early_stopping = EarlyStopping(monitor='train_m', patience=30, mode='min')
    callbacks = [train_val_callback, test_val_callback, early_stopping, learning_rate_monitor]
    return callbacks


def tune_one_config(config: dict, preprocessed_data: dict):
    # Solution of strange bug.
    # See https://github.com/pytorch/pytorch/issues/57794#issuecomment-834976384
    torch.cuda.set_device(0)
    # Load data
    model_config = config["model"]
    model_config['dims'] = dims[model_config['_dim']]
    model_config['bn'] = bns[model_config['_bns']]
    model_config = common.update_model_config(model_config, preprocessed_data)
    train_dataloader = preprocessed_data["train_dataloader"]
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, preprocessed_data['dataset'])
    if not pl_logger:
        callbacks.pop()

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore

    # Train model
    model = mp.Predictor(model_config)
    if pl_logger:
        pl_logger.watch(model)

    trainer = pl.Trainer(
        gpus=use_gpu,
        max_epochs=2000,
        logger=pl_logger,
        callbacks=callbacks,
        deterministic=True,
        checkpoint_callback=False,
        gradient_clip_val=model_config['gradient_clip']
    )
    trainer.fit(model, train_dataloaders=train_dataloader)


def tune_hp(config: dict):
    data_config = config["data"]
    dataset = dataloader.load_data(data_config["dataset_name"])
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=True
    )
    log.info("Data is preprocessed")

    config["model"].update(model_search_space)
    preprocessed_data['dataset'] = dataset
    tune.run(
        tune.with_parameters(tune_one_config, preprocessed_data=preprocessed_data),
        metric="test_m",
        mode="min",
        config=config,
        resources_per_trial={"gpu": 1, "cpu": 16},
        num_samples=-1,
        local_dir="tune",
    )


dims = [
    [5000, 3000, 1000, 750, 750, 500, 300],
    [4000, 2000, 1000, 750, 750, 500, 300],
    [3000, 2000, 1000, 750, 500],
    [1000, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    [500, 300, 200],
    [1000, 500, 1000, 500, 1000, 500],
    [1000, 700, 500, 300, 200, 150],
    [2000, 700, 500, 300, 200, 150],
    [1000, 750, 750, 300, 200, 150],
    [1000, 750, 750, 500, 200, 150],
    [1000, 500, 500, 300, 200, 150],
    [1000, 500, 300, 200, 150],
    [1000, 500, 300, 300, 200, 150],
]

bns = [
    [],
    [],
    [],
    [0],
    [1],
    [2],
    [3],
    [0, 1],
    [0, 1, 2],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4, 5]
]


model_search_space = {
    '_dim': tune.choice(range(len(dims))),
    '_bns': tune.choice(range(len(bns))),
    'lr': tune.choice([1e-3, 5e-4, 3e-4, 1e-4]),
}