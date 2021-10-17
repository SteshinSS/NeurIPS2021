import logging

import pytorch_lightning as pl
import torch
from lab_scripts.data import dataloader
from lab_scripts.mains.x_autoencoder import preprocessing
from lab_scripts.mains.x_autoencoder.preprocessing import (
    base_checkpoint_path, base_config_path)
from lab_scripts.models import x_autoencoder
from lab_scripts.utils import utils
from pytorch_lightning.loggers import WandbLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

log = logging.getLogger("x_autoencoder")


def tune_one_config(config, preprocessed_data=None):
    utils.change_directory_to_repo()
    data_config = config["data"]
    model_config = config["model"]
    train_dataloader = preprocessed_data["train_dataloader"]
    test_dataloader = preprocessed_data["test_dataloader"]
    first_inverse = preprocessed_data["first_test_inverse"]
    second_inverse = preprocessed_data["second_test_inverse"]
    first_true_target = preprocessed_data["test_mod1"]
    second_true_target = preprocessed_data["test_mod2"]

    # Configure training
    validation_callback = x_autoencoder.TargetCallback(
        test_dataloader=test_dataloader,
        first_inverse=first_inverse,
        first_true_target=first_true_target,
        second_inverse=second_inverse,
        second_true_target=second_true_target,
    )

    pl_logger = WandbLogger(
        project="nips2021",
        log_model=False,  # type: ignore
        config=config,
        tags=["tune", "x_autoencoder"],
        config_exclude_keys=["wandb"],
    )

    pl_logger.experiment.define_metric(
        name="true_1_to_2",
        summary="min"
    )
    pl_logger.experiment.define_metric(
        name="true_2_to_1",
        summary="min"
    )

    tune_callback = TuneReportCallback(
        ["loss", "true_1_to_2", "true_2_to_1", "11", "12", "21", "22", "mmd", 'critic', 'train_critic', 'sim'],
        on="epoch_end",
    )

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        log.warning("GPU is not detected.")
    use_gpu = int(use_gpu)  # type: ignore

    # Train model
    model = x_autoencoder.X_autoencoder(model_config)

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=use_gpu,
        logger=pl_logger,
        callbacks=[
            validation_callback,
            tune_callback,
        ],
        deterministic=True,
        checkpoint_callback=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)


def tune_hp(config: dict):
    data_config = config["data"]
    dataset = dataloader.load_data(data_config["dataset_name"])
    model_config = config["model"]
    preprocessed_data = preprocessing.preprocess_data(
        data_config, dataset, model_config["batch_size"], is_train=True
    )
    preprocessed_data["test_mod1"] = dataset["test_mod1"]
    preprocessed_data["test_mod2"] = dataset["test_mod2"]
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
    model_config['total_batches'] = torch.unique(preprocessed_data['train_batch_idx']).shape[0]
    log.info("Data is preprocessed")

    config["model"].update(model_search_space)
    tune.run(
        tune.with_parameters(tune_one_config, preprocessed_data=preprocessed_data),
        metric="true_1_to_2",
        mode="min",
        config=config,
        resources_per_trial={"gpu": 1, "cpu": 16},
        num_samples=-1,
        local_dir="tune",
    )


model_search_space = {
    'critic_lr': tune.grid_search([0.001, 0.003, 0.005]),
    'critic_iterations': tune.grid_search([1, 3, 5, 7, 10]),
    
}
