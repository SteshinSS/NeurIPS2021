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
        tags=["tune"],
        config_exclude_keys=["wandb"],
    )
    pl_logger.experiment.define_metric(name="train_m", summary="min")
    pl_logger.experiment.define_metric(name="test_m", summary="min")
    return pl_logger


def get_callbacks(preprocessed_data: dict, dataset: dict):
    small_idx = preprocessed_data["small_idx"]
    train_val_callback = mp.TargetCallback(
        preprocessed_data["small_dataloader"],
        preprocessed_data["second_train_inverse"],
        dataset["train_mod2"][small_idx],
        prefix="train",
    )

    test_val_callback = mp.TargetCallback(
        preprocessed_data["test_dataloader"],
        preprocessed_data["second_test_inverse"],
        dataset["test_mod2"],
        prefix="test",
    )

    learning_rate_monitor = LearningRateMonitor(
        logging_interval="step",
    )

    early_stopping = EarlyStopping(monitor="test_m", patience=50, mode="min")
    callbacks = [
        train_val_callback,
        test_val_callback,
        early_stopping,
        learning_rate_monitor,
    ]
    return callbacks


def tune_one_config(config: dict, preprocessed_data: dict):
    # Solution of strange bug.
    # See https://github.com/pytorch/pytorch/issues/57794#issuecomment-834976384
    torch.cuda.set_device(0)
    # Load data
    model_config = config["model"]
    model_config["feature_extractor_dims"] = fe_dims[model_config["fe_dims"]]
    model_config["regression_dims"] = re_dims[model_config["re_dims"]]
    model_config["fe_dropout"] = fe_drop[model_config["fe_drop"]]
    model_config["l2_lambda"] = l2[model_config["l2"]]
    model_config["mmd_lambda"] = mmd[model_config["mmd"]]
    model_config["l2_loss_lambda"] = l2_loss[model_config["l2_loss"]]
    model_config["coral_lambda"] = coral[model_config["coral"]]

    model_config = common.update_model_config(model_config, preprocessed_data)
    train_dataloader = preprocessed_data["train_dataloader"]
    log.info("Data is preprocessed")

    # Configure training
    pl_logger = get_logger(config)
    callbacks = get_callbacks(preprocessed_data, preprocessed_data["dataset"])
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
        gradient_clip_val=model_config["gradient_clip"],
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
    preprocessed_data["dataset"] = dataset
    tune.run(
        tune.with_parameters(tune_one_config, preprocessed_data=preprocessed_data),
        metric="test_m",
        mode="min",
        config=config,
        resources_per_trial={"gpu": 1, "cpu": 16},
        num_samples=-1,
        local_dir="tune",
    )


fe_dims = [
    [1000, 700, 500, 300],
    [1000, 700, 500, 300],
    [1000, 750, 600, 400],
    [1000, 700, 500, 300, 300],
    [1000, 700, 500, 300, 300, 300],
]

re_dims = [[200, 200, 150], [200, 150], [200, 200, 200]]

fe_drop = [
    [],
    [],
    [],
    [0],
    [1],
]

l2 = [0.0, 0.0, 0.0, 0.0, 0.0001, 0.0005]

mmd = [0.0, 0.0, 1.0, 10.0]

l2_loss = [0.0, 0.0, 0.001, 0.005]

coral = [0.0, 1.0, 5.0, 10.0]

