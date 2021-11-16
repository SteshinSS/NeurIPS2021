import logging
import pickle
from collections import Counter

import anndata as ad
import numpy as np
import torch
from lab_scripts.data.integration.processor import (get_processor, Processor, TwoOmicsDataset)
from lab_scripts.utils import utils
from torch.utils.data import DataLoader

log = logging.getLogger("je")
base_config_path = "configs/je/"
base_checkpoint_path = "checkpoints/je/"


def get_processor_path(mod_config: dict, task_type: str):
    mod = mod_config["name"]
    return mod_config.get(
        "checkpoint_path",
        base_checkpoint_path + task_type + "_" + mod + ".ckpt",
    )


def load_processor(config, task_type):
    mod = config["name"]
    processor_checkpoint_path = get_processor_path(config, task_type)
    with open(processor_checkpoint_path, "rb") as f:
        processor = pickle.load(f)
    log.info(f"{mod} processor is loaded from {processor_checkpoint_path}")
    return processor


def save_processor(processor, config, task_type):
    """Saves Processor

    Args:
        processor (Processor): processor
        config (dict): Modality configuration
        task_type (str): One of four
        proc_type (str, optional): Either 'input' or 'target' for non-mirror datasets. Defaults to None.
    """
    mod = config["name"]
    processor_checkpoint_path = get_processor_path(config, task_type)
    with open(processor_checkpoint_path, "wb") as f:
        pickle.dump(processor, f)
    log.info(f"{mod} processor is saved to {processor_checkpoint_path}")



def train_processor(mod_config: dict, dataset, task_type: str):
    processor = get_processor(mod_config)
    processor.fit(dataset)
    save_processor(processor, mod_config, task_type)
    return processor


def add_correction_dataloaders(result, dataset, first_processor, second_processor, config):
    correction_dataloaders = []
    correction_batches = config["batch_correct"]
    first = dataset['train_mod1']
    second = dataset['train_mod2']
    dataset_batches = first.obs["batch"].astype("string")
    for cor_batch in correction_batches:
        selected_idx = dataset_batches.apply(lambda batch: batch == cor_batch).values
        first_one_batch = first[selected_idx]
        first_X_one_batch = first_processor.transform(first_one_batch)
        second_one_batch = second[selected_idx]
        second_X_one_batch = second_processor.transform(second_one_batch)
        preprocessed_dataset = TwoOmicsDataset(first_X_one_batch, second_X_one_batch)
        correction_dataloaders.append(
            DataLoader(
                preprocessed_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=1,
            )
        )
    result["correction_dataloaders"] = correction_dataloaders
    return result


def add_train_dataloaders(result, dataset, first_processor, second_processor, batch_size, mixup):
    first_X = first_processor.transform(dataset["train_mod1"])
    result["first_features"] = first_X.shape[1]
    second_X = second_processor.transform(dataset["train_mod2"])
    result["second_features"] = second_X.shape[1]
    train_dataset = TwoOmicsDataset(first_X, second_X, batch_idx=None, mixup=mixup)

    result["train_shuffled_dataloader"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )

    result["train_unshuffled_dataloader"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    small_idx = np.arange(first_X.shape[0])
    np.random.shuffle(small_idx)
    small_idx = small_idx[:512]
    small_dataset = TwoOmicsDataset(first_X[small_idx], second_X[small_idx])
    result["small_train_dataloader"] = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return result




def preprocess_data(config: dict, dataset, resources_dir=None):
    """Preprocesses data.

    Args:
        config (dict): Data configuration
        dataset (dict): A dict with train_mod1, train_mod2, test_mod1, test_mod2
        batch_size (int): Batch size

    Returns:
        dict: returns train/test dataloaders, functions to invert preprocessing
        transformation and number of features in each modalities.
    """
    if resources_dir is not None:
        global base_config_path
        base_config_path = resources_dir + base_config_path
        global base_checkpoint_path
        base_checkpoint_path = resources_dir + base_checkpoint_path
    first_processor = train_processor(
        config["mod1"], dataset["train_mod1"], config["task_type"]
    )
    second_processor = train_processor(
        config["mod2"], dataset["train_mod2"], config["task_type"]
    )

    result = {}  # type: ignore
    result = add_train_dataloaders(result, dataset, first_processor, second_processor, config['batch_size'], config['mixup'])
    result = add_correction_dataloaders(result, dataset, first_processor, second_processor, config)
    return result


def update_model_config(config: dict, preprocessed_data: dict):
    model_config = config['model']
    model_config['first_dim'].insert(0, preprocessed_data['first_features'])
    model_config['second_dim'].insert(0, preprocessed_data['second_features'])
    model_config['common_dim'].insert(0, model_config['first_dim'][-1] + model_config['second_dim'][-1])
    model_config["total_correction_batches"] = len(config["data"]["batch_correct"])
    return model_config