import logging
import pickle

import anndata as ad
import numpy as np
from collections import Counter
import torch
from lab_scripts.data.integration.processor import (Processor, TwoOmicsDataset)
from torch.utils.data import DataLoader

log = logging.getLogger("mp")
base_config_path = "configs/mp/mp/"
base_checkpoint_path = "checkpoints/mp/mp/"


def get_processor_path(
    mod_config: dict, task_type: str, mod: str, proc_type: str = None
):
    if not proc_type:
        proc_type = ""
    else:
        proc_type = "_" + proc_type

    return mod_config.get(
        "checkpoint_path",
        base_checkpoint_path + task_type + "_" + mod + proc_type + ".ckpt",
    )


def load_processor(config, task_type, mod, proc_type=None):
    processor_checkpoint_path = get_processor_path(config, task_type, mod, proc_type)
    with open(processor_checkpoint_path, "rb") as f:
        processor = pickle.load(f)
    if not proc_type:
        proc_type = ""
    log.info(f"{mod} {proc_type} processor is loaded from {processor_checkpoint_path}")
    return processor


def save_processor(processor, config, task_type, mod, proc_type=None):
    """Saves Processor

    Args:
        processor (Processor): processor
        config (dict): Modality configuration
        task_type (str): One of four
        proc_type (str, optional): Either 'input' or 'target' for non-mirror datasets. Defaults to None.
    """
    processor_checkpoint_path = get_processor_path(config, task_type, mod, proc_type)
    with open(processor_checkpoint_path, "wb") as f:
        pickle.dump(processor, f)
    log.info(f"{mod} processor is saved to {processor_checkpoint_path}")


def get_batch_idx(dataset: ad.AnnData):
    """Returns tensor of batch indices.

    Args:
        dataset (ad.AnnData): Dataset

    Returns:
        torch.Tensor: of batches
    """
    mapping = {
        item: i
        for (i, item) in enumerate(
            dataset.obs["batch"].cat.remove_unused_categories().unique()
        )
    }
    batch_idx = (
        dataset.obs["batch"].cat.remove_unused_categories().apply(lambda x: mapping[x])
    )
    batch_idx = batch_idx.to_numpy()
    batch_idx = torch.tensor(batch_idx)
    return batch_idx


def apply_processor(
    config: dict,
    dataset_train: ad.AnnData,
    dataset_val: ad.AnnData,
    dataset_test: ad.AnnData,
    mod: str,
    task_type: str,
    is_train: bool,
):
    if is_train:
        processor = Processor(config, mod)
        train_X, train_inverse = processor.fit_transform(dataset_train)
        val_X, val_inverse = processor.transform(dataset_val)
        test_X, test_inverse = processor.transform(dataset_test)
        save_processor(processor, config, task_type, mod)
    else:
        processor = load_processor(config, task_type, mod)
        train_X, train_inverse = processor.fit_transform(dataset_train)
        val_X, val_inverse = processor.fit_transform(dataset_val)
        test_X, test_inverse = processor.transform(dataset_test)
    
    return train_X, train_inverse, val_X, val_inverse, test_X, test_inverse


def preprocess_one_dataset(
    mod_config: dict,
    dataset_train: ad.AnnData,
    dataset_val: ad.AnnData,
    dataset_test: ad.AnnData,
    task_type: str,
    is_train: bool,
):
    """Preprocess not mirror dataset

    Args:
        mod_config (dict): Modality data configuration
        dataset_train (ad.AnnData): Train dataset
        dataset_test (ad.AnnData): Test dataset
        task_type (str): One of four types

    Returns:
        tuple: train dict, test dict and number of features
    """
    mod = mod_config["name"]
    train_X, train_inverse, val_X, val_inverse, test_X, test_inverse = apply_processor(
        mod_config,
        dataset_train,
        dataset_val,
        dataset_test,
        mod,
        task_type,
        is_train,
    )

    train_batch_idx = get_batch_idx(dataset_train)
    val_batch_idx = get_batch_idx(dataset_val)
    test_batch_idx = get_batch_idx(dataset_test)

    train = {
        "X": train_X,
        "inverse": train_inverse,
        "batch_idx": train_batch_idx,
    }

    val = {
        'X': val_X,
        'inverse': val_inverse,
        'batch_idx': val_batch_idx
    }

    test = {
        "X": test_X,
        "inverse": test_inverse,
        "batch_idx": test_batch_idx,
    }

    inputs_features = train_X.shape[1]

    return train, val, test, inputs_features


def calculate_batch_weights(batch_idx):
    counter = Counter(batch_idx.numpy())
    total = 0
    for key, value in counter.items():
        total += value
    total_batches = torch.unique(batch_idx).shape[0]
    weights = np.zeros((total_batches))
    for batch in range(total_batches):
        weights[batch] = total / counter[batch]
    weights /= weights.sum()
    total_after_weighting = 0.0
    for batch in range(total_batches):
        total_after_weighting += counter[batch] * weights[batch]
    coef = total / total_after_weighting
    return weights * coef



def preprocess_data(config: dict, dataset, batch_size, is_train):
    """Preprocesses data.

    Args:
        config (dict): Data configuration
        dataset (dict): A dict with train_mod1, train_mod2, test_mod1, test_mod2
        batch_size (int): Batch size

    Returns:
        dict: returns train/test dataloaders, functions to invert preprocessing
        transformation and number of features in each modalities.
    """
    (
        first_train,
        first_val,
        first_test,
        first_input_features,
    ) = preprocess_one_dataset(
        config["mod1"],
        dataset["train_mod1"],
        dataset['val_mod1'],
        dataset["test_mod1"],
        config["task_type"],
        is_train,
    )
    (
        second_train,
        second_val,
        second_test,
        second_input_features,
    ) = preprocess_one_dataset(
        config["mod2"],
        dataset["train_mod2"],
        dataset['val_mod2'],
        dataset["test_mod2"],
        config["task_type"],
        is_train,
    )


    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False

    train_dataset = TwoOmicsDataset(first_train['X'], second_train['X'], first_train['batch_idx'])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=cuda,
        num_workers=1,
        drop_last=True,
    )

    val_dataset = TwoOmicsDataset(first_val['X'], second_val['X'], first_val['batch_idx'])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=cuda,
        num_workers=1,
    )

    small_idx = np.arange(first_train['X'].shape[0])
    np.random.shuffle(small_idx)
    small_idx = small_idx[:512]
    small_train_dataloader = DataLoader(
        torch.utils.data.Subset(train_dataset, small_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataset = TwoOmicsDataset(first_test['X'], second_test['X'], first_test['batch_idx'])
    if cuda:
        test_dataset.to("cuda")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_shuffled_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    result = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        'test_shuffled_dataloader': test_shuffled_dataloader,
        'val_dataloader': val_dataloader,
        "small_dataloader": small_train_dataloader,
        "small_idx": small_idx,
        "first_train_inverse": first_train["inverse"],
        "first_val_inverse": first_val["inverse"],
        "first_test_inverse": first_test["inverse"],
        "first_input_features": first_input_features,
        "second_train_inverse": second_train["inverse"],
        "second_val_inverse": second_val["inverse"],
        "second_test_inverse": second_test["inverse"],
        "second_input_features": second_input_features,
        "train_batch_idx": first_train["batch_idx"],
        'test_batch_idx': first_test['batch_idx'],
        "train_batch_weights": calculate_batch_weights(first_train["batch_idx"]),
    }
    return result
