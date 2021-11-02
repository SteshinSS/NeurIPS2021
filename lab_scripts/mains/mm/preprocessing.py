import logging
import pickle
from collections import Counter

import anndata as ad
import numpy as np
import torch
from lab_scripts.data.integration.processor import TwoOmicsDataset, get_processor
from torch.utils.data import DataLoader

log = logging.getLogger("mm")
base_config_path = "configs/mm/"
base_checkpoint_path = "checkpoints/mm/"


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


def get_batch_idx(dataset: ad.AnnData):
    """Returns tensor of batch indices.

    Args:
        dataset (ad.AnnData): Dataset

    Returns:
        torch.Tensor: of batches
    """
    if "batch" not in dataset.obs:
        return None
    dataset_batches = dataset.obs["batch"].astype("string")
    mapping = {item: i for (i, item) in enumerate(dataset_batches.unique())}
    batch_idx = dataset_batches.apply(lambda x: mapping[x])
    batch_idx = batch_idx.to_numpy()
    batch_idx = torch.tensor(batch_idx)
    return batch_idx


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


def train_processor(mod_config: dict, dataset, task_type: str):
    processor = get_processor(mod_config)
    processor.fit(dataset)
    save_processor(processor, mod_config, task_type)
    return processor


def preprocess_test_data(config, dataset):
    result = {}
    first_processor = load_processor(config["mod1"], config["task_type"])
    first_X = first_processor.transform(dataset["test_mod1"])
    result["first_features"] = first_X.shape[1]
    second_processor = load_processor(config["mod2"], config["task_type"])
    second_X = second_processor.transform(dataset["test_mod2"])
    result["second_features"] = second_X.shape[1]
    batch_idx = get_batch_idx(dataset["test_mod1"])
    dataset = TwoOmicsDataset(first_X, second_X, batch_idx)
    result["test_dataloader"] = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False
    )
    result["train_batch_weights"] = np.ones((len(config["train_batches"])))
    return result


def add_train_dataloader(
    result, first_processor, second_processor, dataset, batch_size
):
    first_X = first_processor.transform(dataset["train_mod1"])
    result["first_features"] = first_X.shape[1]
    second_X = second_processor.transform(dataset["train_mod2"])
    result["second_features"] = second_X.shape[1]
    batch_idx = get_batch_idx(dataset["train_mod1"])
    train_dataset = TwoOmicsDataset(first_X, second_X, batch_idx)
    result["train_shuffled_dataloader"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    result["train_batch_weights"] = calculate_batch_weights(batch_idx)

    result["train_unshuffled_dataloader"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    small_idx = np.arange(first_X.shape[0])
    np.random.shuffle(small_idx)
    small_idx = small_idx[:512]
    result["small_train_dataloader"] = DataLoader(
        torch.utils.data.Subset(train_dataset, small_idx),
        batch_size=batch_size,
        shuffle=False,
    )
    result["small_idx"] = small_idx


def add_test_dataloader(result, first_processor, second_processor, dataset, batch_size):
    first_X = first_processor.transform(dataset["test_mod1"])
    second_X = second_processor.transform(dataset["test_mod2"])
    batch_idx = get_batch_idx(dataset["test_mod1"])
    test_dataset = TwoOmicsDataset(first_X, second_X, batch_idx)
    result["test_dataloader"] = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )


def add_val_dataloader(result, first_processor, second_processor, dataset, batch_size):
    first_X = first_processor.transform(dataset["val_mod1"])
    second_X = second_processor.transform(dataset["val_mod2"])
    batch_idx = get_batch_idx(dataset["val_mod1"])
    val_dataset = TwoOmicsDataset(first_X, second_X, batch_idx)
    result["val_dataloader"] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )


def preprocess_train_data(config, dataset):
    result = {}

    first_processor = train_processor(
        config["mod1"], dataset["train_mod1"], config["task_type"]
    )
    second_processor = train_processor(
        config["mod2"], dataset["train_mod2"], config["task_type"]
    )

    train_sorted_idx = dataset["train_sol"].uns["pairing_ix"].astype(np.int32)
    dataset["train_mod2"] = dataset["train_mod2"][train_sorted_idx]
    test_sorted_idx = dataset["test_sol"].uns["pairing_ix"].astype(np.int32)
    dataset["test_mod2"] = dataset["test_mod2"][test_sorted_idx]

    add_train_dataloader(
        result, first_processor, second_processor, dataset, config["batch_size"]
    )
    add_test_dataloader(
        result, first_processor, second_processor, dataset, config["batch_size"]
    )
    if "val_mod1" in dataset:
        add_val_dataloader(
            result, first_processor, second_processor, dataset, config["batch_size"]
        )
    return result


def preprocess_data(config: dict, dataset, mode, resources_dir=None):
    if resources_dir is not None:
        global base_config_path
        base_config_path = resources_dir + base_config_path
        global base_checkpoint_path
        base_checkpoint_path = resources_dir + base_checkpoint_path

    if mode == "train":
        return preprocess_train_data(config, dataset)
    elif mode == "tune":
        pass
    elif mode == "test":
        return preprocess_test_data(config, dataset)
    else:
        raise NotImplementedError()


def update_model_config(config: dict, preprocessed_data: dict):
    model_config = config["model"]
    model_config["first_dim"].insert(0, preprocessed_data["first_features"])
    model_config["second_dim"].insert(0, preprocessed_data["second_features"])
    return model_config
