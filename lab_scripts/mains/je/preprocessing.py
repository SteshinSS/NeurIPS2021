import logging
import pickle
from collections import Counter

import anndata as ad
import numpy as np
import torch
from lab_scripts.data.integration.processor import (Processor,
                                                    TwoOmicsDataset)
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


def preprocess_one_dataset(
    processor: Processor,
    dataset: ad.AnnData,
):
    X, inverse_transform = processor.transform(dataset)
    result = {
        "X": X,
        "inverse": inverse_transform,
        "batch_idx": get_batch_idx(dataset),
    }
    return result


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
    processor = Processor(mod_config)
    processor.fit(dataset)
    save_processor(processor, mod_config, task_type)
    return processor


def get_correction_dataloaders(
    config: dict,
    dataset: dict,
    first_processor: Processor,
    second_processor: Processor,
    batch_size: int,
):
    result = []  # type: ignore

    correction_batches = config["batch_correct"]
    common_first_dataset = ad.concat([dataset["train_mod1"], dataset["test_mod1"]])
    common_second_dataset = ad.concat([dataset["train_mod2"], dataset["test_mod2"]])
    dataset_batches = common_first_dataset.obs["batch"].astype("string")
    for cor_batch in correction_batches:
        selected_idx = dataset_batches.apply(lambda batch: batch == cor_batch).values
        first_batch = common_first_dataset[selected_idx]
        first = preprocess_one_dataset(first_processor, first_batch)
        second_batch = common_second_dataset[selected_idx]
        second = preprocess_one_dataset(second_processor, second_batch)
        batch_dataset = TwoOmicsDataset(first["X"], second["X"])
        result.append(
            DataLoader(
                batch_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=1,
            )
        )
    return result


def preprocess_data(config: dict, dataset, batch_size, is_train, resources_dir=None):
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
    for key, value in dataset.items():
        if isinstance(value, ad.AnnData):
            dataset[key] = utils.convert_to_dense(value)
    result = {}
    if not is_train:
        first_processor = load_processor(config["mod1"], config["task_type"])
        first_test = preprocess_one_dataset(first_processor, dataset["test_mod1"])
        result["first_features"] = first_test["X"].shape[1]

        second_processor = load_processor(config["mod2"], config["task_type"])
        second_test = preprocess_one_dataset(second_processor, dataset["test_mod2"])
        result["second_features"] = second_test["X"].shape[1]

        test_dataset = TwoOmicsDataset(first_test["X"], second_test["X"])
        result["test_dataloader"] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        result['total_correction_batches'] = len(config['batch_correct'])
        return result

    first_processor = train_processor(
        config["mod1"], dataset["train_mod1"], config["task_type"]
    )
    second_processor = train_processor(
        config["mod2"], dataset["train_mod2"], config["task_type"]
    )

    first_train = preprocess_one_dataset(first_processor, dataset["train_mod1"])
    result["first_features"] = first_train["X"].shape[1]
    second_train = preprocess_one_dataset(second_processor, dataset["train_mod2"])
    result["second_features"] = second_train["X"].shape[1]
    train_dataset = TwoOmicsDataset(
        first_train["X"], second_train["X"], first_train["batch_idx"]
    )
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
    )

    small_idx = np.arange(first_train["X"].shape[0])
    np.random.shuffle(small_idx)
    small_idx = small_idx[:512]
    result["small_train_dataloader"] = DataLoader(
        torch.utils.data.Subset(train_dataset, small_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    first_test = preprocess_one_dataset(first_processor, dataset["test_mod1"])
    second_test = preprocess_one_dataset(second_processor, dataset["test_mod2"])
    test_dataset = TwoOmicsDataset(
        first_test["X"], second_test["X"]
    )
    result["test_dataloader"] = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    result["train_batch_weights"] = calculate_batch_weights(first_train["batch_idx"])

    correction_dataloaders = get_correction_dataloaders(
        config, dataset, first_processor, second_processor, batch_size
    )
    result["correction_dataloaders"] = correction_dataloaders
    result["total_correction_batches"] = len(correction_dataloaders)  # type: ignore
    return result


def update_model_config(model_config: dict, preprocessed_data: dict):
    model_config['first_dim'].insert(0, preprocessed_data['first_features'])
    model_config['second_dim'].insert(0, preprocessed_data['second_features'])
    model_config['common_dim'].insert(0, model_config['first_dim'][-1] + model_config['second_dim'][-1])
    model_config["total_correction_batches"] = preprocessed_data['total_correction_batches']
    return model_config