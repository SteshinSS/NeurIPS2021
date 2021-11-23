import logging
import pickle
from collections import Counter

import anndata as ad
import numpy as np
import torch
from lab_scripts.data.integration.processor import (OneOmicDataset,
                                                    TwoOmicsDataset,
                                                    get_processor)
from torch.utils.data import DataLoader

log = logging.getLogger("mp")
base_config_path = "configs/mp/"
base_checkpoint_path = "checkpoints/mp/"


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
    test_dataset = OneOmicDataset(first_X)
    result["test_dataloader"] = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )
    second_processor = load_processor(config["mod2"], config["task_type"])
    result["second_test_inverse"] = second_processor.get_inverse_transform(
        dataset["test_mod1"]
    )
    result["second_features"] = second_processor.features
    result["train_batch_weights"] = np.ones((len(config['train_batches'])))
    if config['task_type'] == 'atac_to_gex':
        result["train_batch_weights"] = np.ones((len(config['train_batches']) - 1))

    if 'prediction_weights' in config:
        result['prediction_weights'] = [0.0] * second_processor.features
    return result


def add_train_dataloader(
    result, first_processor, second_processor, dataset, batch_size, mixup
):
    first_X = first_processor.transform(dataset["train_mod1"])
    result["first_features"] = first_X.shape[1]
    second_X = second_processor.transform(dataset["train_mod2"])
    result["second_train_inverse"] = second_processor.get_inverse_transform(
        dataset["train_mod2"]
    )
    result["second_features"] = second_X.shape[1]
    batch_idx = get_batch_idx(dataset["train_mod1"])
    train_dataset = TwoOmicsDataset(first_X, second_X, batch_idx, mixup=mixup)
    result["train_shuffled_dataloader"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    result["train_batch_weights"] = calculate_batch_weights(batch_idx)

    train_unshuffled_dataset = OneOmicDataset(first_X)
    result["train_unshuffled_dataloader"] = DataLoader(
        train_unshuffled_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    small_idx = np.arange(first_X.shape[0])
    np.random.shuffle(small_idx)
    small_idx = small_idx[:512]
    small_dataset = first_X[small_idx]
    result["small_train_dataloader"] = DataLoader(
        OneOmicDataset(small_dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    result["small_idx"] = small_idx
    result['small_train_inverse'] = second_processor.get_inverse_transform(dataset['train_mod2'][small_idx])


def add_correction_dataloaders(result, dataset, first_processor, config):
    correction_dataloaders = []
    correction_batches = config["batch_correct"]
    common_dataset = ad.concat([dataset["train_mod1"], dataset["test_mod1"]])
    dataset_batches = common_dataset.obs["batch"].astype("string")
    for cor_batch in correction_batches:
        selected_idx = dataset_batches.apply(lambda batch: batch == cor_batch).values
        one_batch_dataset = common_dataset[selected_idx]
        one_batch_X = first_processor.transform(one_batch_dataset)
        preprocessed_dataset = OneOmicDataset(one_batch_X)
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


def add_test_dataloader(result, first_processor, second_processor, dataset, batch_size):
    first_X = first_processor.transform(dataset["test_mod1"])
    result["second_test_inverse"] = second_processor.get_inverse_transform(
        dataset["test_mod2"]
    )
    test_dataset = OneOmicDataset(first_X)
    result["test_dataloader"] = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )


def add_val_dataloader(result, first_processor, second_processor, dataset, batch_size):
    first_X = first_processor.transform(dataset["val_mod1"])
    result["second_val_inverse"] = second_processor.get_inverse_transform(
        dataset["val_mod2"]
    )
    val_dataset = OneOmicDataset(first_X)
    result["val_dataloader"] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

def add_prediction_weights(result, config):
    prediction_weight_path = config['prediction_weights']
    weights = np.loadtxt(prediction_weight_path, delimiter=",")
    weights = np.exp(weights * config['prediction_weight_lambda'])
    result['prediction_weights'] = weights


def preprocess_train_data(config, dataset):
    result = {}
    first_processor = train_processor(
        config["mod1"], dataset["train_mod1"], config["task_type"]
    )
    second_processor = train_processor(
        config["mod2"], dataset["train_mod2"], config["task_type"]
    )
    add_train_dataloader(
        result, first_processor, second_processor, dataset, config["batch_size"], config['mixup']
    )
    add_test_dataloader(
        result, first_processor, second_processor, dataset, config["batch_size"]
    )
    add_correction_dataloaders(result, dataset, first_processor, config)
    if "val_mod1" in dataset:
        add_val_dataloader(
            result, first_processor, second_processor, dataset, config["batch_size"]
        )
    if 'prediction_weights' in config:
        add_prediction_weights(result, config)
    return result


def preprocess_data(config: dict, dataset, mode=None, resources_dir=None):
    if resources_dir is not None:
        global base_config_path
        base_config_path = resources_dir + base_config_path
        global base_checkpoint_path
        base_checkpoint_path = resources_dir + base_checkpoint_path

    if mode == "test":
        return preprocess_test_data(config, dataset)
    elif mode == "tune":
        return None
    elif mode == "train":
        return preprocess_train_data(config, dataset)
    else:
        raise NotImplementedError()


def update_model_config(config: dict, preprocessed_data: dict):
    model_config = config["model"]
    model_config["feature_extractor_dims"].insert(
        0, preprocessed_data["first_features"]
    )
    if model_config['concat_input']:
        regression_input_dim = model_config['feature_extractor_dims'][-1] + preprocessed_data['first_features']
    else:
        regression_input_dim = model_config['feature_extractor_dims'][-1]
    model_config["regression_dims"].insert(
        0, regression_input_dim
    )
    model_config["regression_dims"].append(preprocessed_data["second_features"])
    model_config["total_correction_batches"] = len(config["data"]["batch_correct"])
    model_config["batch_weights"] = preprocessed_data["train_batch_weights"]
    if 'prediction_weights' in preprocessed_data:
        model_config['prediction_weights'] = preprocessed_data['prediction_weights']
    model_config['task_type'] = config['data']['task_type']
    return model_config
