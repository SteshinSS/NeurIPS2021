import logging
import pickle

import anndata as ad
import torch
from lab_scripts.data.integration.processor import (
    FourOmicsDataset,
    Processor,
    TwoOmicsDataset,
)
from torch.utils.data import DataLoader

log = logging.getLogger("x_autoencoder")
base_config_path = "configs/mp/x_autoencoder/"
base_checkpoint_path = "checkpoints/mp/x_autoencoder/"


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
    with open(processor_checkpoint_path, 'rb') as f:
        processor = pickle.load(f)
    if not proc_type:
        proc_type = ''
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


def apply_processor(config: dict, dataset_train: ad.AnnData, dataset_test: ad.AnnData, mod: str, task_type: str, is_train: bool, proc_type: str = None):
    if is_train:
        processor = Processor(config, mod)
        train_X, train_inverse = processor.fit_transform(dataset_train)
        test_X, test_inverse = processor.transform(dataset_test)
        save_processor(processor, config, task_type, mod, proc_type)
        return train_X, train_inverse, test_X, test_inverse
    else:
        processor = load_processor(config, task_type, mod, proc_type)
        train_X, train_inverse = processor.fit_transform(dataset_train)
        test_X, test_inverse = processor.transform(dataset_test)
        return train_X, train_inverse, test_X, test_inverse



def preprocess_mirror_dataset(
    mod_config: dict,
    dataset_train: ad.AnnData,
    dataset_test: ad.AnnData,
    task_type: str,
    is_train: bool,
):
    """Processes a mirror dataset

    Args:
        mod_config (dict): Modality data configuration
        dataset_train (ad.AnnData): Train dataset
        dataset_test (ad.AnnData): Test dataset
        task_type (str): One of four

    Returns:
        tuple: train dict, test dict and number of features
    """
    mod = mod_config['name']
    train_X, train_inverse, test_X, test_inverse = apply_processor(mod_config, dataset_train, dataset_test, mod, task_type, is_train)

    train_batch_idx = get_batch_idx(dataset_train)
    test_batch_idx = get_batch_idx(dataset_test)

    train = {"X": train_X, "inverse": train_inverse, "batch_idx": train_batch_idx}
    test = {"X": test_X, "inverse": test_inverse, "batch_idx": test_batch_idx}
    n_features = train_X.shape[1]

    return train, test, n_features, n_features


def preprocess_not_mirror_dataset(
    mod_config: dict,
    dataset_train: ad.AnnData,
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
    mod = mod_config['name']
    input_train_X, _, input_test_X, _ = apply_processor(mod_config["inputs"], dataset_train, dataset_test, mod, task_type, is_train, "input")
    target_train_X, train_inverse, target_test_X, test_inverse = apply_processor(mod_config["targets"], dataset_train, dataset_test, mod, task_type, is_train, "targets")

    train_batch_idx = get_batch_idx(dataset_train)
    test_batch_idx = get_batch_idx(dataset_test)

    train = {
        "input_X": input_train_X,
        "target_X": target_train_X,
        "inverse": train_inverse,
        "batch_idx": train_batch_idx,
    }

    test = {
        "input_X": input_test_X,
        "target_X": target_test_X,
        "inverse": test_inverse,
        "batch_idx": test_batch_idx,
    }

    inputs_features = input_train_X.shape[1]
    target_features = target_train_X.shape[1]

    return train, test, inputs_features, target_features


def preprocess_one_dataset(
    mod_config: dict,
    dataset_train: ad.AnnData,
    dataset_test: ad.AnnData,
    task_type: str,
    is_train: bool,
):
    """Processes dataset for one modality

    Args:
        mod_config (dict): Modality data configuration
        dataset_train (ad.AnnData): Train dataset
        dataset_test (ad.AnnData): Test dataset
        task_type (str): One of four tasks

    Returns:
        tuple: train dict, test dict and number of features
    """
    if "inputs" in mod_config:
        return preprocess_not_mirror_dataset(
            mod_config, dataset_train, dataset_test, task_type, is_train
        )
    else:
        return preprocess_mirror_dataset(
            mod_config, dataset_train, dataset_test, task_type, is_train
        )


def get_dataset_arguments(first, is_first_mirror, second, is_second_mirror):
    """Returns list of arguments for FourOmicsDataset construction.

    Args:
        first (torch.Tensor): First modality feature matrix
        is_first_mirror (bool): Is it a mirror dataset
        second (torch.Tensor): Second modality feature matrix
        is_second_mirror (bool): Is it a mirror dataset

    Returns:
        List: Of arguments
    """
    arguments = []
    if is_first_mirror:
        arguments.append(first["X"])
        arguments.append(first["X"])
    else:
        arguments.append(first["input_X"])
        arguments.append(first["target_X"])

    if is_second_mirror:
        arguments.append(second["X"])
        arguments.append(second["X"])
    else:
        arguments.append(second["input_X"])
        arguments.append(second["target_X"])
    arguments.append(first["batch_idx"])
    return arguments


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
    first_train, first_test, first_input_features, first_target_features = preprocess_one_dataset(
        config["mod1"], dataset["train_mod1"], dataset["test_mod1"], config["task_type"], is_train
    )
    second_train, second_test, second_input_features, second_target_features = preprocess_one_dataset(
        config["mod2"], dataset["train_mod2"], dataset["test_mod2"], config["task_type"], is_train
    )
    first_small, _, _, _ = preprocess_one_dataset(
        config['mod1'], dataset['train_mod1'][:512], dataset['test_mod1'][:50], config['task_type'], is_train
    )
    second_small, _, _, _ =preprocess_one_dataset(
        config['mod2'], dataset['train_mod2'][:512], dataset['test_mod2'][:50], config['task_type'], is_train
    )

    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    is_first_mirror = "inputs" not in config["mod1"]
    is_second_mirror = "inputs" not in config["mod2"]
    if is_first_mirror and is_second_mirror:
        train_dataset = TwoOmicsDataset(
            first_train["X"], second_train["X"], first_train["batch_idx"]
        )
        test_dataset = TwoOmicsDataset(
            first_test["X"], second_test["X"], first_test["batch_idx"]
        )
        small_dataset = TwoOmicsDataset(
            first_small['X'], second_small['X'], first_small['batch_idx']
        )
    else:
        train_arguments = get_dataset_arguments(
            first_train, is_first_mirror, second_train, is_second_mirror
        )
        train_dataset = FourOmicsDataset(*train_arguments)
        test_arguments = get_dataset_arguments(
            first_test, is_first_mirror, second_test, is_second_mirror
        )
        test_dataset = FourOmicsDataset(*test_arguments)
        small_arguments = get_dataset_arguments(
            first_small, is_first_mirror, second_small, is_second_mirror
        )
        small_dataset = FourOmicsDataset(*small_arguments)
    
    if cuda:
        test_dataset.to('cuda')
        small_dataset.to('cuda')
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=cuda,
        num_workers=1,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    small_train_dataloader = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    result = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "small_train_dataloader": small_train_dataloader,

        "first_train_inverse": first_train["inverse"],
        "first_test_inverse": first_test["inverse"],
        "first_small_inverse": first_small['inverse'],
        "first_input_features": first_input_features,
        "first_target_features": first_target_features,
        
        "second_train_inverse": second_train["inverse"],
        "second_test_inverse": second_test["inverse"],
        'second_small_inverse': second_small['inverse'],
        "second_input_features": second_input_features,
        "second_target_features": second_target_features,

        'train_batch_idx': first_train['batch_idx'],
    }
    return result
