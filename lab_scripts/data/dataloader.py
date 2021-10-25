"""Module for opening dataset.

Imagine you have the official dataset, some processed public_1 dataset, and some public_2 
dataset. You want to use first two for training. Instead of opening them manually add
them into this module, so you need only to call dataloader.load_data('my_dataset').
"""

from lab_scripts.utils import utils
import anndata as ad
import numpy as np

library = {
    "mp/official/adt_to_gex": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad",
    },
    "mp/official/gex_to_adt": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad",
    },
    "mp/official/atac_to_gex": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad",
    },
    "mp/official/gex_to_atac": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad",
    },
}

COMMON_GEX_ADT = "data/official/common/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.manual_formatting.output_rna.h5ad"
COMMON_GEX_ATAC = "data/official/common/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.manual_formatting.output_rna.h5ad"
COMMON_ADT = "data/official/common/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.manual_formatting.output_mod2.h5ad"
COMMON_ATAC = "data/official/common/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.manual_formatting.output_mod2.h5ad"


def filter_genes(gex_dataset, filter_genes_params):
    if filter_genes_params is None:
        return gex_dataset

    fraction, path_to_genes = filter_genes_params
    if fraction >= 1.0:
        return gex_dataset
        
    weights = np.loadtxt(path_to_genes, delimiter=',')
    sorted_weights = np.sort(weights)
    total_len = sorted_weights.shape[0]
    best = int(total_len * fraction)
    bound = sorted_weights[-best]
    print(bound)

    gex_dataset.var['weight'] = weights
    gex_dataset = gex_dataset[:, weights > bound]
    return gex_dataset


def select_batches(dataset, batches):
    dataset_batches = dataset.obs['batch'].astype('string')
    selected_idx = dataset_batches.apply(lambda batch: batch in batches).values
    return dataset[selected_idx]


def load_custom_data(task_type, train_batches, test_batches, filter_genes_params=None, val_size=None):
    if task_type != 'gex_to_adt':
        raise NotImplementedError()
    
    gex = ad.read_h5ad(COMMON_GEX_ADT)
    adt = ad.read_h5ad(COMMON_ADT)

    gex = filter_genes(gex, filter_genes_params)
    train_gex = select_batches(gex, train_batches)
    test_gex = select_batches(gex, test_batches)

    train_adt = select_batches(adt, train_batches)
    test_adt = select_batches(adt, test_batches)

    result = {
        'test_mod1': test_gex,
        'test_mod2': test_adt,
    }

    if val_size:
        all_idx = np.arange(train_adt.shape[0])
        np.random.shuffle(all_idx)
        val_idx = all_idx[:val_size]
        train_idx = all_idx[val_size:]
        result['train_mod1'] = train_gex[train_idx]
        result['val_mod1'] = train_gex[val_idx]
        result['train_mod2'] = train_adt[train_idx]
        result['val_mod2'] = train_adt[val_idx]
    else:
        result['train_mod1'] = train_gex
        result['train_mod2'] = train_adt
    return result


def load_data(name, val_size=None, filter_genes=None, train_batches=None, val_batches=None):
    """Opens dataset by name.

    Current datasets:
        Official:
            'mp/official/gex_to_adt'
            'mp/official/adt_to_gex'
            'mp/official/atac_to_gex'
            'mp/official/gex_to_atac'
    """
    files = library[name]
    result = dict()
    for name, path in files.items():
        result[name] = ad.read_h5ad(path)
    result = inject_size_factors(result)

    if train_batches is not None:
        batches = result['train_mod1'].obs['batch'].astype('string')
        train_idx = batches.apply(lambda batch: batch in train_batches).values
        result['train_mod1'] = result['train_mod1'][train_idx]
        result['train_mod2'] = result['train_mod2'][train_idx]

    if filter_genes:
        fraction, path_to_genes = filter_genes
        weights = np.loadtxt(path_to_genes, delimiter=',')
        sorted_weights = np.sort(weights)
        total_len = sorted_weights.shape[0]
        best = int(total_len * fraction)
        bound = sorted_weights[-best]
        print(bound)

        result['train_mod1'].var['weight'] = weights
        result['train_mod1'] = result['train_mod1'][:, result['train_mod1'].var['weight'] > bound]
        result['test_mod1'].var['weight'] = weights
        result['test_mod1'] = result['test_mod1'][:, result['test_mod1'].var['weight'] > bound]


    if val_size:
        idx = np.arange(result['train_mod1'].shape[0])
        np.random.shuffle(idx)
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]
        result['val_mod1'] = result['train_mod1'][val_idx]
        result['val_mod2'] = result['train_mod2'][val_idx]
        result['train_mod1'] = result['train_mod1'][train_idx]
        result['train_mod2'] = result['train_mod2'][train_idx]
    return result


def inject_size_factors(data: dict):
    """At the current moment, authors forgot to save size_factors in datasets.
    This function adds them.

    Args:
        data (dict): load_data's output
    """
    def inject(train, test, mod, task_type):
        if mod == 'gex':
            if task_type == 'gex_to_adt':
                common = ad.read_h5ad(COMMON_GEX_ADT)
            else:
                common = ad.read_h5ad(COMMON_GEX_ATAC)
        else:
            return train, test
        
        common_train = common[common.obs['is_train'], :]
        train.obs['size_factors'] = common_train.obs['size_factors']
        common_test = common[~common.obs['is_train'], :]
        test.obs['size_factors'] = common_test.obs['size_factors']
        return train, test

    first_mod = utils.get_mod(data['train_mod1'])
    second_mod = utils.get_mod(data['train_mod2'])
    task_type = utils.get_task_type(first_mod, second_mod)
    data['train_mod1'], data['test_mod1'] = inject(data['train_mod1'], data['test_mod1'], first_mod, task_type)
    data['train_mod2'], data['test_mod2'] = inject(data['train_mod2'], data['test_mod2'], second_mod, task_type)
    return data


if __name__=='__main__':
    data = load_data('mp/official/gex_to_adt')
    print(data['train_mod1'].obs['size_factors'].shape)
