"""Module for opening dataset.

Imagine you have the official dataset, some processed public_1 dataset, and some public_2 
dataset. You want to use first two for training. Instead of opening them manually add
them into this module, so you need only to call dataloader.load_data('my_dataset').
"""


import anndata as ad

library = {
    "pm/official/adt_to_gex": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad",
    },
    "pm/official/gex_to_adt": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad",
    },
    "pm/official/atac_to_gex": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad",
    },
    "pm/official/gex_to_atac": {
        "train_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad",
        "train_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad",
        "test_mod1": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad",
        "test_mod2": "data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad",
    },
}


def load_data(name):
    """Opens dataset by name.

    Current datasets:
        Official:
            'pm/official/gex_to_adt'
            'pm/official/adt_to_gex'
            'pm/official/atac_to_gex'
            'pm/official/gex_to_atac'
    """
    files = library[name]
    result = dict()
    for name, path in files.items():
        result[name] = ad.read_h5ad(path)
    return result
