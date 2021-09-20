import pathlib
import anndata as ad


library = {
    'data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2': {
        'input_train_mod1': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad',
        'input_train_mod2': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad',
        'input_test_mod1': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad',
        'input_test_mod2': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
    },

    'data/official/predict_modality/openproblems_bmmc_cite_phase1_rna': {
        'input_train_mod1': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad',
        'input_train_mod2': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad',
        'input_test_mod1': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad',
        'input_test_mod2': 'data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad',
    },

    'data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2': {
        'input_train_mod1': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad',
        'input_train_mod2': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad',
        'input_test_mod1': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad',
        'input_test_mod2': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
    },

    'data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna': {
        'input_train_mod1': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
        'input_train_mod2': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad',
        'input_test_mod1': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad',
        'input_test_mod2': 'data/official/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad',
    }
}

def load_data(name):
    files = library[name]
    result = dict()
    for name, path in files.items():
        result[name] = ad.read_h5ad(path)
    return result