## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': 'output/datasets/match_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
    'input_train_mod2': 'output/datasets/match_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
    'input_train_sol': 'output/datasets/match_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_sol.h5ad',
    'input_test_mod1': 'output/datasets/match_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad',
    'input_test_mod2': 'output/datasets/match_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad',
    'output': 'output.h5ad',
}
meta = {
    "resources_dir": "",
}
## VIASH END
import sys
sys.path.append(meta["resources_dir"])

# import as usual
import logging
import anndata as ad
from lab_scripts.mains import mm as main

logging.basicConfig(level=logging.INFO)
print(meta["resources_dir"])

input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_train_sol = ad.read_h5ad(par['input_train_sol'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])

resources_dir = meta["resources_dir"]
if resources_dir:
    # It contains path to folder with resources. Let's add slash to concatenate it later.
    resources_dir += "/"
adata = main.predict_submission(
    input_train_mod1, input_train_mod2, input_train_sol, input_test_mod1, input_test_mod2, resources_dir
)

adata.uns["method_id"] = "KhrameevaLab"
adata.write_h5ad(par["output"], compression="gzip")
