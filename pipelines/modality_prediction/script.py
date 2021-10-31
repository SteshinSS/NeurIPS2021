# VIASH START
# This block will be replaced during viash building. Use it for debugging purposes.
par = {
    "input_train_mod1": "../../data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad",
    "input_train_mod2": "../../data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad",
    "input_test_mod1": "../../data/official/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad",
    "output": "output.h5ad",
}
meta = {
    "resources_dir": "",
}
# VIASH END

# add imported modules in vias
import sys

sys.path.append(meta["resources_dir"])

# import as usual
import logging
import anndata as ad
from lab_scripts.mains.mp import main

logging.basicConfig(level=logging.INFO)
print(meta["resources_dir"])

input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])

resources_dir = meta["resources_dir"]
if resources_dir:
    # It contains path to folder with resources. Let's add slash to concatenate it later.
    resources_dir += "/"
adata = main.predict_submission(
    input_train_mod1, input_train_mod2, input_test_mod1, resources_dir
)

adata.uns["method_id"] = "khrameeva_lab_submission"
adata.write_h5ad(par["output"], compression="gzip")
