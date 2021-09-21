# VIASH START
# This block will be replaced during viash building. Use it for debugging purposes.
par = {
    "input_train_mod1": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad",
    "input_train_mod2": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad",
    "input_test_mod1": "sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad",
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
from lab_scripts.mains import baseline_linear

logging.basicConfig(level=logging.INFO)
print(meta["resources_dir"])
import os

print(os.getcwd())

input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])

resources_dir = meta["resources_dir"] + "/"
adata = baseline_linear.predict_submission(
    input_train_mod1, input_train_mod2, input_test_mod1, resources_dir
)

adata.uns["method_id"] = "khrameeva_lab_submission"
adata.write_h5ad(par["output"], compression="gzip")
