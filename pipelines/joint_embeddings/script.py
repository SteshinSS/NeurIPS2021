## VIASH START
dataset_path = '../../data/official/joint_embedding/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.censor_dataset.output_'
# dataset_path = 'output/datasets/joint_embedding/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_'

par = {
    'input_mod1': dataset_path + 'mod1.h5ad',
    'input_mod2': dataset_path + 'mod2.h5ad',
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
import torch
from lab_scripts.mains.je.main import predict_submission
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('je')

import anndata as ad



input_mod1 = ad.read_h5ad(par['input_mod1'])
input_mod2 = ad.read_h5ad(par['input_mod2'])

resources_dir = meta["resources_dir"]
if resources_dir:
    # It contains path to folder with resources. Let's add slash to concatenate it later.
    resources_dir += "/"
adata = predict_submission(
    input_mod1, input_mod2, resources_dir
)

adata.uns["method_id"] = "KhrameevaLab"
adata.write_h5ad(par["output"], compression="gzip")