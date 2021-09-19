"""This is our main script"""

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1':
        'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'input_train_mod2':
        'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'input_test_mod1':
        'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad',
    'distance_method':
        'minkowski',
    'output':
        'output.h5ad',
    'n_pcs':
        50,
}
meta = {
    'resources_dir': '',
}
## VIASH END

# add imported modules in vias
import sys

sys.path.append(meta['resources_dir'])

import logging
import anndata as ad

from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

from lab_scripts.models.baselines import baseline_linear

logging.basicConfig(level=logging.INFO)

# TODO: change this to the name of your method
method_id = "python_starter_kit"

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

adata = baseline_linear.fit_predict(
    input_train_mod1, input_train_mod2, input_test_mod1
)
adata.uns['method_id'] = method_id

logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression="gzip")
