"""This is our main script"""

## VIASH START
# This block will be replaced during viash building. Use it for debugging purposes.
par = {
    'input_train_mod1':
        'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'input_train_mod2':
        'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'input_test_mod1':
        'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad',
    'output':
        'output.h5ad',
    'config':
        'configs/README.md',
    'checkpoint':
        'checkpoints/README.md'
}
meta = {
    'resources_dir': '',
}
## VIASH END

# add imported modules in vias
import sys
sys.path.append(meta['resources_dir'])

# import as usual
import anndata as ad
from lab_scripts.mains import baseline_official


input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

adata = baseline_official.main(
    input_train_mod1, input_train_mod2, input_test_mod1
)

adata.uns['method_id'] = 'khrameeva_lab_submission'
adata.write_h5ad(par['output'], compression="gzip")
