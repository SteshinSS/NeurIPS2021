import urllib.request
import tempfile
import anndata as ad
import scanpy as sc
import pandas as pd

import tarfile
import numpy as np
import gzip
from scipy import io

par = {
    "id": "azimuth_ref",
    "input_count": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE164378&format=file",
    "input_meta": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE164378&format=file&file=GSE164378%5Fsc%2Emeta%2Edata%5F3P%2Ecsv%2Egz",
    "organism": "human",
    "output_rna": "output_rna.h5ad",
    "output_mod2": "output_mod2.h5ad"
}

tar_temp = tempfile.NamedTemporaryFile()
url = par['input_count']
urllib.request.urlretrieve(url, tar_temp.name)

meta_temp = tempfile.NamedTemporaryFile()
url = par['input_meta']
urllib.request.urlretrieve(url, meta_temp.name)

samples = ['GSM5008737_RNA_3P', 'GSM5008738_ADT_3P'] # first sample is rna, second is protein data
adatas = [] 

with tarfile.open(tar_temp.name, 'r') as tar:
    for sample in samples:
        with gzip.open(tar.extractfile(sample + '-matrix.mtx.gz'), 'rb') as mm:
            X = io.mmread(mm).T.tocsr()
        obs = pd.read_csv(
            tar.extractfile(sample + '-barcodes.tsv.gz'), 
            compression='gzip',
            header=None, 
            sep='\t',
            index_col=0
        )
        obs.index.name = None
        var = pd.read_csv(
            tar.extractfile(sample + '-features.tsv.gz'), 
            compression='gzip',
            header=None, 
            sep='\t'
        ).iloc[:, :1]
        var.columns = ['names']
        var.index = var['names'].values
        adata = ad.AnnData(X=X, obs=obs, var=var)

        adata.var_names_make_unique()
        adatas.append(adata)
    tar.close()

adata_RNA = adatas[0]
adata_ADT = adatas[1]


meta = pd.read_csv(meta_temp.name, index_col = 0, compression = "gzip")
meta_adt = meta.loc[:,~meta.columns.str.endswith('RNA')]
meta_rna = meta.loc[:,~meta.columns.str.endswith('ADT')]

# set obs
adata_RNA.obs = adata_RNA.obs.join(meta_rna).rename(columns = {'Batch':'seq_batch', 'donor':'batch'})
adata_RNA.obs['cell_type'] = adata_RNA.obs['celltype.l2']
adata_ADT.obs = adata_ADT.obs.join(meta_adt).rename(columns = {'Batch':'seq_batch', 'donor':'batch'})
adata_ADT.obs['cell_type'] = adata_ADT.obs['celltype.l2']

#  set var
adata_RNA.var['feature_types'] = "GEX"
adata_ADT.var['feature_types'] = "ADT"

# set uns 
uns = { "dataset_id" : par["id"], "organism" : par["organism"] }
adata_RNA.uns = uns
adata_ADT.uns = uns

# save output
adata_RNA.write_h5ad("data/raw/gex_adt/azimuth_RNA.h5ad", compression = "gzip")
adata_ADT.write_h5ad(par['data/raw/gex_adt/azimuth_ADT.h5ad'], compression = "gzip")