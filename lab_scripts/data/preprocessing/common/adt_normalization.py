import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import tempfile


def CLR_transform(initial_sparse):
    """CLR-transform
    Args:
        initial_sparse (ad.AnnData.X): sparse matrix
    Returns:
        transformed_sparse (ad.AnnData.X): sparse matrix
    """
    array = initial_sparse.toarray()
    logn1 = np.log(array + 1)
    mean = np.nanmean(logn1, axis=1)
    exponent = np.exp(mean)
    ratio = (array / exponent[:, None]) + 1
    T_clr = np.log(ratio)
    transformed_sparse = csr_matrix(T_clr)
    return transformed_sparse


def normalize_by_batch(data: ad.AnnData):
    """CLR-transform the initial counts. Each batch is transformed independently.
    Args:
        data (ad.AnnData): Dataset
    Returns:
        ad.AnnData: Dataset
    """
    # Save batch list & save var and uns attributes of initial AnnData object
    batches = sorted(data.obs["batch"].value_counts().index.tolist())
    var = data.var
    uns = data.uns

    # Put initial AnnData object into tempfile
    batch_files = []
    temp_file_full = tempfile.NamedTemporaryFile("wb")
    data.write_h5ad(temp_file_full.name)
    del data

    # CLR-transform by batch, put each batch into tempfile, save path to each tempfile into array
    for i in range(len(batches)):
        data = ad.read_h5ad(temp_file_full.name)
        batch = data[data.obs["batch"] == batches[i]].copy()
        del data
        batch.X = CLR_transform(batch.X)
        temp_file_batch = tempfile.NamedTemporaryFile("wb", delete=False)
        batch.write_h5ad(temp_file_batch.name)
        batch_files.append(temp_file_batch.name)

    # Reload all batches from tempfiles and concatenate them into normalized AnnData object
    for i in range(len(batch_files)):
        if i == 0:
            data = ad.read_h5ad(batch_files[i])
        else:
            data = ad.concat([data, ad.read_h5ad(batch_files[i])], axis=0)

    # Add var and uns attributes to normalized AnnData object
    data.var = var
    data.uns = uns
    return data
