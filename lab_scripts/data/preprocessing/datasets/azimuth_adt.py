import logging

import anndata as ad
import pandas as pd
import numpy as np
import yaml  # type: ignore
from lab_scripts.data.preprocessing.common import adt_normalization, adt_qc
from lab_scripts.utils import r_utils

INPUT_PATH = "data/raw/gex_adt/azimuth_adt.h5ad"
# UTPUT_PATH = "data/preprocessed/gex_adt/azimuth_adt.h5ad"
CONFIG = "configs/data/adt/azimuth.yaml"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("azimuth_adt")


def remove_isoproteins(data):

    # Find isotypic rat proteins
    protein_names = data.var.index.tolist()
    isotype_proteins = []
    cols_to_remove = []
    for i in range(len(protein_names)):
        if protein_names[i][0:2] == "Ra":
            isotype_proteins.append(protein_names[i])
            cols_to_remove.append(i)
        else:
            continue

    # Create df from sparse matrix, extract only columns with isoproteins from df
    # Add columns to data.obsm layer
    isotype_df = pd.DataFrame.sparse.from_spmatrix(
        data.X, index=data.obs.index, columns=data.var.index
    )
    isotype_df = isotype_df[isotype_proteins]
    data.obsm["isotype_controls"] = isotype_df

    # Filter out isoproteins from data.var & data.X
    data.var_filtered = data.var.drop(index=isotype_proteins)
    all_cols = np.arange(data.X.shape[1])
    cols_to_keep = list(set(all_cols) - set(cols_to_remove))
    data.X_filtered = data.X[:, cols_to_keep]

    # Create new AnnData object with separated isoproteins & remove initial AnnData object
    data_new = ad.AnnData(
        X=data.X_filtered,
        var=data.var_filtered,
        obs=data.obs,
        uns=data.uns,
        obsm=data.obsm,
    )
    del data

    return data_new


def preprocess(data, config):
    data = remove_isoproteins(data)

    data = adt_qc.standard_qc(data, config)
    data = adt_normalization.normalize_by_batch(data)
    data.write(OUTPUT_PATH, compression="gzip")
    log.info("ADT dataset has been preprocessed. Result is saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    r_utils.activate_R_envinronment()
    data = ad.read_h5ad(INPUT_PATH)
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    preprocess(data, config)
