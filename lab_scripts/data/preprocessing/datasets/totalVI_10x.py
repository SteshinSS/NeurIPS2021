import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml  # type: ignore
from scipy.sparse import csr_matrix

sys.path.append(str(Path.cwd()))

from lab_scripts.data.preprocessing.common import gex_normalization, gex_qc
from lab_scripts.utils import r_utils

INPUT_PATH = "data/raw/gex_adt/totalVI_10x.h5ad"
OUTPUT_GEX_PATH = "data/preprocessed/gex_adt/totalVI_10x_gex.h5ad"
OUTPUT_ADT_PATH = "data/preprocessed/gex_adt/totalVI_10x_adt.h5ad"
UNS = {"dataset_id": "totalVI_10x", "organism": "human"}

COMMON_GEX_CONFIG = "configs/data/gex/common.yaml"
GEX_CONFIG = "configs/data/gex/totalVI_10x.yaml"

COMMON_ADT_CONFIG = "configs/data/adt/common.yaml"
ADT_CONFIG = "configs/data/adt/totalVI_10x.yaml"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("totalVI_10x")


def CLR_transform(sparse):
    """
    implements the CLR transform used in CITEseq (need to confirm in Seurat's code)
    https://doi.org/10.1038/nmeth.4380
    source: https://github.com/theislab/scanpy/pull/1117
    """
    df = pd.DataFrame.sparse.from_spmatrix(sparse)
    logn1 = np.log(df + 1)
    T_clr = logn1.sub(logn1.mean(axis=1), axis=0)
    return T_clr


def adt_do_normalization(data):
    data.X = CLR_transform(data.X)
    return data


def adt_do_quality_control(data):
    data.obs["total_counts"] = np.asarray(np.sum(data.X, axis=1)).reshape(-1)
    sc.pp.filter_cells(data, min_counts=1100)
    sc.pp.filter_cells(data, max_counts=24000)

    data.obs["n_genes"] = np.asarray((data.X > 0).sum(axis=1)).reshape(-1)
    sc.pp.filter_cells(data, min_genes=8)  # 60% of n_genes
    sc.pp.filter_genes(data, min_cells=20)
    return data


def get_adt(data):
    data_adt = ad.AnnData(
        X=data.obsm["protein_expression"],
        var=pd.DataFrame(index=list(data.uns["protein_names"])),
        uns=UNS,
    )

    data_adt.obs.index = data.obs.index
    data_adt.var["feature_types"] = "ADT"
    return data_adt


def preprocess_adt(data):
    with open(COMMON_ADT_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    # Delete last 10 characters in protein names "_TotalSeqB"
    # CD3_TotalSeqB -> CD3
    data.var.index = [x[:-10] for x in data.var.index.tolist()]
    data = adt_do_quality_control(data)
    data = adt_do_normalization(data)
    data.write(OUTPUT_ADT_PATH)
    log.info(
        "ADT dataset has been preprocessed. Result is saved to %s", OUTPUT_ADT_PATH
    )


def get_gex(data):
    data_gex = ad.AnnData(
        X=csr_matrix(data.X),
        obs=data.obs.loc[:, ["n_genes", "percent_mito", "n_counts"]],
        var=data.var,
        uns=UNS,
    )
    data_gex.var["feature_types"] = "GEX"
    return data_gex


def preprocess_gex(data):
    with open(COMMON_GEX_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    # Update common config with current dataset config
    with open(GEX_CONFIG, "r") as f:
        config.update(yaml.safe_load(f))

    log.info("Quality Control...")
    data = gex_qc.standard_qc(data, config)
    log.info("Normalizing...")
    data = gex_normalization.standard_normalization(data, config)
    data.write(OUTPUT_GEX_PATH)
    log.info(
        "GEX dataset has been preprocessed. Result is saved to %s", OUTPUT_GEX_PATH
    )


if __name__ == "__main__":
    r_utils.activate_R_envinronment()
    data = ad.read_h5ad(INPUT_PATH)

    gex_data = get_gex(data)
    preprocess_gex(gex_data)

    adt_data = get_adt(data)
    preprocess_adt(adt_data)
