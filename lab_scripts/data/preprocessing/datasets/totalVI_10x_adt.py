import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml  # type: ignore

from lab_scripts.utils import r_utils

INPUT_PATH = "data/raw/gex_adt/totalVI_10x_adt.h5ad"
OUTPUT_PATH = "data/preprocessed/gex_adt/totalVI_10x_adt.h5ad"

COMMON_CONFIG = "configs/data/adt/common.yaml"
CONFIG = "configs/data/adt/totalVI_10x.yaml"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("totalVI_10x_adt")


def CLR_transform(matrix):
    """
    implements the CLR transform used in CITEseq (need to confirm in Seurat's code)
    https://doi.org/10.1038/nmeth.4380
    source: https://github.com/theislab/scanpy/pull/1117
    """
    df = pd.DataFrame(matrix)
    logn1 = np.log(df + 1)
    T_clr = logn1.sub(logn1.mean(axis=1), axis=0)
    return T_clr


def do_normalization(data):
    data.X = CLR_transform(data.X)
    return data


def do_quality_control(data):
    data.obs["total_counts"] = np.asarray(np.sum(data.X, axis=1)).reshape(-1)
    sc.pp.filter_cells(data, min_counts=1100)
    sc.pp.filter_cells(data, max_counts=24000)

    data.obs["n_genes"] = np.asarray((data.X > 0).sum(axis=1)).reshape(-1)
    sc.pp.filter_cells(data, min_genes=8)  # 60% of n_genes
    sc.pp.filter_genes(data, min_cells=20)
    return data


def preprocess(data, config):
    # Delete last 10 characters in protein names "_TotalSeqB"
    # CD3_TotalSeqB -> CD3
    data.var.index = [x[:-10] for x in data.var.index.tolist()]
    data = do_quality_control(data)
    data = do_normalization(data)
    data.write(OUTPUT_PATH)
    log.info("ADT dataset has been preprocessed. Result is saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    r_utils.activate_R_envinronment()
    data = ad.read_h5ad(INPUT_PATH)
    with open(COMMON_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    preprocess(data, config)
