import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
from rpy2.robjects import numpy2ri, pandas2ri, r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.sparse import csr_matrix

sys.path.append(str(Path.cwd()))

from lab_scripts.utils import r_utils

numpy2ri.activate()


INPUT_PATH = "data/raw/gex_adt/totalVI_x_malt_k.h5ad"
OUTPUT_GEX_PATH = "data/preprocessed/gex_adt/totalVI_x_malt_k_gex.h5ad"
OUTPUT_ADT_PATH = "data/preprocessed/gex_adt/totalVI_x_malt_k_adt.h5ad"
UNS = {"dataset_id": "totalVI_x_malt_k", "organism": "human"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("totalVI_x_malt_k")


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


def preprocess_adt(data):
    # Delete last 10 characters in protein names "_TotalSeqB"
    # CD3_TotalSeqB -> CD3
    data.var.index = [x[:-10] for x in data.var.index.tolist()]
    data = adt_do_quality_control(data)
    data = adt_do_normalization(data)
    return data


def get_adt(data):
    data_ADT = ad.AnnData(
        X=csr_matrix(data.obsm["protein_expression"]),
        var=pd.DataFrame(index=list(data.uns["protein_names"])),
        uns=UNS,
    )

    data_ADT.obs.index = data.obs.index
    data_ADT.var["feature_types"] = "ADT"
    return data_ADT


def gex_do_normalization(data):
    log.info("Normalizing...")
    cluster_data = data.copy()
    sc.pp.normalize_per_cell(cluster_data, counts_per_cell_after=1e6)
    sc.pp.log1p(cluster_data)
    sc.pp.pca(cluster_data, n_comps=15)
    sc.pp.neighbors(cluster_data)
    sc.tl.louvain(cluster_data, key_added="groups", resolution=0.5)

    ## Calculate size factors
    scran = importr("scran")
    clusters = cluster_data.obs["groups"]
    with localconverter(ro.default_converter + pandas2ri.converter):
        clusters = ro.conversion.py2rpy(clusters)
    data_matrix = data.X.T.toarray()
    size_factors = scran.calculateSumFactors(
        data_matrix, clusters=clusters, **{"min.mean": 0.1}
    )
    data.obs["size_factors"] = size_factors

    data.layers["counts"] = data.X.copy()

    # Normalize & log-transform
    data.X /= data.obs["size_factors"].values[:, None]
    data.layers["log_norm"] = sc.pp.log1p(data.X)
    return data


def get_total_detected_genes(data):
    total_detected_genes = (data.X > 0).sum(axis=1)
    total_detected_genes = np.asarray(total_detected_genes).reshape(-1)
    return total_detected_genes


def get_mito_genes_percent(data):
    is_mito = data.var_names.str.startswith("MT-")
    total_mito_genes = np.sum(data[:, is_mito].X, axis=1).A1
    total_all_genes = np.sum(data.X, axis=1).A1
    mito_genes_percent = (total_mito_genes / total_all_genes) * 100.0
    return mito_genes_percent


def gex_do_quality_control(data):
    log.info("Quality control...")
    ## Fraction of mito genes
    data.obs["pct_counts_mt"] = get_mito_genes_percent(data)
    data = data[data.obs["pct_counts_mt"] < 20]

    ## Total number of counts
    data.obs["n_counts"] = np.asarray(np.sum(data.X, axis=1)).reshape(-1)
    sc.pp.filter_cells(data, min_counts=1500)
    sc.pp.filter_cells(data, max_counts=40000)

    ## Total number of detected genes
    data.obs["n_genes"] = get_total_detected_genes
    sc.pp.filter_cells(data, min_genes=700)
    sc.pp.filter_genes(data, min_cells=20)
    return data


def preprocess_gex(data):
    data = gex_do_quality_control(data)
    data = gex_do_normalization(data)
    return data


def get_gex(data):
    data_gex = ad.AnnData(
        X=csr_matrix(data.X),
        obs=data.obs.loc[:, ["n_genes", "percent_mito", "n_counts"]],
        var=data.var,
        uns=UNS,
    )
    data_gex.var["feature_types"] = "GEX"
    return data_gex


if __name__ == "__main__":
    r_utils.activate_R_envinronment()

    data = ad.read_h5ad(INPUT_PATH)

    data_gex = get_gex(data)
    data_gex = preprocess_gex(data_gex)
    data_gex.write(OUTPUT_GEX_PATH)
    log.info(
        "GEX dataset has been preprocessed. Result is saved to %s", OUTPUT_GEX_PATH
    )

    data_adt = get_adt(data)
    data_adt = preprocess_adt(data_adt)
    data_adt.write(OUTPUT_ADT_PATH)
    log.info(
        "ADT dataset has been preprocessed. Result is saved to %s", OUTPUT_ADT_PATH
    )
