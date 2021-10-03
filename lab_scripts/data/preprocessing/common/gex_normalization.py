import anndata as ad
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

numpy2ri.activate()


def get_clusters(data: ad.AnnData):
    """Clustrizes cells for size factor calculation.

    Args:
        data (ad.AnnData): Dataset

    Returns:
        pd.Series: Clusters of the cells
    """
    cluster_data = data.copy()
    sc.pp.normalize_per_cell(cluster_data, counts_per_cell_after=1e6)
    sc.pp.log1p(cluster_data)
    sc.pp.pca(cluster_data, n_comps=15)  # is it necessary?
    sc.pp.neighbors(cluster_data)  # Why?
    sc.tl.louvain(
        cluster_data, key_added="groups", resolution=0.5
    )  # Why resolution=0.5?
    return cluster_data.obs["groups"]


def calculate_size_factors(data: ad.AnnData, clusters: pd.Series):
    """Calculates size_factors by the scran R package.

    Args:
        data (ad.AnnData): Dataset
        clusters (pd.Series): Clusterization of the cells

    Returns:
        ad.AnnData: Dataset with .obs["size_factors"]
    """
    scran = importr("scran")

    # Convert pd.Series to R object
    with localconverter(ro.default_converter + pandas2ri.converter):
        clusters = ro.conversion.py2rpy(clusters)

    data_matrix = data.X.T.toarray()
    size_factors = scran.calculateSumFactors(
        data_matrix, clusters=clusters, **{"min.mean": 0.1}
    )
    data.obs["size_factors"] = size_factors
    return data


def normalize(data: ad.AnnData):
    """Divides the counts by size factors.

    Args:
        data (ad.AnnData): Dataset

    Returns:
        ad.AnnData: Dataset
    """
    data.layers["counts"] = data.X.copy()

    # Normalize & log-transform
    data.X /= data.obs["size_factors"].values[:, None]
    data.layers["log_norm"] = sc.pp.log1p(data.X)
    return data


def standard_normalization(data: ad.AnnData, config: dict):
    clusters = get_clusters(data)
    data = calculate_size_factors(data, clusters)
    data = normalize(data)
    return data
