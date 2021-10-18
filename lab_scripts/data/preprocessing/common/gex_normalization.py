import anndata as ad
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import numpy as np
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

numpy2ri.activate()


def get_clusters(data: ad.AnnData, resolution: float = 0.5):
    """Clustrizes cells for size factor calculation.

    Args:
        data (ad.AnnData): Dataset
        resoultion (float): See docs of scanpy.tl.louvain

    Returns:
        pd.Series: Clusters of the cells
    """
    cluster_data = data.copy()
    sc.pp.normalize_per_cell(cluster_data, counts_per_cell_after=1e6)
    sc.pp.log1p(cluster_data)
    sc.pp.neighbors(cluster_data)
    sc.tl.louvain(cluster_data, key_added="groups", resolution=resolution)
    return cluster_data.obs["groups"]


def calculate_size_factors(
    data_matrix: np.ndarray, clusters: pd.Series, min_mean: float = 0.1
):
    """Calculates size_factors by the scran R package.

    Args:
        data (ad.AnnData): Dataset
        clusters (pd.Series): Clusterization of the cells
        min_mean (float): See docs of scran::computeSumFactors()

    Returns:
        ad.AnnData: Dataset with .obs["size_factors"]
    """
    scran = importr("scran")

    # Convert pd.Series to R object
    with localconverter(ro.default_converter + pandas2ri.converter):
        clusters = ro.conversion.py2rpy(clusters)

    # data_matrix = data.X.T.toarray()
    size_factor = scran.calculateSumFactors(
        data_matrix, clusters=clusters, **{"min.mean": min_mean}
    )
    # data.obs["size_factors"] = size_factors
    return size_factor


def normalize(data: ad.AnnData):
    """Divides the counts by size factors.

    Args:
        data (ad.AnnData): Dataset

    Returns:
        ad.AnnData: Dataset
    """
    # Normalize & log-transform
    data.layers["counts"] = data.X.copy()
    data.X /= data.obs["size_factors"].values[:, None]
    data.X = sc.pp.log1p(data.X)
    return data


def standard_normalization(data: ad.AnnData, config: dict):
    clusters = get_clusters(data)
    data.obs['size_factors'] = calculate_size_factors(data.X.T.toarray(), clusters)
    data = normalize(data)
    return data
