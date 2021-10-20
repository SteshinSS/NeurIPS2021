import rpy2.robjects as ro
import scanpy as sc
import numpy as np
import tempfile
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


def normalize_by_batch(data: ad.AnnData):
    """Normalizes initial counts. Each batch is transformed independently.
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

    # Normalize by batch
    for i in range(len(batches)):
        data = ad.read_h5ad(temp_file_full.name)
        batch = data[data.obs["batch"] == batches[i]].copy()
        del data
        clusters = get_clusters(batch)
        batch.obs["size_factors"] = calculate_size_factors(
            batch.X.T.toarray(), clusters
        )
        batch = normalize(batch)
        del clusters

        # Put each batch into tempfile, save path to each tempfile into array
        temp_file_batch = tempfile.NamedTemporaryFile("wb", delete=False)
        batch.write_h5ad(temp_file_batch.name)
        batch_files.append(temp_file_batch.name)

    # Reload all batches from tempfiles and concatenate them into normalized AnnData object
    batch_reloaded = []
    for i in range(len(batch_files)):
        batch = ad.read_h5ad(batch_files[i])
        batch_reloaded.append(batch)
    data = ad.concat(batch_reloaded, axis=0)

    # Add var and uns attributes to normalized AnnData object
    data.var = var
    data.uns = uns
    return data
