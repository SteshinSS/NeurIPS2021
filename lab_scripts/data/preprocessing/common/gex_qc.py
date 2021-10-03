import anndata as ad
import numpy as np
import scanpy as sc


def filter_mito_fraction(data: ad.AnnData, threshold: float):
    """Filters cells by fraction of mitochondrial genes.

    Args:
        data (ad.AnnData): Dataset
        threshold (float): Maximum allowed fraction of mito genes.

    Returns:
        ad.AnnData: Filtered dataset.
    """
    is_mito = data.var_names.str.startswith("MT-")
    total_mito_genes = np.sum(data[:, is_mito].X, axis=1).A1
    total_all_genes = np.sum(data.X, axis=1).A1
    mito_genes_percent = total_mito_genes / total_all_genes
    data.obs["pct_counts_mt"] = mito_genes_percent
    return data[data.obs["pct_counts_mt"] < threshold]


def filter_count(data: ad.AnnData, min_counts: int, max_counts: int):
    """Filters cells by number of counts.

    Args:
        data (ad.AnnData): Dataset
        min_counts (int): Minimal allowed number of counts.
        max_counts (int): Maximal allowed number of counts.

    Returns:
        ad.AnnData: Filtered dataset.
    """
    data.obs["n_counts"] = np.asarray(np.sum(data.X, axis=1)).reshape(-1)
    sc.pp.filter_cells(data, min_counts=min_counts)
    sc.pp.filter_cells(data, max_counts=max_counts)
    return data


def filter_genes(data: ad.AnnData, min_genes: int, min_cells: int):
    """Filters cells and genes.

    Args:
        data (ad.AnnData): Dataset
        min_genes (int): Minimal allowed number of detected genes in cells. Filters cells.
        min_cells (int): Minimal number of expressing cells for each gene. Filters genes.

    Returns:
        ad.AnnData: Filtered dataset.
    """
    total_detected_genes = (data.X > 0).sum(axis=1)
    data.obs["n_genes"] = np.asarray(total_detected_genes).reshape(-1)
    sc.pp.filter_cells(data, min_genes=min_genes)
    sc.pp.filter_genes(data, min_cells=min_cells)
    return data


def standard_qc(data: ad.AnnData, config: dict):
    """Filters data by mito genes, number of counts and number of genes"""
    data = filter_mito_fraction(data, config["mito_max_fraction"])
    data = filter_count(data, config["cell_min_counts"], config["cell_max_counts"])
    data = filter_genes(data, config["cell_min_genes"], config["gene_min_cells"])
    return data
