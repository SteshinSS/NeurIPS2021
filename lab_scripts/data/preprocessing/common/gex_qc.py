import anndata as ad
import numpy as np
import scanpy as sc
from typing import Optional


def calculate_mito_fraction(data: ad.AnnData):
    is_mito = data.var_names.str.startswith("MT-")
    total_mito_genes = np.sum(data[:, is_mito].X, axis=1).A1
    total_all_genes = np.sum(data.X, axis=1).A1
    mito_genes_percent = total_mito_genes / total_all_genes
    data.obs["pct_counts_mt"] = mito_genes_percent
    return data


def filter_mito_fraction(data: ad.AnnData, threshold: float):
    """Filters cells by fraction of mitochondrial genes.

    Args:
        data (ad.AnnData): Dataset
        threshold (float): Maximum allowed fraction of mito genes.

    Returns:
        ad.AnnData: Filtered dataset.
    """
    if "pct_counts_mt" not in data.obs:
        data = calculate_mito_fraction(data)
    return data[data.obs["pct_counts_mt"] < threshold]


def filter_count(
    data: ad.AnnData, min_counts: Optional[int], max_counts: Optional[int]
):
    """Filters cells by number of counts.

    Args:
        data (ad.AnnData): Dataset
        min_counts (int): Minimal allowed number of counts.
        max_counts (int): Maximal allowed number of counts.

    Returns:
        ad.AnnData: Filtered dataset.
    """
    data.obs["n_counts"] = np.asarray(np.sum(data.X, axis=1)).reshape(-1)
    if min_counts is not None:
        sc.pp.filter_cells(data, min_counts=min_counts)
    if max_counts is not None:
        sc.pp.filter_cells(data, max_counts=max_counts)
    return data


def calculate_n_cells(data: ad.AnnData):
    """For each gene calculates total number of cells expressing this gene.
    Result is saved to data.var['n_cells']

    Args:
        data (ad.AnnData): Dataset

    Returns:
        ad.AnnData: Dataset
    """
    total_cells_expressing_gene = data.X.sum(axis=0).A1
    data.var["n_cells"] = total_cells_expressing_gene
    return data


def filter_genes(data: ad.AnnData, min_genes: Optional[int], min_cells: Optional[int]):
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
    if min_genes is not None:
        sc.pp.filter_cells(data, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(data, min_cells=min_cells)
    return data


def standard_qc(data: ad.AnnData, config: dict):
    """Filters data by mito genes, number of counts and number of genes"""
    data = filter_mito_fraction(data, config["mito_max_fraction"])
    data = filter_count(data, config["cell_min_counts"], config["cell_max_counts"])
    data = filter_genes(data, config["cell_min_genes"], config["gene_min_cells"])
    return data
