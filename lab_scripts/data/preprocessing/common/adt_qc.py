import anndata as ad
import numpy as np
import scanpy as sc
from typing import Optional


def filter_isotype_count(
    data: ad.AnnData, min_isocounts: Optional[int], max_isocounts: Optional[int]
):
    """Filters cells by number of counts in isotype controls.
    Args:
        data (ad.AnnData): Dataset
    Returns:
        ad.AnnData: Dataset with data.obs['iso_count']
    """
    isocounts_sum = np.sum(data.obsm["isotype_controls"], axis=1)
    data.obs["iso_count"] = np.asarray(isocounts_sum).reshape(-1)
    if min_isocounts is not None:
        data = data[data.obs["iso_count"] > min_isocounts]
    if max_isocounts is not None:
        data = data[data.obs["iso_count"] < max_isocounts]
    return data


def filter_total_count(
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
    """For each gene calculates total number of cells expressing this protein.
    Result is saved to data.var['n_cells']
    Args:
        data (ad.AnnData): Dataset
    Returns:
        ad.AnnData: Dataset
    """
    total_cells_expressing_protein = data.X.sum(axis=0).A1
    data.var["n_cells"] = total_cells_expressing_protein
    return data


def filter_genes(
    data: ad.AnnData, min_proteins: Optional[int], min_cells: Optional[int]
):
    """Filters cells and genes.
    Args:
        data (ad.AnnData): Dataset
        min_proteins (int): Minimal allowed number of detected proteins in cells. Filters cells.
        min_cells (int): Minimal number of expressing cells for each gene. Filters genes.
    Returns:
        ad.AnnData: Filtered dataset.
    """
    total_detected_proteins = (data.X > 0).sum(axis=1)
    data.obs["n_antibodies_by_counts"] = np.asarray(total_detected_proteins).reshape(-1)
    if min_proteins is not None:
        sc.pp.filter_cells(data, min_genes=min_proteins)
    if min_cells is not None:
        sc.pp.filter_genes(data, min_cells=min_cells)
    return data


def standard_qc(data: ad.AnnData, config: dict):
    """Filters data by number of isotype counts, number of total counts and number of proteins"""
    isotype_min_counts = config.get("isotype_min_counts", None)
    isotype_max_counts = config.get("isotype_max_counts", None)
    if isotype_min_counts or isotype_max_counts:
        data = filter_isotype_count(data, isotype_min_counts, isotype_max_counts)

    cell_min_counts = config.get("cell_min_counts", None)
    cell_max_counts = config.get("cell_max_counts", None)
    if cell_min_counts or cell_max_counts:
        data = filter_total_count(data, cell_min_counts, cell_max_counts)

    cell_min_proteins = config.get("cell_min_proteins", None)
    proteins_min_cells = config.get("proteins_min_cells", None)
    if cell_min_proteins or proteins_min_cells:
        data = filter_genes(data, cell_min_proteins, proteins_min_cells)
    return data
