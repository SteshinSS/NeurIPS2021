import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import os
import tempfile
import subprocess


def filter_fragments(
    data: ad.AnnData, min_fragments: Optional[int], max_fragments: Optional[int]
):
    """Filters cells by total number of fragments.
    Args:
        data (ad.AnnData): Dataset with .obs['fragments']
    Returns:
        ad.AnnData: Filtered dataset.
    """
    if min_fragments is not None:
        data = data[data.obs['fragments'] > min_fragments]
    if max_fragments is not None:
        data = data[data.obs['fragments'] < max_fragments]
    return data


def filter_FRiP(
    data: ad.AnnData, min_frip: Optional[int]
):
    """Filters cells by FRiP value.
    Args:
        data (ad.AnnData): Dataset with .obs['FRiP']
    Returns:
        ad.AnnData: Filtered dataset.
    """
    if min_frip is not None:
        data = data[data.obs['FRiP'] > min_frip]
    return data


def filter_blacklist(
    data: ad.AnnData, max_blacklist: Optional[int]
):
    """Filters cells by blacklisted fraction.
    Args:
        data (ad.AnnData): Dataset with .obs['blacklist_fraction']
    Returns:
        ad.AnnData: Filtered dataset.
    """
    if max_blacklist is not None:
        data = data[data.obs['blacklist_fraction'] < max_blacklist]
    return data


def filter_nucleosome_signal(
    data: ad.AnnData, max_nucleosome: Optional[int]
):
    """Filters cells by nucleosome signal.
    Args:
        data (ad.AnnData): Dataset with .obs['nucleosome_signal']
    Returns:
        ad.AnnData: Filtered dataset.
    """
    if max_nucleosome is not None:
        data = data[data.obs['nucleosome_signal'] < max_nucleosome]
    return data


def filter_TSS(
    data: ad.AnnData, min_TSS: Optional[int]
):
    """Filters cells by TSS enrichment score.
    Args:
        data (ad.AnnData): Dataset with .obs['TSS.enrichment']
    Returns:
        ad.AnnData: Filtered dataset.
    """
    if min_TSS is not None:
        data = data[data.obs['TSS.enrichment'] > min_TSS]
    return data


def qc_filter(data: ad.AnnData, config: dict):
    """Filters data by number of total fragments, FRiP, counts in blacklisted fraction, 
    nucleosome signal, TSS enrichment score.
    """
    fragments_min_value = config.get("fragments_min_value", None)
    fragments_max_value = config.get("fragments_max_value", None)
    if fragments_min_value or fragments_max_value:
        data = filter_fragments(data, fragments_min_value, fragments_max_value)

    frip_min_value = config.get("frip_min_value", None)
    if frip_min_value:
        data = filter_FRiP(data, frip_min_value)
        
    blacklist_max_value = config.get("blacklist_max_value", None)
    if blacklist_max_value:
        data = filter_blacklist(data, blacklist_max_value)

    nucleosome_max_value = config.get("nucleosome_max_value", None)
    if nucleosome_max_value:
        data = filter_nucleosome_signal(data, nucleosome_max_value)
        
    tss_min_value = config.get("tss_min_value", None)
    if tss_min_value:
        data = filter_TSS(data, tss_min_value)
    return data


def batch_number(INPUT_PATH):
    path, dirs, files = next(os.walk(INPUT_PATH))
    file_count = len(files)
    batch_number = int(file_count/3)
    batches = [i + str(j) for i, j in zip(['rep']*batch_number, list(range(1, batch_number+1)))]
    return batches


def process_by_batch(INPUT_PATH, config: dict):
    """Calculates QC & filters & binarizes data. Each batch is transformed independently.
    Args:
        INPUT_PATH: Path to the folder with raw data (sparse matrices, fragment files & indices).
    Returns:
        List of ad.AnnData files: Filtered and binarized dataset, each batch is stored independetly.
    """
    # Save batch list
    batches = batch_number(INPUT_PATH)
    batch_files = []

    # Process by batch, put each batch into tempfile, save path to each tempfile into array    
    for i in range(len(batches)):
        
        # Load full data, extract batch, put non-processed batch into tempfile
        batch = sc.read_10x_h5(INPUT_PATH + 'filtered_feature_bc_matrix_{0}.h5'.format(batches[i]), gex_only = False)
        batch.var_names_make_unique()
        batch = batch[:, batch.var.feature_types == "Peaks"].copy()
        batch.var.feature_types = 'ATAC'

        temp_file_batch = tempfile.NamedTemporaryFile("wb", delete=False)
        batch.write_h5ad(temp_file_batch.name)
        print(batch.shape)
        
        # Subprocess call. Save qc_table into tempfile
        command = 'Rscript'
        path2script = './atac_qc.R'
        batch_path = temp_file_batch.name
        fragment_path = INPUT_PATH + 'fragments_{0}.tsv.gz'.format(batches[i])
        output_temp = tempfile.NamedTemporaryFile("wb").name
        subprocess.call([command, path2script, batch_path, fragment_path, output_temp])

        # Load qc_table from tempfile, process: filter & binarize
        # Save processed batch into another tempfile
        qc_table = pd.read_csv(output_temp, delimiter=';')
        batch.obs = qc_table[['fragments', 'FRiP', 'TSS.enrichment', 'nucleosome_signal', 'blacklist_fraction']]
        batch = qc_filter(batch, config)
        batch.X[batch.X != 0] = 1
        
        temp_file_batch_processed = tempfile.NamedTemporaryFile("wb", delete=False)
        batch.write_h5ad(temp_file_batch_processed.name)
        batch_files.append(temp_file_batch_processed.name)
            
    return batch_files