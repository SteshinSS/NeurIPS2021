import anndata as ad
import numpy as np
import scanpy as sc
import logging
from scib.metrics import (
    cell_cycle,
    graph_connectivity,
    nmi,
    silhouette,
    silhouette_batch,
    trajectory_conservation,
)
from scib.metrics.clustering import opt_louvain

log = logging.getLogger("je")


def create_anndata(solution: ad.AnnData, embeddings: np.ndarray):
    data = ad.AnnData(
        X=embeddings,
        obs=solution.obs,
        uns=solution.uns
    )
    data.obsm["X_emb"] = embeddings
    sc.pp.neighbors(data, use_rep="X_emb")
    return data


def asw_batch(data: ad.AnnData):
    # https://github.com/theislab/scib/blob/main/scib/metrics/silhouette.py
    return silhouette_batch(
        data, batch_key="batch", group_key="cell_type", embed="X_emb", verbose=False
    )


def asw_label(data: ad.AnnData):
    # https://github.com/theislab/scib/blob/main/scib/metrics/silhouette.py
    return silhouette(data, group_key="cell_type", embed="X_emb")


def calculate_nmi(data: ad.AnnData):
    opt_louvain(
        data,
        label_key="cell_type",
        cluster_key="cluster",
        plot=False,
        inplace=True,
        force=True,
    )
    return nmi(data, group1="cluster", group2="cell_type")


def cc_cons(data: ad.AnnData, solution: ad.AnnData):
    score = cell_cycle(
        adata_pre=solution,
        adata_post=data,
        batch_key="batch",
        embed="X_emb",
        recompute_cc=False,
        organism="human",
    )
    return score


def ti_cons(data: ad.AnnData, solution: ad.AnnData):
    score_rna = trajectory_conservation(
        adata_pre=solution,
        adata_post=data,
        label_key='cell_type',
        pseudotime_key='pseudotime_order_GEX'
    )

    adt_atac_trajectory = 'pseudotime_order_ATAC' if 'pseudotime_order_ATAC' in solution.obs else 'pseudotime_order_ADT'
    score_adt_atac = trajectory_conservation(
        adata_pre=solution,
        adata_post=data,
        label_key='cell_type',
        pseudotime_key=adt_atac_trajectory
    )

    score_mean = (score_rna + score_adt_atac) / 2
    return score_mean


def graph_conn(data: ad.AnnData):
    return graph_connectivity(data, label_key='cell_type')


def calculate_metrics(data: ad.AnnData, solution: ad.AnnData):
    result = {}
    log.info('Calculating asw_batch...')
    result['asw_batch'] = asw_batch(data)

    log.info('Calculating asw_label...')
    result['asw_label'] = asw_label(data)

    log.info('Calculating nmi...')
    result['nmi'] = calculate_nmi(data)

    log.info('Calculating cc_cons...')
    result['cc_cons'] = cc_cons(data, solution)

    log.info('Calculating ti_cons...')
    result['ti_cons'] = ti_cons(data, solution)

    log.info('Calculating graph_conn...')
    result['graph_conn'] = graph_conn(data)
    return result
