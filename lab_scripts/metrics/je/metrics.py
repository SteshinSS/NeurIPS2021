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


def create_anndata(train_mod1: ad.AnnData, embeddings: np.ndarray):
    result_ad = train_mod1.copy()
    result_ad["X_emb"] = embeddings
    sc.pp.neighbors(result_ad, use_rep="X_emb")
    return result_ad


def asw_batch(data: ad.AnnData):
    # https://github.com/theislab/scib/blob/main/scib/metrics/silhouette.py
    _, result = silhouette_batch(
        data, batch_key="batch", group_key="cell_type", embed="X_emb", verbose=False
    )

    return result["silhouette_score"].mean()


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


def cc_cons(data: ad.AnnData):
    score = cell_cycle(
        adata_pre=data,
        adata_post=data,
        batch_key="batch",
        embed="X_emb",
        recompute_cc=True,
        organism="human",
    )
    return score


def ti_cons(data: ad.AnnData):
    pass


def graph_conn(data: ad.AnnData):
    pass


def calculate_metric(data: ad.AnnData):
    result = {}
    log.info('Calculating asw_batch...')
    result['asw_batch'] = asw_batch(data)

    log.info('Calculating asw_label...')
    result['asw_label'] = asw_label(data)

    log.info('Calculating nmi...')
    result['nmi'] = calculate_nmi(data)

    log.info('Calculating cc_cons...')
    result['cc_cons'] = cc_cons(data)
    return result
