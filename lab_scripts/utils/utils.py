from typing import Tuple

import anndata as ad


def get_mod(dataset: ad.AnnData):
    """Returns type of dataset 'ADT', 'GEX' or 'ATAC'."""
    feature_types = dataset.var["feature_types"]
    assert len(feature_types.unique()) == 1
    return dataset.var["feature_types"][0]


def get_task_type(
    mod1: str,
    mod2: str,
) -> str:
    task_type = mod1 + "_to_" + mod2

    allowed_types = set([
        "GEX_to_ATAC",
        "ATAC_to_GEX",
        "GEX_to_ADT",
        "ADT_to_GEX",
    ])
    if task_type not in allowed_types:
        raise ValueError(f"Inappropriate type of input datasets: {mod1, mod2}")
    return task_type


def convert_to_dense(dataset: ad.AnnData):
    """Returns a copy of AnnData dataset with dense X matrix."""
    result = dataset.copy()
    result.X = dataset.X.toarray()
    return result
