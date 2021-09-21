import anndata as ad
from scipy.sparse.csc import csc_matrix


def get_mod(dataset: ad.AnnData) -> str:
    """Returns type of dataset 'adt', 'gex' or 'atac'."""
    feature_types = dataset.var["feature_types"]
    assert len(feature_types.unique()) == 1
    return dataset.var["feature_types"][0].lower()


def get_task_type(
    mod1: str,
    mod2: str,
) -> str:
    task_type = mod1 + "_to_" + mod2

    allowed_types = set(
        [
            "gex_to_atac",
            "atac_to_gex",
            "gex_to_adt",
            "adt_to_gex",
        ]
    )
    if task_type not in allowed_types:
        raise ValueError(f"Inappropriate type of input datasets: {mod1, mod2}")
    return task_type


def convert_to_dense(dataset: ad.AnnData):
    """Returns a copy of AnnData dataset with dense X matrix."""
    result = dataset.copy()
    if isinstance(result.X, csc_matrix):
        result.X = dataset.X.toarray()
    return result
