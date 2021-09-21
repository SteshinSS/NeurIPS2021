import logging

import anndata as ad
import numpy as np
from lab_scripts.utils import utils
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet

log = logging.getLogger("linear_regression")


def apply_svd(dataset: ad.AnnData, svd):
    """Returns np.ndarray matrix out of AnnData dataset."""
    mod = utils.get_mod(dataset)
    log.info(f"Reducing dimension for mod {mod} into {svd.n_components} dimensions...")
    X = svd.transform(dataset.X)
    return X


def invert_svd(X: np.ndarray, svd):
    return svd.inverse_transform(X)


def predict(
    test_mod1: np.ndarray,
    regressor,
):
    log.info("Predicting...")
    predictions = regressor.predict(test_mod1)
    return predictions


def train_svd(train_matrix, n_components):
    svd = TruncatedSVD(n_components, algorithm="arpack")
    svd.fit(train_matrix)
    return svd


def train_regressor(
    train_mod1,
    train_mod2,
    config: dict,
):
    regressor = ElasticNet(
        selection="random", alpha=config["alpha"], l1_ratio=config["l1_ratio"]
    )
    log.info("Fitting...")
    regressor.fit(train_mod1, train_mod2)

    return regressor
