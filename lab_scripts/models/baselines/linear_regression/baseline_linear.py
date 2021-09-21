import logging
import pickle
from typing import Dict

import anndata as ad
import numpy as np
from lab_scripts.utils import utils
from sklearn.linear_model import ElasticNet

log = logging.getLogger('linear_regression')


def train(train_mod1: ad.AnnData, train_mod2: ad.AnnData, config: dict):
    train_mod1 = utils.convert_to_dense(train_mod1)
    train_mod2 = utils.convert_to_dense(train_mod2)
    regressor = ElasticNet(
        selection='random',
        alpha=config['alpha'],
        l1_ratio=config['l1_ratio']
    )
    log.info('Fitting...')
    regressor.fit(train_mod1.X, train_mod2.X)
    return regressor


def predict(
    train_mod1: ad.AnnData,
    train_mod2: ad.AnnData,
    test_mod1: ad.AnnData,
    regressor,
) -> ad.AnnData:
    test_mod1 = utils.convert_to_dense(test_mod1)
    predictions = regressor.predict(test_mod1.X)

    result = ad.AnnData(
        X=predictions,
        obs=test_mod1.obs,
        var=train_mod2.var,
        uns={"dataset_id": train_mod1.uns["dataset_id"]},
    )
    return result
