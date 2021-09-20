import logging
import pickle
from typing import Dict

import anndata as ad
import numpy as np
from lab_scripts.utils import utils
from sklearn.linear_model import ElasticNet

log = logging.getLogger('linear_regression')


def train(input_train_mod1: ad.AnnData, input_train_mod2: ad.AnnData, config: dict):
    input_train_mod1 = utils.convert_to_dense(input_train_mod1)
    input_train_mod2 = utils.convert_to_dense(input_train_mod2)
    regressor = ElasticNet(
        selection='random',
        alpha=config['alpha'],
        l1_ratio=config['l1_ratio']
    )
    log.info('Fitting...')
    regressor.fit(input_train_mod1.X, input_train_mod2.X)
    return regressor


def predict(
    input_train_mod1: ad.AnnData,
    input_train_mod2: ad.AnnData,
    input_test_mod1: ad.AnnData,
    regressor,
) -> ad.AnnData:
    input_test_mod1 = utils.convert_to_dense(input_test_mod1)
    predictions = regressor.predict(input_test_mod1.X)

    result = ad.AnnData(
        X=predictions,
        obs=input_test_mod1,
        var=input_train_mod2,
        uns={"dataset_id": input_train_mod1.uns["dataset_id"]},
    )
    return result
