import anndata as ad
import numpy as np
from sklearn import metrics
from typing import Union
from lab_scripts.utils import utils

input_type = Union[np.ndarray, ad.AnnData]


def calculate_target(y_pred: input_type, y: input_type):
    if isinstance(y_pred, ad.AnnData):
        y_pred = utils.convert_to_dense(y_pred).X
    if isinstance(y, ad.AnnData):
        y = utils.convert_to_dense(y).X
    if np.isnan(y_pred).any():
        return np.NAN
    return np.sqrt(metrics.mean_squared_error(y_pred, y))
