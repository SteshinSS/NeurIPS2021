import anndata as ad
from sklearn import metrics


def calculate_target(y_pred: ad.AnnData, y: ad.AnnData):
    assert type(y_pred) == type(y)
    y_pred = y_pred.X
    y = y.X.toarray()
    return metrics.mean_squared_error(y_pred, y, squared=False)
