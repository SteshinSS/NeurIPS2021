import anndata as ad
from sklearn.metrics import mean_square_error


def calculate_target(y_pred: ad.AnnData, y: ad.AnnData):
    assert type(y_pred) == type(y)
    y_pred = y_pred.X
    y = y.X
    return mean_square_error(y_pred, y, squared=False)
