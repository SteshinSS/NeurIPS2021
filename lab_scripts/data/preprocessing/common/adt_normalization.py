import numpy as np


def CLR_transform(X: np.ndarray):
    """
    Implements CLR transform from the original CITE-seq paper
    https://doi.org/10.1038/nmeth.4380
    See also discussion here: https://github.com/theislab/scanpy/pull/1117#issuecomment-635963691
    """
    logX = np.log(X + 1)
    return logX - logX.mean(axis=1)[0]
