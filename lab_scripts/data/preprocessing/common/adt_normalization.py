import numpy as np


def CLR_transform(counts):
    logn1 = np.log(counts+1)
    mean = np.nanmean(logn1, axis = 1)
    exponent = np.exp(mean)
    ratio = (counts/exponent[:,None]) + 1
    T_clr = np.log(ratio)
    return T_clr
