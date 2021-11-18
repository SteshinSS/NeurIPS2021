import numpy as np
import torch

def calculate_target(sol_pred):
    row_sum = sol_pred.sum(axis=1)
    np.testing.assert_allclose(row_sum, 1.0, rtol=1e-5)
    score = torch.diagonal(sol_pred).sum()
    return (score / sol_pred.shape[0]) * 1000.0