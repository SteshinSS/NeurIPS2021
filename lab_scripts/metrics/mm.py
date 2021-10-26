import numpy as np

def calculate_target(sol_pred: np.ndarray, sol_true: np.ndarray):
    row_sum = sol_pred.sum(axis=1)
    np.testing.assert_allclose(row_sum, 1.0, rtol=1e-5)
    score = (sol_pred * sol_true).sum()
    return (score / sol_pred.shape[0]) * 1000.0