"""Official baseline from here: https://github.com/openproblems-bio/neurips2021-notebooks/blob/main/notebooks/explore/explore_multiome.ipynb"""
import numpy as np
import anndata as ad

def fit_predict(input_train_mod1, input_train_mod2, input_test_mod1):
    '''Dummy method that predicts mean(input_train_mod2) for all cells'''
    y_pred = np.repeat(input_train_mod2.layers["log_norm"].mean(axis=0).reshape(-1,1).T, input_test_mod1.shape[0], axis=0)
    
    # Prepare the ouput data object
    pred_test_mod2 = ad.AnnData(
        X=y_pred,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,
    )
    
    pred_test_mod2.uns["method"] = "mean"

    return pred_test_mod2