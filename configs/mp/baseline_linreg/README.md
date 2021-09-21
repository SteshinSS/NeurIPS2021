# baseline_linreg config
```python
alpha  # See sklearn.ElasticNet
l1_ration  # See sklearn.ElasticNet

data  # Name of dataset
checkpoint_path  # Optional path to model's checkpoint

mod1:  # Optional preprocessing
  reduce_dim:  # Optional dim reduction  
    n_components:  # Number of components of svd
    checkpoint_path  # Optional path to svd checkpoint 

mod2:  # The same
```