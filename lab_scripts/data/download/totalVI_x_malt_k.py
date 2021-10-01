import requests  # type: ignore
import shutil

url = "https://github.com/YosefLab/totalVI_reproducibility/raw/master/data/malt_10k_protein_v3.h5ad"
response = requests.get(url, stream=True)

path = "data/raw/gex_adt/totalVI_x_malt_k.h5ad"
with open(path, "xb") as f:
    shutil.copyfileobj(response.raw, f)
