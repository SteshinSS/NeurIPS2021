import requests  # type: ignore
import shutil

URL = "https://github.com/YosefLab/totalVI_reproducibility/raw/master/data/malt_10k_protein_v3.h5ad"
response = requests.get(URL, stream=True)

PATH = "data/raw/gex_adt/totalVI_x_malt_k.h5ad"
with open(PATH, "xb") as f:
    shutil.copyfileobj(response.raw, f)
