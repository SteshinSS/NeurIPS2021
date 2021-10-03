import requests  # type: ignore
import shutil
import anndata as ad
import tempfile
import pandas as pd
from scipy.sparse import csr_matrix

URL = "https://github.com/YosefLab/totalVI_reproducibility/raw/master/data/malt_10k_protein_v3.h5ad"
PATH_GEX = "data/raw/gex_adt/totalVI_10x_gex.h5ad"
PATH_ADT = "data/raw/gex_adt/totalVI_10x_adt.h5ad"

UNS = {"dataset_id": "totalVI_10x", "organism": "human"}

# Download dataset into a temporary file
response = requests.get(URL, stream=True)
temp_file = tempfile.NamedTemporaryFile()
shutil.copyfileobj(response.raw, temp_file)

data = ad.read_h5ad(temp_file.name)

data_gex = ad.AnnData(
    X=csr_matrix(data.X),
    obs=data.obs,
    var=data.var,
    uns=UNS,
)
data_gex.write(PATH_GEX)

data_adt = ad.AnnData(
    X=data.obsm["protein_expression"],
    var=pd.DataFrame(index=list(data.uns["protein_names"])),
    uns=UNS,
)
data_adt.obs.index = data.obs.index
data_adt.write(PATH_ADT)
