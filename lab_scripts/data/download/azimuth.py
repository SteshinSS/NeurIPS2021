import anndata as ad
import pandas as pd
import logging

import tarfile
import gzip
from scipy import io

import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from lab_scripts.utils import utils

URL_COUNTS = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE164378&format=file"
URL_META = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE164378&format=file&file=GSE164378%5Fsc%2Emeta%2Edata%5F3P%2Ecsv%2Egz"
PATH_GEX = "data/raw/gex_adt/azimuth_gex.h5ad"
PATH_ADT = "data/raw/gex_adt/azimuth_adt.h5ad"
UNS = {"dataset_id": "azimuth", "organism": "human"}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("azimuth_download")

log.info("Downloading from %s...", URL_COUNTS)
tar_temp = utils.download_to_tempfile(URL_COUNTS)
samples = [
    "GSM5008737_RNA_3P",
    "GSM5008738_ADT_3P",
]  # first sample is rna, second is protein data
adatas = []

log.info("Downloading is finished. Splitting...")

with tarfile.open(tar_temp.name, "r") as tar:
    for sample in samples:
        with gzip.open(tar.extractfile(sample + "-matrix.mtx.gz"), "rb") as mm:  # type: ignore
            X = io.mmread(mm).T.tocsr()
        obs = pd.read_csv(
            tar.extractfile(sample + "-barcodes.tsv.gz"),
            compression="gzip",
            header=None,
            sep="\t",
            index_col=0,
        )
        obs.index.name = None
        var = pd.read_csv(
            tar.extractfile(sample + "-features.tsv.gz"),
            compression="gzip",
            header=None,
            sep="\t",
        ).iloc[:, :1]
        var.columns = ["names"]
        var.index = var["names"].values
        adata = ad.AnnData(X=X, obs=obs, var=var)

        adata.var_names_make_unique()
        adatas.append(adata)
    tar.close()

meta_temp = utils.download_to_tempfile(URL_META)
meta = pd.read_csv(meta_temp.name, index_col=0, compression="gzip")

adata_gex = adatas[0]
meta_gex = meta.loc[:, ~meta.columns.str.endswith("ADT")]
adata_gex.obs = adata_gex.obs.join(meta_gex).rename(
    columns={"Batch": "seq_batch", "donor": "batch"}
)
adata_gex.obs["cell_type"] = adata_gex.obs["celltype.l2"]
adata_gex.var["feature_types"] = "GEX"
adata_gex.uns = UNS
adata_gex.write_h5ad(PATH_GEX, compression="gzip")
log.info("GEX dataset in downloaded to %s", PATH_GEX)

adata_adt = adatas[1]
meta_adt = meta.loc[:, ~meta.columns.str.endswith("RNA")]
adata_adt.obs = adata_adt.obs.join(meta_adt).rename(
    columns={"Batch": "seq_batch", "donor": "batch"}
)
adata_adt.obs["cell_type"] = adata_adt.obs["celltype.l2"]
adata_adt.var["feature_types"] = "ADT"
adata_adt.uns = UNS
adata_adt.write_h5ad(PATH_ADT, compression="gzip")
log.info("ADT dataset in downloaded to %s", PATH_ADT)
