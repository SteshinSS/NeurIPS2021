import logging

import anndata as ad
import yaml  # type: ignore
from lab_scripts.data.preprocessing.common import adt_normalization
from lab_scripts.utils import r_utils

INPUT_PATH = "data/raw/gex_adt/totalVI_10x_adt.h5ad"
OUTPUT_PATH = "data/preprocessed/gex_adt/totalVI_10x_adt.h5ad"
CONFIG = "configs/data/adt/totalVI_10x.yaml"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("totalVI_10x_adt")


def preprocess(data, config):
    # Delete last 10 characters in protein names "_TotalSeqB"
    # CD3_TotalSeqB -> CD3
    data.var.index = [x[:-10] for x in data.var.index.tolist()]
    data.X = adt_normalization.CLR_transform(data.X)
    data.write(OUTPUT_PATH, compression="gzip")
    log.info("ADT dataset has been preprocessed. Result is saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    r_utils.activate_R_envinronment()
    data = ad.read_h5ad(INPUT_PATH)
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    preprocess(data, config)
