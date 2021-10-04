import logging

import anndata as ad
import yaml  # type: ignore

from lab_scripts.data.preprocessing.common import gex_normalization, gex_qc
from lab_scripts.utils import r_utils

INPUT_PATH = "data/raw/gex_adt/totalVI_10x_gex.h5ad"
OUTPUT_PATH = "data/preprocessed/gex_adt/totalVI_10x_gex.h5ad"

CONFIG = "configs/data/gex/totalVI_10x.yaml"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("totalVI_10x_gex")


def preprocess(data, config):
    log.info("Quality Control...")
    data = gex_qc.standard_qc(data, config)
    log.info("Normalizing...")
    data = gex_normalization.standard_normalization(data, config)
    data.write(OUTPUT_PATH, compression="gzip")
    log.info("GEX dataset has been preprocessed. Result is saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    r_utils.activate_R_envinronment()
    data = ad.read_h5ad(INPUT_PATH)

    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    preprocess(data, config)
