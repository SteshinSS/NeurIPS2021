import logging
import sys
import pandas as pd
from pathlib import Path

import anndata as ad
import yaml  # type: ignore

# sys.path.append(str(Path.cwd()))

from lab_scripts.data.preprocessing.common import gex_normalization, gex_qc
from lab_scripts.utils import r_utils

INPUT_PATH = "data/raw/gex_adt/azimuth_gex.h5ad"
OUTPUT_PATH = "data/preprocessed/gex_adt/azimuth_gex.h5ad"

CONFIG = "configs/data/gex/azimuth.yaml"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("azimuth_gex")


def preprocess(data, config):
    log.info("Quality Control...")
    data = gex_qc.standard_qc(data, config)
    log.info("Normalizing...")
    all_batches = data.obs["batch"].unique()
    size_factors = []
    for i, batch in enumerate(all_batches):
        log.info(f"Processing {i} batch ({batch}) out of {len(all_batches)}...")
        batch_data = data[data.obs["batch"] == batch]
        clusters = gex_normalization.get_clusters(batch_data)
        batch_data = gex_normalization.calculate_size_factors(batch_data, clusters)
        size_factors.append(batch_data.obs["size_factors"])
    data.obs["size_factors"] = pd.concat(size_factors)
    data = gex_normalization.normalize(data)
    data.write(OUTPUT_PATH)
    log.info("GEX dataset has been preprocessed. Result is saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    r_utils.activate_R_envinronment()
    data = ad.read_h5ad(INPUT_PATH)

    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    preprocess(data, config)
