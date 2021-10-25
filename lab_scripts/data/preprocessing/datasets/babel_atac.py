import logging
import anndata as ad
import pandas as pd
import numpy as np
import yaml  # type: ignore
from lab_scripts.data.preprocessing.common import atac_preprocessing
from lab_scripts.utils import r_utils

INPUT_PATH = "data/raw/gex_atac/"
OUTPUT_PATH = "data/preprocessed/gex_atac/"
CONFIG = "configs/data/atac/babel.yaml"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("babel_atac")

def preprocess(input_path: str, config: dict):
    # Get path to the folder with input files and config
    # Returns list of temp paths to processed batches (AnnData objects)
    batch_list = atac_preprocessing.process_by_batch(INPUT_PATH, config)
    nums = list(range(1, len(batch_list)+1))
    
    for num, batch_path in enumerate(batch_list, 1):
        batch = ad.read_h5ad(batch_path)
        batch.write_h5ad(OUTPUT_PATH + f'babel_atac_rep{num}.h5ad', compression = "gzip")  
    log.info("ATAC dataset has been preprocessed. Result is saved to %s", OUTPUT_PATH)

    
if __name__ == "__main__":
    r_utils.activate_R_envinronment()
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    preprocess(INPUT_PATH, config) 