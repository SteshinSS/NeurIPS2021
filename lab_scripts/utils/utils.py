import tempfile
from pathlib import Path
import os
import anndata as ad
import requests  # type: ignore
from scipy.sparse.csc import csc_matrix
from tqdm import tqdm
import pytorch_lightning as pl


def get_mod(dataset: ad.AnnData) -> str:
    """Returns type of dataset 'adt', 'gex' or 'atac'."""
    feature_types = dataset.var["feature_types"]
    assert len(feature_types.unique()) == 1
    return dataset.var["feature_types"][0].lower()


def get_task_type(
    mod1: str,
    mod2: str,
) -> str:
    task_type = mod1 + "_to_" + mod2

    allowed_types = set(
        [
            "gex_to_atac",
            "atac_to_gex",
            "gex_to_adt",
            "adt_to_gex",
        ]
    )
    if task_type not in allowed_types:
        raise ValueError(f"Inappropriate type of input datasets: {mod1, mod2}")
    return task_type


def convert_to_dense(dataset: ad.AnnData):
    """Returns a copy of AnnData dataset with dense X matrix."""
    result = dataset.copy()
    if isinstance(result.X, csc_matrix):
        result.X = dataset.X.toarray()
    return result


def download_to_tempfile(url: str):
    """Download file into tempfile

    Args:
        url (str): url of a file

    Returns:
        file: temporary file.

    Example:
        temp_file = download_to_tempfile('https://site.com/dataset')
        data = ad.read_h5ad(temp_file.name)
    """
    temp_file = tempfile.NamedTemporaryFile("wb")

    # In such way there is no need to keep file in memory.
    # The file will be saved by blocks.
    # The progress bar is took from here https://github.com/shaypal5/tqdl/blob/master/tqdl/core.py
    response = requests.get(url, stream=True, timeout=None)
    response.raise_for_status()

    file_size = int(response.headers.get("Content-Length", None))
    block_size = 1024
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        temp_file.write(data)
    progress_bar.close()
    temp_file.flush()
    if file_size != 0 and progress_bar.n != file_size:
        print("Something went wrong in downloading")
    return temp_file


def change_directory_to_repo():
    """Changes working directory to the repository root folder.
    """
    current_dir = Path.cwd()
    for parent in current_dir.parents:
        # Repository is the first folder with the .git folder
        files = list(parent.glob('.git'))
        if files:
            os.chdir(str(parent))


def set_deafult_seed(seed=228):
    pl.seed_everything(seed, workers=True)