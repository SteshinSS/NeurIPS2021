from typing import no_type_check

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from sklearn.preprocessing import StandardScaler
from lab_scripts.data.preprocessing.common.adt_normalization import CLR_transform
from torch.utils.data import Dataset


class Processor:
    """Pipeline for datasets last-stage preprocessing.
    Does model-depending data preprocessing.
    """

    def __init__(self, config: dict, mod: str = None):
        if mod is None:
            mod = config['name']
        self.gene_fraction = config.get("gene_fraction", None)
        self.gene_path = config.get("gene_path", None)
        if self.gene_fraction is not None and self.gene_fraction < 1.0:
            self._select_genes()
        self.use_normalized = config["use_normalized"]
        self.scale = config["scale"]
        self.mod = mod
        type = config['type']
        if type == 'float':
            self.type = np.float32
        elif type == 'int':
            self.type = np.int32  # type: ignore
        else:
            raise NotImplementedError()
        self.fitted = False

    def fit(self, dataset: ad.AnnData):
        if self.gene_fraction is not None and self.gene_fraction < 1.0:
            dataset = dataset[:, self.selected_genes]

        matrix = self._get_matrix(dataset)
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(matrix)
        self.fitted = True

    def transform(self, dataset: ad.AnnData):
        if not self.fitted:
            raise RuntimeError(
                "Processor is not fitted yet. Run .fit() on a train dataset"
            )
        if self.gene_fraction is not None and self.gene_fraction < 1.0:
            dataset = dataset[:, self.selected_genes]
        matrix = self._get_matrix(dataset)
        if self.scale:
            matrix = self.scaler.transform(matrix)

        args = {}  # type: ignore
        if self.mod == "gex" and not self.use_normalized:
            size_factors = dataset.obs["size_factors"].to_numpy()
            size_factors = np.expand_dims(size_factors, axis=-1)
            args['size_factors'] = size_factors
        inverse_transform = self._get_inverse_transform(args)
        return torch.tensor(matrix), inverse_transform

    def fit_transform(self, dataset: ad.AnnData):
        self.fit(dataset)
        return self.transform(dataset)

    def _get_inverse_transform(self, args):
        @no_type_check
        def f(matrix: torch.Tensor, small_idx: np.ndarray = None):
            matrix = matrix.numpy()
            if f.scale:
                matrix = f.scaler.inverse_transform(matrix)
            if not f.use_normalized:
                if f.mod == "gex":
                    if small_idx is not None:
                        matrix = matrix / f.size_factors[small_idx]
                    else:
                        matrix = matrix / f.size_factors
                    matrix = np.log(matrix + 1)
                elif f.mod == "adt":
                    matrix = CLR_transform(matrix)
                else:
                    raise NotImplementedError()
            return matrix

        f.scale = self.scale
        if self.scale:
            f.scaler = self.scaler
        f.use_normalized = self.use_normalized
        f.mod = self.mod
        if self.mod == "gex" and not self.use_normalized:
            f.size_factors = args["size_factors"]
        return f

    def _select_genes(self):
        weights = np.loadtxt(self.gene_path, delimiter=",")
        sorted_weights = np.sort(weights)
        total_len = sorted_weights.shape[0]
        best = int(total_len * self.gene_fraction)
        bound = sorted_weights[-best]
        self.selected_genes = weights > bound

    def _get_matrix(self, dataset: ad.AnnData):
        if self.use_normalized:
            if isinstance(dataset.X, np.ndarray):
                return dataset.X.astype(self.type)
            else:
                return dataset.X.toarray().astype(self.type)
        else:
            return dataset.layers["counts"].toarray().astype(self.type)


class OneOmicDataset(Dataset):
    def __init__(self, first: torch.Tensor):
        super().__init__()
        self.first = first
    
    def __len__(self):
        return self.first.shape[0]
    
    def __getitem__(self, idx):
        return self.first[idx]


class TwoOmicsDataset(Dataset):
    def __init__(self, first: torch.Tensor, second: torch.Tensor, batch_idx=None):
        super().__init__()
        self.first = first
        self.second = second
        self.batch_idx = batch_idx

    def __len__(self):
        return self.first.shape[0]

    def __getitem__(self, idx):
        if self.batch_idx is not None:
            return self.first[idx], self.second[idx], self.batch_idx[idx]
        else:
            return self.first[idx], self.second[idx]
    
    def to(self, device):
        self.first = self.first.to(device)
        self.second = self.second.to(device)
        if self.batch_idx is not None:
            self.batch_idx = self.batch_idx.to(device)


class FourOmicsDataset(Dataset):
    def __init__(self, first, first_target, second, second_target, batch_idx=None):
        super().__init__()
        self.first = first
        self.first_target = first_target
        self.second = second
        self.second_target = second_target
        self.batch_idx = batch_idx
    
    def __len__(self):
        return self.first.shape[0]
    
    def __getitem__(self, idx):
        return (self.first[idx], self.second[idx]), (self.first_target[idx], self.second_target[idx]), self.batch_idx[idx]
    
    def to(self, device):
        self.first = self.first.to(device)
        self.first_target = self.first_target.to(device)
        self.second = self.second.to(device)
        self.second_target = self.second_target.to(device)
        self.batch_idx = self.batch_idx.to(device)
