from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class Processor:
    """Pipeline for datasets last-stage preprocessing.
    Does model-depending data preprocessing.
    """

    def __init__(self, config: dict):
        self.top_n_genes = config.get("top_n_genes", None)
        self.use_normalized = config.get("use_normalized", None)
        self.scale = config.get("scale", None)
        self.fitted = False

    def fit(self, dataset: ad.AnnData):
        if self.top_n_genes:
            self._find_highly_variable_genes(dataset)
            dataset = dataset[:, self.highly_variable_genes]
        
        matrix = self._get_matrix(dataset)
        if self.scale:
            self.scaler = MinMaxScaler()
            self.scaler.fit(matrix)
        self.fitted = True

    def transform(self, dataset: ad.AnnData):
        if not self.fitted:
            raise RuntimeError(
                "Processor is not fitted yet. Run .fit() on a train dataset"
            )
        if self.top_n_genes:
            dataset = dataset[:, self.highly_variable_genes]
        matrix = self._get_matrix(dataset)
        if self.scale:
            matrix = self.scaler.transform(matrix)
        return torch.tensor(matrix, dtype=torch.float32)

    def fit_transform(self, dataset: ad.AnnData):
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_transform(
        self, matrix: np.ndarray, size_factors: Optional[pd.Series] = None
    ):
        if self.scale:
            matrix = self.scaler.inverse_transform(matrix)
        if not self.use_normalized:
            matrix = matrix * size_factors.to_numpy()  # type: ignore
        return matrix

    def construct_anndata(self, matrix: np.ndarray, dataset):
        raise NotImplementedError()

    def _find_highly_variable_genes(self, dataset: ad.AnnData):
        self.highly_variable_genes = sc.pp.highly_variable_genes(
            dataset, n_top_genes=self.top_n_genes, inplace=False
        )["highly_variable"]

    def _get_matrix(self, dataset: ad.AnnData):
        if self.use_normalized:
            return dataset.X.toarray()
        else:
            return dataset.layers["counts"].toarray()


class TwoOmicsDataset(Dataset):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        self.first = first
        self.second = second

    def __len__(self):
        return self.first.shape[0]

    def __getitem__(self, idx):
        return (self.first[idx], self.second[idx])
