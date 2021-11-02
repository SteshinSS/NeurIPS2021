from typing import no_type_check

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import logging


class Processor:
    def __init__(self, config: dict):
        self.fitted = False
        self.scale = config["scale"]

    def fit(self, dataset: ad.AnnData):
        raise NotImplementedError()

    def transform(self, dataset: ad.AnnData):
        raise NotImplementedError()

    def fit_transform(self, dataset: ad.AnnData):
        self.fit(dataset)
        return self.transform(dataset)

    def get_inverse_transform(self, dataset: ad.AnnData = None):
        raise NotImplementedError()


class GEXProcessor(Processor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.log = logging.getLogger("GEXProcessor")
        self.use_normalized = config["use_normalized"]
        gene_fraction = config.get("gene_fraction", None)
        gene_path = config.get("gene_path", None)
        if gene_fraction is not None and gene_fraction < 1.0:
            self.selected_genes = self._select_genes(gene_fraction, gene_path)
        else:
            self.selected_genes = None
        type = config.get("type", None)
        if type is None or type == "float":
            self.type = np.float32
        elif type == "int":
            self.type = np.int32  # type: ignore
        else:
            raise NotImplementedError()

    def fit(self, dataset: ad.AnnData):
        if self.selected_genes is not None:
            dataset = dataset[:, self.selected_genes]

        matrix = self._get_matrix(dataset)
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(matrix)
        self.features = matrix.shape[1]
        self.fitted = True

    def transform(self, dataset: ad.AnnData):
        if not self.fitted:
            raise RuntimeError(
                "Processor is not fitted yet. Run .fit() on a train dataset"
            )
        if self.selected_genes is not None:
            dataset = dataset[:, self.selected_genes]
        matrix = self._get_matrix(dataset)
        if self.scale:
            matrix = self.scaler.transform(matrix)

        return torch.tensor(matrix)

    def get_inverse_transform(self, dataset: ad.AnnData = None):
        @no_type_check
        def f(matrix: torch.Tensor):
            matrix = matrix.numpy()
            if f.scale:
                matrix = f.scaler.inverse_transform(matrix)
            if not f.use_normalized:
                matrix = matrix / f.size_factors
                matrix = np.log(matrix + 1)
            return matrix

        f.scale = self.scale
        if self.scale:
            f.scaler = self.scaler
        f.use_normalized = self.use_normalized
        if not self.use_normalized:
            if dataset is None:
                raise RuntimeError(
                    "To get inverse transformation of not normalized values, pass the dataset with size_factors."
                )
            f.size_factors = self.get_size_factors(dataset)
        return f

    def get_size_factors(self, dataset: ad.AnnData):
        size_factors = dataset.obs["size_factors"].to_numpy()
        return np.expand_dims(size_factors, axis=-1)

    def _select_genes(self, gene_fraction, gene_path):
        weights = np.loadtxt(gene_path, delimiter=",")
        sorted_weights = np.sort(weights)
        total_len = sorted_weights.shape[0]
        best = int(total_len * gene_fraction)
        bound = sorted_weights[-best]
        return weights > bound

    def _get_matrix(self, dataset: ad.AnnData):
        if self.use_normalized:
            if isinstance(dataset.X, np.ndarray):
                return dataset.X.astype(self.type)
            else:
                return dataset.X.toarray().astype(self.type)
        else:
            return dataset.layers["counts"].toarray().astype(self.type)


class ATACProcessor(Processor):
    def __init__(self, config: dict):
        self.log = logging.getLogger("ATACProcessor")
        super().__init__(config)
        self.mapping_file = config["mapping_file"]
        self.use_gene_activity = config["use_gene_activity"]
        if self.use_gene_activity:
            self._prepare_gene_mapping()
        self.window = config["window"]

    def fit(self, dataset: ad.AnnData):
        if self.scale:
            matrix = self.get_matrix(dataset, fit=True)
            self.scaler = StandardScaler()
            self.scaler.fit(matrix)
        self.fitted = True

    def transform(self, dataset: ad.AnnData):
        if not self.fitted:
            raise RuntimeError(
                "Processor is not fitted yet. Run .fit() on a train dataset"
            )
        matrix = self.get_matrix(dataset, fit=False)
        if self.scale:
            matrix = self.scaler.transform(matrix)
        return torch.tensor(matrix, dtype=torch.float32)

    def get_inverse_transform(self, dataset: ad.AnnData = None):
        @no_type_check
        def f(matrix: torch.Tensor):
            matrix = matrix.numpy()
            if f.scale:
                matrix = f.scaler.inverse_transform(matrix)
            return matrix

        f.scale = self.scale
        if self.scale:
            f.scaler = self.scaler
        return f

    def get_matrix(self, dataset: ad.AnnData, fit: bool):
        stage = "fit" if fit else "transform"
        result_matrices = []
        if self.window > 0:
            self.log.info(f"Creating windows in {stage}...")
            result_matrices.append(self._create_windows(dataset))
        if self.use_gene_activity:
            self.log.info(f"Creating gene activity matrix in {stage}...")
            result_matrices.append(self._create_gene_activity_matrix(dataset))
        if self.window <= 0 and not self.use_gene_activity:
            # Just take atac matric as it is
            result_matrices.append(dataset.X.toarray())
        return np.concatenate(result_matrices, axis=1)

    def _create_windows(self, dataset: ad.AnnData):
        X = dataset.X
        window = self.window
        windows = []
        for i in range(0, X.shape[1] - window, window):
            windows.append(X[:, i : i + window].sum(axis=1))
        return np.concatenate(windows, axis=1)

    def _create_gene_activity_matrix(self, dataset: ad.AnnData):
        X = dataset.X.toarray()
        gene_activity = np.zeros((dataset.shape[0], self.total_genes.shape[0]))
        for i, region in enumerate(dataset.var.index):
            if region in self.region_to_id.keys():
                gene_id = self.region_to_id[region]
                gene_activity[:, gene_id] += X[:, i].flatten()
        return gene_activity

    def _prepare_gene_mapping(self):
        genes = pd.read_csv(self.mapping_file, index_col=0)["gene_id"]
        self.region_to_gene = {}
        for region, gene in genes.iteritems():
            self.region_to_gene[region] = gene

        self.total_genes = genes.unique()
        self.gene_to_id = {}
        for i, gene in enumerate(self.total_genes):
            self.gene_to_id[gene] = i
        
        self.region_to_id = {}
        for region, gene in genes.iteritems():
            self.region_to_id[region] = self.gene_to_id[gene]


class ADTProcessor(Processor):
    def __init__(self, config: dict):
        super().__init__(config)

    def fit(self, dataset: ad.AnnData):
        matrix = dataset.X.toarray().astype(np.float32)
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(matrix)
        self.features = matrix.shape[1]
        self.fitted = True

    def transform(self, dataset: ad.AnnData):
        if not self.fitted:
            raise RuntimeError(
                "Processor is not fitted yet. Run .fit() on a train dataset"
            )
        matrix = dataset.X.toarray().astype(np.float32)
        if self.scale:
            matrix = self.scaler.transform(matrix)

        return torch.tensor(matrix)

    def get_inverse_transform(self, dataset: ad.AnnData = None):
        @no_type_check
        def f(matrix: torch.Tensor):
            matrix = matrix.numpy()
            if f.scale:
                matrix = f.scaler.inverse_transform(matrix)
            return matrix

        f.scale = self.scale
        if self.scale:
            f.scaler = self.scaler
        return f


def get_processor(config: dict):
    mod = config["name"]
    if mod == "gex":
        return GEXProcessor(config)
    elif mod == "adt":
        return ADTProcessor(config)
    elif mod == "atac":
        return ATACProcessor(config)
    else:
        raise NotImplementedError()


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
        return (
            (self.first[idx], self.second[idx]),
            (self.first_target[idx], self.second_target[idx]),
            self.batch_idx[idx],
        )

    def to(self, device):
        self.first = self.first.to(device)
        self.first_target = self.first_target.to(device)
        self.second = self.second.to(device)
        self.second_target = self.second_target.to(device)
        self.batch_idx = self.batch_idx.to(device)
