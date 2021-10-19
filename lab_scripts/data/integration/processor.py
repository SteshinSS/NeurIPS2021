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

    def __init__(self, config: dict, mod: str):
        self.top_n_genes = config.get("top_n_genes", None)
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
        if self.top_n_genes:
            self._find_highly_variable_genes(dataset)
            dataset = dataset[:, self.highly_variable_genes]

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
        if self.top_n_genes:
            dataset = dataset[:, self.highly_variable_genes]
        matrix = self._get_matrix(dataset)
        if self.scale:
            matrix = self.scaler.transform(matrix)

        args = {}  # type: ignore
        if self.mod == "gex":
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
        def f(matrix: torch.Tensor):
            matrix = matrix.numpy()
            if f.scale:
                matrix = f.scaler.inverse_transform(matrix)
            if not f.use_normalized:
                if f.mod == "gex":
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
        if self.mod == "gex":
            f.size_factors = args["size_factors"]
        return f

    def construct_anndata(self, matrix: np.ndarray, dataset):
        raise NotImplementedError()

    def _find_highly_variable_genes(self, dataset: ad.AnnData):
        self.highly_variable_genes = sc.pp.highly_variable_genes(
            dataset, n_top_genes=self.top_n_genes, inplace=False
        )["highly_variable"]

    def _get_matrix(self, dataset: ad.AnnData):
        if self.use_normalized:
            return dataset.X.toarray().astype(self.type)
        else:
            return dataset.layers["counts"].toarray().astype(self.type)


class TwoOmicsDataset(Dataset):
    def __init__(self, first: torch.Tensor, second: torch.Tensor, batch_idx):
        super().__init__()
        self.first = first
        self.second = second
        self.batch_idx = batch_idx

    def __len__(self):
        return self.first.shape[0]

    def __getitem__(self, idx):
        return (self.first[idx], self.second[idx]), self.batch_idx[idx]
    
    def to(self, device):
        self.first = self.first.to(device)
        self.second = self.second.to(device)
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


def compare(lhs, rhs):
    lhs = np.ma.masked_array(lhs, lhs > 0.0)
    rhs = np.ma.masked_array(rhs, rhs > 0.0)
    difference = (lhs - rhs).mean()
    if difference >= 0.01:
        print(difference)
        assert False


def __test_1():
    from lab_scripts.data import dataloader
    data = dataloader.load_data('mp/official/gex_to_adt')['train_mod1']
    config = {
        'use_normalized': False,
        'scale': False,
    }
    mod = "gex"
    processor = Processor(config, mod)
    
    result, inverse_function = processor.fit_transform(data)
    compare(result, data.layers['counts'].toarray())
    compare(inverse_function(result), data.X.toarray())

def __test_2():
    from lab_scripts.data import dataloader
    data = dataloader.load_data('mp/official/gex_to_adt')['train_mod1']
    config = {
        'use_normalized': True,
        'scale': False
    }
    mod = "gex"
    processor = Processor(config, mod)
    result, inverse_function = processor.fit_transform(data)
    compare(result, data.X.toarray())
    compare(inverse_function(result), data.X.toarray())

def __test_3():
    from lab_scripts.data import dataloader
    data = dataloader.load_data('mp/official/gex_to_adt')['train_mod1']
    config = {
        'use_normalized': True,
        'scale': True
    }
    mod = "gex"
    processor = Processor(config, mod)
    result, inverse_function = processor.fit_transform(data)
    assert np.abs(result.mean()) < 0.01
    compare(inverse_function(result), data.X.toarray())

def __test_4():
    from lab_scripts.data import dataloader
    data = dataloader.load_data('mp/official/gex_to_adt')['train_mod1']
    config = {
        'use_normalized': False,
        'scale': True
    }
    mod = "gex"
    processor = Processor(config, mod)
    result, inverse_function = processor.fit_transform(data)
    assert np.abs(result.mean()) < 0.01
    compare(inverse_function(result), data.X.toarray())

def __test_5():
    from lab_scripts.data import dataloader
    data = dataloader.load_data('mp/official/gex_to_adt')['train_mod2']
    config = {
        'use_normalized': True,
        'scale': True
    }
    mod = "adt"
    processor = Processor(config, mod)
    result, inverse_function = processor.fit_transform(data)
    assert np.abs(result.mean()) < 0.01
    compare(inverse_function(result), data.X.toarray()) 


if __name__=='__main__':
    print('Running tests...')
    __test_1()
    __test_2()
    __test_3()
    __test_4()
    __test_5()
    print('Done')