import anndata as ad
import pytorch_lightning as pl
import torch
from lab_scripts.data import dataloader
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc

class OmicsDataset(Dataset):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        self.first = first
        self.second = second

    def __len__(self):
        return self.first.shape[0]

    def __getitem__(self, idx):
        return (self.first[idx], self.second[idx])


class GexAdtData(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.data = dataloader.load_data(config["data"])
        self.batch_size = config["batch_size"]
        self.normalize = config['normalize']
        self.top_n_genes = config['top_n_genes']

    def setup(self, stage=None):
        sc.pp.highly_variable_genes(self.data['train_mod1'], n_top_genes=self.top_n_genes)
        highly_variable_idx = self.data['train_mod1'].var['highly_variable']
        self.data['test_mod1'] = self.data['test_mod1'][:, highly_variable_idx]
        self.data['train_mod1'] = self.data['train_mod1'][:, highly_variable_idx]

        self.scaler_mod1 = MinMaxScaler()
        train_mod1 = self._get_matrix(self.data["train_mod1"])
        train_mod1 = self.scaler_mod1.fit_transform(train_mod1)
        train_mod1 = torch.tensor(train_mod1, dtype=torch.float32)

        self.scaler_mod2 = MinMaxScaler()
        train_mod2 = self._get_matrix(self.data["train_mod2"])
        train_mod2 = self.scaler_mod2.fit_transform(train_mod2)
        train_mod2 = torch.tensor(train_mod2, dtype=torch.float32)
        self.train = OmicsDataset(train_mod1, train_mod2)

        
        val_mod1 = self._get_matrix(self.data["test_mod1"])
        val_mod1 = self.scaler_mod1.transform(val_mod1)
        val_mod1 = torch.tensor(val_mod1, dtype=torch.float32)

        val_mod2 = self._get_matrix(self.data["test_mod2"])
        val_mod2 = self.scaler_mod2.transform(val_mod2)
        val_mod2 = torch.tensor(val_mod2, dtype=torch.float32)
        self.val = OmicsDataset(val_mod1, val_mod2)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
    
    def _get_matrix(self, adata: ad.AnnData):
        if self.normalize:
            return adata.X.toarray()
        else:
            return adata.layers['counts'].toarray()
    
    def unscale_data(self, first, second):
        first = self.scaler_mod1.inverse_transform(first)
        second = self.scaler_mod2.inverse_transform(second)
        return first, second


if __name__=='__main__':
    config = {
        'batch_size': 128,
        'normalize': True,
        'data': 'mp/official/gex_to_adt'
    }
    data = GexAdtData(config)
    data.setup('test')
    train = data.train_dataloader()
    for a in train:
        print(a[0].dtype)
        break