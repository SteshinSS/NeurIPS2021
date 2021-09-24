import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data


class BaselineDataloader(data.Dataset):
    def __init__(self, gex_X, adt_X):
        self.gex_X = gex_X
        self.adt_X = adt_X

    def __len__(self):
        return self.gex_X.shape[0]

    def __getitem__(self, idx):
        return self.gex_X[idx], self.adt_X[idx]


class BaselineModel(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.lr = config["lr"]
        self.input_features = config["input_features"]
        self.output_features = config["output_features"]

        self.linear_1 = nn.Linear(self.input_features, 500)
        self.linear_2 = nn.Linear(500, 300)
        self.linear_3 = nn.Linear(300, self.output_features)

        self.dropout = None
        if config["use_dropout"]:
            self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.tanh(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear_2(x)
        x = torch.tanh(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear_3(x)
        return x

    def training_step(self, batch, batch_idx):
        gex, adt = batch
        adt_pred = self.forward(gex)
        loss = F.mse_loss(adt_pred, adt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        gex, adt = batch
        adt_pred = self.forward(gex)
        loss = F.mse_loss(adt_pred, adt)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.3, cooldown=7
        )
        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }
        return optimizer_config
