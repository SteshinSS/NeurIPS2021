import logging
from typing import Callable, List, Sequence

import anndata as ad
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lab_scripts.metrics import mp
from torch import nn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("x_autoencoder")


class FCNet(pl.LightningModule):
    def __init__(self, dims: Sequence, activation: Callable):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)


class Encoder(pl.LightningModule):
    def __init__(self, dims: Sequence, activation: Callable):
        super().__init__()
        self.net = FCNet(dims, activation)
        self.activation = activation

    def forward(self, x):
        x = self.net(x)
        return self.activation(x)


class Decoder(pl.LightningModule):
    def __init__(self, dims: Sequence, activation: Callable):
        super().__init__()
        self.net = FCNet(dims, activation)
        self.activation = activation

    def forward(self, x):
        x = self.net(x)
        return x


class X_autoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.lr = config["lr"]
        self.activation = config["activation"]
        if self.activation == "relu":
            activation = F.relu
        elif self.activation == "selu":
            activation = F.leaky_relu  # type: ignore
        elif self.activation == "tanh":
            activation = torch.tanh  # type: ignore
        self.loss = nn.MSELoss()

        first_layer_dims = config["first_dims"]
        first_layer_dims.append(config["latent_dim"])
        log.info("First encoder dims: %s", str(first_layer_dims))
        self.first_encoder = Encoder(first_layer_dims, activation)
        self.first_decoder = Decoder(list(reversed(first_layer_dims)), activation)

        second_layer_dims = config["second_dims"]
        second_layer_dims.append(config["latent_dim"])
        log.info("Second encoder dims: %s", str(second_layer_dims))
        self.second_encoder = Encoder(second_layer_dims, activation)
        self.second_decoder = Decoder(list(reversed(second_layer_dims)), activation)

    def forward(self, first, second):
        first_latent = self.first_encoder(first)
        first_to_first = self.first_decoder(first_latent)
        first_to_second = self.second_decoder(first_latent)

        second_latent = self.second_encoder(second)
        second_to_first = self.first_decoder(second_latent)
        second_to_second = self.second_decoder(second_latent)

        return first_to_first, first_to_second, second_to_first, second_to_second

    def training_step(self, batch, batch_idx):
        first, second = batch
        first_to_first, first_to_second, second_to_first, second_to_second = self(
            first, second
        )
        first_to_first_loss = self.loss(first_to_first, first)
        second_to_first_loss = self.loss(second_to_first, first)
        first_loss = first_to_first_loss + second_to_first_loss

        first_to_second_loss = self.loss(first_to_second, second)
        second_to_second_loss = self.loss(second_to_second, second)
        second_loss = first_to_second_loss + second_to_second_loss
        loss = first_loss + second_loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        first, second = batch
        first_to_first, first_to_second, second_to_first, second_to_second = self(
            first, second
        )
        loss = (
            self.loss(first_to_first, first)
            + self.loss(first_to_second, second)
            + self.loss(second_to_first, first)
            + self.loss(second_to_second, second)
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return first, second, first_to_second, second_to_first

    def predict_step(self, batch, batch_idx):
        first, second = batch
        first_to_first, first_to_second, second_to_first, second_to_second = self(
            first, second
        )
        return first_to_second, second_to_first

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=50, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            },
        }


def calculate_metric(
    raw_predictions: List, inverse_transform: Callable, target_dataset: ad.AnnData
):
    """Calculates target metric for modality prediction task

    Args:
        raw_predictions (List): one modality predictions
        inverse_transform (Callable):
        target_dataset (ad.AnnData):

    Returns:
        float: target metric
    """
    predictions = torch.cat(raw_predictions, dim=0).cpu().detach()
    predictions = inverse_transform(predictions)
    return mp.calculate_target(predictions, target_dataset)


class TargetCallback(pl.Callback):
    def __init__(
        self,
        test_dataloader,
        first_inverse=None,
        first_target=None,
        second_inverse=None,
        second_target=None,
    ):
        self.test_dataloader = test_dataloader
        self.first_inverse = first_inverse
        self.first_target = first_target
        self.second_inverse = second_inverse
        self.second_target = second_target

    def on_train_epoch_end(self, trainer, pl_module):
        first_to_second = []
        second_to_first = []
        for i, batch in enumerate(self.test_dataloader):
            first_to_second_batch, second_to_first_batch = pl_module.predict_step(
                batch, i
            )
            first_to_second.append(first_to_second_batch)
            second_to_first.append(second_to_first_batch)

        if self.second_inverse is not None:
            metric = calculate_metric(
                first_to_second, self.second_inverse, self.second_target
            )
            pl_module.log("true_first_to_second", metric, prog_bar=True)

        if self.first_inverse is not None:
            metric = calculate_metric(
                second_to_first, self.first_inverse, self.first_target
            )
            pl_module.log("true_second_to_first", metric, prog_bar=True)
