import logging
from typing import Callable, List, Sequence

import anndata as ad
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lab_scripts.metrics import mp
from lab_scripts.utils import utils
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
        self.use_mmd = config['use_mmd_loss']
        self.mmd_lambda = config['mmd_lambda']

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

    def forward(self, first, second, batch_idx = None):
        first_latent = self.first_encoder(first)
        first_to_first = self.first_decoder(first_latent)
        first_to_second = self.second_decoder(first_latent)

        second_latent = self.second_encoder(second)
        second_to_first = self.first_decoder(second_latent)
        second_to_second = self.second_decoder(second_latent)

        if batch_idx is not None:
            mmd_loss = calculate_mmd_loss(first_latent, batch_idx)
            mmd_loss += calculate_mmd_loss(second_latent, batch_idx)
            return first_to_first, first_to_second, second_to_first, second_to_second, mmd_loss
        else:
            return first_to_first, first_to_second, second_to_first, second_to_second

    def training_step(self, batch, _):
        inputs, targets, batch_idx = batch
        first, second = inputs
        if self.use_mmd:
            first_to_first, first_to_second, second_to_first, second_to_second, mmd_loss = self(
                first, second, batch_idx
            )
            mmd_loss = self.mmd_lambda * mmd_loss
            self.log("mmd_loss", mmd_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        else:
            first_to_first, first_to_second, second_to_first, second_to_second = self(
                first, second
            )
            mmd_loss = 0.0
        first_target, second_target = targets
        first_to_first_loss = self.loss(first_to_first, first_target)
        second_to_first_loss = self.loss(second_to_first, first_target)
        first_loss = first_to_first_loss + second_to_first_loss

        first_to_second_loss = self.loss(first_to_second, second_target)
        second_to_second_loss = self.loss(second_to_second, second_target)
        second_loss = first_to_second_loss + second_to_second_loss
        loss = first_loss + second_loss + mmd_loss * self.mmd_lambda
        return loss


    def predict_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        first, second = inputs
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
        first_true_target=None,
        second_inverse=None,
        second_true_target=None,
    ):
        self.test_dataloader = test_dataloader
        self.first_inverse = first_inverse
        self.first_true_target = first_true_target
        self.second_inverse = second_inverse
        self.second_true_target = second_true_target

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
                first_to_second, self.second_inverse, self.second_true_target
            )
            pl_module.log("true_first_to_second", metric, prog_bar=True)

        if self.first_inverse is not None:
            metric = calculate_metric(
                second_to_first, self.first_inverse, self.first_true_target
            )
            pl_module.log("true_second_to_first", metric, prog_bar=True)


def calculate_mmd_loss(X, batch_idx):
    mmd_loss = 0.0
    reference_batch = X[batch_idx == 0]
    reference_mmd = mmd_for_two_batches(reference_batch, reference_batch)
    unique_batches = torch.unique(batch_idx)
    for batch in unique_batches:
        if batch == 0:
            continue
        other_batch = X[batch_idx == batch]
        loss = reference_mmd
        loss -= 2 * mmd_for_two_batches(reference_batch, other_batch)
        loss += mmd_for_two_batches(other_batch, other_batch)
        mmd_loss += loss

    mmd_loss /= X.shape[1]  # normalize for embedding length
    return -mmd_loss


def mmd_for_two_batches(first, second):
    result = 0.0
    for first_row in first:
        diff = second - first_row
        result += (diff**2).sum()  # squared distance between first_row and each row
    result = result / (first.shape[0] * second.shape[0])
    return result


def _generate_sample(loc, std, shape):
    first_distribution = torch.distributions.normal.Normal(loc, std)
    return first_distribution.sample(shape)

if __name__=='__main__':
    print('Testing calculate MMD loss...')
    utils.set_deafult_seed()
    first = _generate_sample(0.0, 1.0, [50, 20])
    second = _generate_sample(10.0, 1.0, [100, 20])
    third = _generate_sample(10.0, 1.0, [200, 20])
    X = torch.cat([first, second, third], dim=0)
    batch_idx = []
    for _ in range(first.shape[0]):
        batch_idx.append(0)
    for _ in range(second.shape[0]):
        batch_idx.append(1)
    for _ in range(third.shape[0]):
        batch_idx.append(2)
    batch_idx = torch.tensor(batch_idx)  # type: ignore
    new_idx = torch.randperm(X.shape[0])
    X = X[new_idx]
    batch_idx = batch_idx[new_idx]  # type: ignore
    print(calculate_mmd_loss(X, batch_idx))


