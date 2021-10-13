import logging
from typing import Callable, List

import anndata as ad
import pytorch_lightning as pl
import torch
from lab_scripts.metrics import mp
from lab_scripts.utils import utils
from torch import nn
from torch.distributions.negative_binomial import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("x_autoencoder")


def zinb_loss(predicted_parameters, targets):
    zinb_r, zinb_p, dropout = predicted_parameters
    zinb_distribution = ZeroInflatedNegativeBinomial(zinb_r, probs=zinb_p, gate=dropout)
    log_loss = zinb_distribution.log_prob(targets).mean()
    return -log_loss

def convert_zinb_parameters_to_mean(parameters):
    zinb_r, zinb_p, dropout = parameters
    zinb_distribution = ZeroInflatedNegativeBinomial(zinb_r, probs=zinb_p, gate=dropout)
    return zinb_distribution.mean


def nb_loss(predicted_parameters, targets):
    nb_r, nb_p = predicted_parameters
    nb_distribution = NegativeBinomial(nb_r, nb_p)
    log_loss = nb_distribution.log_prob(targets).mean()
    return -log_loss


def convert_nb_parameters_to_mean(parameters):
    nb_r, nb_p = parameters
    nb_distribution = NegativeBinomial(nb_r, nb_p)
    return nb_distribution.mean


def get_loss(loss_name: str):
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "nb":
        return nb_loss
    elif loss_name == 'zinb':
        return zinb_loss


def get_activation(activation_name: str):
    if activation_name == "leaky_relu":
        return nn.LeakyReLU()


class Encoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["encoder"]
        dims.insert(0, config["input_features"])
        dims.append(latent_dim)
        activation = get_activation(config["activation"])

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)  # type: ignore
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MSEDecoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["decoder"]
        dims.insert(0, latent_dim)
        dims.append(config["target_features"])
        activation = get_activation(config["activation"])

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            if i + 1 != len(dims) - 1:
                net.append(activation)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class NBDecoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["decoder"]
        dims.insert(0, latent_dim)
        activation = get_activation(config["activation"])

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)
        self.net = nn.Sequential(*net)

        self.to_r = nn.Linear(dims[-1], config["target_features"])
        self.to_p = nn.Linear(dims[-1], config["target_features"])

    def forward(self, x):
        eps = 1e-8
        y = self.net(x)
        nb_r = torch.exp(self.to_r(y)) + eps
        nb_p = torch.sigmoid(self.to_p(y))
        return nb_r, nb_p * 0.999

class ZINBDecoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["decoder"]
        dims.insert(0, latent_dim)
        activation = get_activation(config["activation"])

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)
        self.net = nn.Sequential(*net)

        self.to_r = nn.Linear(dims[-1], config["target_features"])
        self.to_p = nn.Linear(dims[-1], config["target_features"])
        self.to_dropout = nn.Linear(dims[-1], config["target_features"])

    def forward(self, x):
        eps = 1e-8
        y = self.net(x)
        nb_r = torch.exp(self.to_r(y)) + eps
        nb_p = torch.sigmoid(self.to_p(y))
        dropout = torch.sigmoid(self.to_dropout(y))
        return nb_r, nb_p * 0.999, dropout


def get_decoder(loss_name: str):
    if loss_name == "nb":
        return NBDecoder
    elif loss_name == "mse":
        return MSEDecoder
    elif loss_name == 'zinb':
        return ZINBDecoder
    else:
        raise NotImplementedError


class X_autoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.lr = config["lr"]
        self.use_mmd = config["use_mmd_loss"]
        self.mmd_lambda = config["mmd_lambda"]

        first_config = config["first"]
        self.first_config = first_config
        self.first_loss = get_loss(first_config["loss"])
        self.first_encoder = Encoder(first_config, config["latent_dim"])
        self.first_decoder = get_decoder(first_config["loss"])(
            first_config, config["latent_dim"]
        )

        second_config = config["second"]
        self.second_config = second_config
        self.second_loss = get_loss(second_config["loss"])
        self.second_encoder = Encoder(second_config, config["latent_dim"])
        self.second_decoder = get_decoder(second_config["loss"])(
            second_config, config["latent_dim"]
        )

    def forward(self, first, second):
        first_latent = self.first_encoder(first)
        first_to_first = self.first_decoder(first_latent)
        first_to_second = self.second_decoder(first_latent)

        second_latent = self.second_encoder(second)
        second_to_first = self.first_decoder(second_latent)
        second_to_second = self.second_decoder(second_latent)

        result = {
            "first_latent": first_latent,
            "first_to_first": first_to_first,
            "first_to_second": first_to_second,
            "second_latent": second_latent,
            "second_to_first": second_to_first,
            "second_to_second": second_to_second,
        }
        return result

    def training_step(self, batch, _):
        if len(batch) == 2:
            inputs, batch_idx = batch
            targets = inputs
        elif len(batch) == 3:
            inputs, targets, batch_idx = batch
        first, second = inputs
        result = self(first, second)
        loss = self.calculate_loss(result, targets, batch_idx)
        self.log("train_loss", loss, on_step=True, prog_bar=False, logger=True)
        return loss

    def calculate_loss(self, result: dict, targets, batch_idx):
        first_target, second_target = targets
        loss = 0.0
        loss += self.first_loss(result["first_to_first"], first_target)
        loss += self.first_loss(result["second_to_first"], first_target)
        loss += self.second_loss(result["first_to_second"], second_target)
        loss += self.second_loss(result["second_to_second"], second_target)
        if self.use_mmd:
            mmd_loss = calculate_mmd_loss(result["first_latent"], batch_idx)
            mmd_loss += calculate_mmd_loss(result["second_latent"], batch_idx)
            mmd_loss *= self.mmd_lambda
            loss += mmd_loss
            self.log(
                "mmd",
                mmd_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        return loss

    def predict_step(self, batch, _):
        if len(batch) == 2:
            inputs, _ = batch
        elif len(batch) == 3:
            inputs, _, _ = batch
        first, second = inputs
        result = self(first, second)
        first_to_second = result["first_to_second"]
        second_to_first = result["second_to_first"]
        if self.second_config["loss"] == "nb":
            first_to_second = convert_nb_parameters_to_mean(first_to_second)
        elif self.second_config['loss'] == "zinb":
            first_to_second = convert_zinb_parameters_to_mean(first_to_second)
        if self.first_config["loss"] == "nb":
            second_to_first = convert_nb_parameters_to_mean(second_to_first)
        elif self.first_config['loss'] == 'zinb':
            second_to_first = convert_zinb_parameters_to_mean(second_to_first)
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
                first_to_second,
                self.second_inverse,
                self.second_true_target,
            )
            pl_module.log("true_1_to_2", metric, prog_bar=True)

        if self.first_inverse is not None:
            metric = calculate_metric(
                second_to_first, self.first_inverse, self.first_true_target
            )
            pl_module.log("true_2_to_1", metric, prog_bar=True)


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

    return -mmd_loss


def mmd_for_two_batches(first, second):
    result = 0.0
    if first.shape[0] == 0 or second.shape[0] == 0:
        return result
    for first_row in first:
        diff = second - first_row
        dist = (diff ** 2).sum(axis=1)  # **(0.5)
        result += dist.sum()
        # result += (diff ** 2).sum()  # squared distance between first_row and each row
    result = result / (first.shape[0] * second.shape[0])
    return result


def _generate_sample(loc, std, shape):
    first_distribution = torch.distributions.normal.Normal(loc, std)
    return first_distribution.sample(shape)


if __name__ == "__main__":
    print("Testing calculate MMD loss...")
    utils.set_deafult_seed()
    first = _generate_sample(0.0, 0.01, [50, 20])
    second = _generate_sample(10.0, 0.01, [100, 20])
    third = _generate_sample(4.0, 0.01, [200, 20])
    X = torch.cat([first, second], dim=0)
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
