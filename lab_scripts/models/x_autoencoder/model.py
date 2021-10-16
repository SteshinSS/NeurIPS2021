import logging
from typing import Callable, List

import anndata as ad
from numpy import diff
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from lab_scripts.metrics import mp
from lab_scripts.utils import utils
import numpy as np
from torch import nn
from torch.distributions.negative_binomial import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from torch.distributions.log_normal import LogNormal
import wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("x_autoencoder")

def lognorm_loss(predictied_parameters, targets):
    eps = 1e-6
    loc, scale = predictied_parameters
    exp_distribution = LogNormal(loc, scale)
    exp_loss = exp_distribution.log_prob(targets + eps).mean()
    return -exp_loss

def zinb_loss(predicted_parameters, targets):
    zinb_r, zinb_p, dropout = predicted_parameters
    zinb_distribution = ZeroInflatedNegativeBinomial(zinb_r, probs=zinb_p, gate=dropout)
    log_loss = zinb_distribution.log_prob(targets).mean()
    return -log_loss


def nb_loss(predicted_parameters, targets):
    nb_r, nb_p = predicted_parameters
    nb_distribution = NegativeBinomial(nb_r, nb_p)
    log_loss = nb_distribution.log_prob(targets).mean()
    return -log_loss


def get_loss(loss_name: str):
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "nb":
        return nb_loss
    elif loss_name == "zinb":
        return zinb_loss
    elif loss_name == 'lognorm':
        return lognorm_loss


def get_activation(activation_name: str):
    if activation_name == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_name == "tanh":
        return nn.Tanh()


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
            if i - 1 in config['dropout_pos']:
                net.append(nn.Dropout(config['dropout']))  # type: ignore
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class VAEEncoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["encoder"]
        dims.insert(0, config["input_features"])
        activation = get_activation(config["activation"])
        batch_norm_pos = config["batchnorm_pos"]

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)  # type: ignore
            if i - 1 in batch_norm_pos:
                net.append(nn.BatchNorm1d(dims[i + 1]))  # type: ignore
        self.net = nn.Sequential(*net)

        self.to_mean = nn.Linear(dims[-1], latent_dim)
        self.to_var = nn.Linear(dims[-1], latent_dim)

    def forward(self, x):
        y = self.net(x)
        mu = self.to_mean(y)
        logvar = self.to_var(y)
        self.kl_loss = self.calculate_kl_loss(mu, logvar)
        z = self.reparameterize(mu, logvar)
        return z

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + torch.exp(0.5 * logvar) * eps

    def calculate_kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss / torch.numel(mu)


def get_encoder(is_vae: bool):
    if is_vae:
        return VAEEncoder
    else:
        return Encoder


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
        self.activation = activation

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)
        self.net = nn.Sequential(*net)

        self.to_r = self.create_last_sequential(dims[-1], config['target_features'], self.activation)
        self.to_p = self.create_last_sequential(dims[-1], config['target_features'], self.activation)

    def forward(self, x):
        eps = 1e-6
        y = self.net(x)
        nb_r = F.softplus(self.to_r(y)) + eps
        nb_p = torch.sigmoid(self.to_p(y))
        return nb_r, nb_p * (1 - eps)

    def create_last_sequential(self, last_dim, target_features, activation):
        sequential = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.Tanh(),
            nn.Linear(last_dim, target_features),
        )
        return sequential
    
    def to_prediction(self, parameters):
        nb_r, nb_p = parameters
        nb_distribution = NegativeBinomial(nb_r, nb_p)
        return nb_distribution.mean


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

        self.to_r = self.create_last_sequential(
            dims[-1], config["target_features"], activation
        )
        self.to_p = self.create_last_sequential(
            dims[-1], config["target_features"], activation
        )
        self.to_dropout = self.create_last_sequential(
            dims[-1], config["target_features"], activation
        )

    def forward(self, x):
        eps = 1e-6
        y = self.net(x)
        nb_r = F.softplus(self.to_r(y)) + eps
        nb_p = torch.sigmoid(self.to_p(y))
        dropout = torch.sigmoid(self.to_dropout(y))
        return nb_r, nb_p * (1 - eps), dropout * (1 - eps)

    def create_last_sequential(self, last_dim, target_features, activation):
        sequential = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.Tanh(),
            nn.Linear(last_dim, target_features),
        )
        return sequential
    
    def to_prediction(self, parameters):
        zinb_r, zinb_p, dropout = parameters
        zinb_distribution = ZeroInflatedNegativeBinomial(zinb_r, probs=zinb_p, gate=dropout)
        return zinb_distribution.mean

class LogNormDecoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["decoder"]
        dims.insert(0, latent_dim)
        activation = get_activation(config["activation"])
        self.activation = activation

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)
        self.net = nn.Sequential(*net)

        self.to_loc = self.create_last_sequential(dims[-1], config['target_features'], self.activation)
        self.to_scale = self.create_last_sequential(dims[-1], config['target_features'], self.activation)

    def forward(self, x):
        eps = 1e-6
        y = self.net(x)
        loc = self.to_loc(y)
        scale = F.softplus(self.to_scale(y)) + eps
        return loc, scale

    def create_last_sequential(self, last_dim, target_features, activation):
        sequential = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.Tanh(),
            nn.Linear(last_dim, target_features),
        )
        return sequential
    
    def to_prediction(self, parameters):
        loc, scale = parameters
        return loc


def get_decoder(loss_name: str):
    if loss_name == "nb":
        return NBDecoder
    elif loss_name == "mse":
        return MSEDecoder
    elif loss_name == "zinb":
        return ZINBDecoder
    elif loss_name == 'lognorm':
        return LogNormDecoder
    else:
        raise NotImplementedError


class BatchCritic(pl.LightningModule):
    def __init__(self, latent_dim: int, total_batches: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, total_batches),
            nn.Softmax()
        )
        self.bias = np.log(total_batches)
    
    def forward(self, x):
        return self.net(x)

    def get_loss(self, predicted, true):
        cross_entropy = F.cross_entropy(predicted, true)
        return max(self.bias - cross_entropy, 0.0)

def get_critic(critic_name: str):
    if critic_name == 'discriminator':
        return BatchCritic


class X_autoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.lr = config["lr"]
        self.use_mmd = config["use_mmd_loss"]
        self.mmd_lambda = config["mmd_lambda"]
        self.use_sim_loss = config["use_sim_loss"]
        self.sim_lambda = config["sim_lambda"]
        self.use_critic = config["use_critic"]
        self.critic_lambda = config["critic_lambda"]
        self.critic_type = config['critic_type']
        self.patience = config["patience"]
        self.vae = config["vae"]

        first_config = config["first"]
        self.first_config = first_config
        self.first_loss = get_loss(first_config["loss"])
        self.first_encoder = get_encoder(self.vae)(first_config, config["latent_dim"])
        self.first_decoder = get_decoder(first_config["loss"])(
            first_config, config["latent_dim"]
        )

        second_config = config["second"]
        self.second_config = second_config
        self.second_loss = get_loss(second_config["loss"])
        self.second_encoder = get_encoder(self.vae)(second_config, config["latent_dim"])
        self.second_decoder = get_decoder(second_config["loss"])(
            second_config, config["latent_dim"]
        )

        if self.use_critic:
            self.first_critic = get_critic(self.critic_type)(config['latent_dim'], config['total_batches'])
            self.second_critic = get_critic(self.critic_type)(config['latent_dim'], config['total_batches'])

        first_to_second = config["first_to_second"]
        self.first_weight = first_to_second / (1 + first_to_second)
        self.second_weight = 1 / (1 + first_to_second)


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

        if self.use_critic:
            result['first_critic'] = self.first_critic(first_latent)
            result['second_critic'] = self.second_critic(second_latent)

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
        first_to_first = (
            self.first_loss(result["first_to_first"], first_target) * self.first_weight
        )
        self.log(
            "11",
            first_to_first,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        first_loss = first_to_first

        second_to_first = (
            self.first_loss(result["second_to_first"], first_target) * self.first_weight
        )
        self.log(
            "21",
            second_to_first,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        first_loss += second_to_first

        first_to_second = (
            self.second_loss(result["first_to_second"], second_target)
            * self.second_weight
        )
        self.log(
            "12",
            first_to_second,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        second_loss = first_to_second

        second_to_second = (
            self.second_loss(result["second_to_second"], second_target)
            * self.second_weight
        )
        self.log(
            "22",
            second_to_second,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        second_loss += second_to_second

        loss = first_loss + second_loss

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
        if self.vae:
            kl_loss = self.first_encoder.kl_loss + self.second_encoder.kl_loss
            self.log(
                "kl", kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
            )
            loss += kl_loss
        if self.use_sim_loss:
            sim_loss = F.mse_loss(
                result["first_latent"],
                result["second_latent"],
            )
            sim_loss *= self.sim_lambda
            self.log(
                "sim",
                sim_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
            loss += sim_loss  # type: ignore
        
        if self.use_critic:
            critic_loss = self.first_critic.get_loss(result['first_critic'], batch_idx)
            critic_loss += self.second_critic.get_loss(result['second_critic'], batch_idx)
            critic_loss *= self.critic_lambda
            self.log("critic", critic_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            loss += critic_loss

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
        if self.second_config['loss'] != 'mse':
            first_to_second = self.second_decoder.to_prediction(first_to_second)
        if self.first_config['loss'] != 'mse':
            second_to_first = self.first_decoder.to_prediction(second_to_first)
        return first_to_second, second_to_first

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=self.patience, verbose=True
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
            
        logger = trainer.logger
        if self.second_inverse is not None:
            predictions = torch.cat(first_to_second, dim=0).cpu().detach()
            predictions = self.second_inverse(predictions)
            metric = mp.calculate_target(predictions, self.second_true_target)
            pl_module.log("true_1_to_2", metric, prog_bar=True)

            difference = predictions - self.second_true_target.X.toarray()
            if logger:
                logger.experiment.log({
                    'second_difference': wandb.Histogram(difference)
                })

        if self.first_inverse is not None:
            predictions = torch.cat(second_to_first, dim=0).cpu().detach()
            predictions = self.first_inverse(predictions)
            metric = mp.calculate_target(predictions, self.first_true_target)
            pl_module.log("true_2_to_1", metric, prog_bar=True)

            difference = predictions - self.first_true_target.X.toarray()
            if logger:
                logger.experiment.log({
                    'first_difference': wandb.Histogram(difference)
                })


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
