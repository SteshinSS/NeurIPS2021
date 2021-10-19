from collections import OrderedDict

import pytorch_lightning as pl
import torch
from lab_scripts.models.common import plugins
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from torch import nn
from torch.distributions.negative_binomial import NegativeBinomial
from torch.nn import functional as F


class Encoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["encoder"]
        dims.insert(0, config["input_features"])
        dims.append(latent_dim)
        activation = plugins.get_activation(config["activation"])

        net = []
        for i in range(len(dims) - 1):
            net.append((f"{i}_Linear", nn.Linear(dims[i], dims[i + 1])))
            net.append((f"{i}_Activation", activation))  # type: ignore
            if i - 1 in config["encoder_bn"]:
                net.append((f"{i}_BatchNorm", nn.BatchNorm1d(dims[i + 1])))  # type: ignore
        self.net = nn.Sequential(OrderedDict(net))
        plugins.init(self.net, config["activation"])

    def forward(self, x):
        return self.net(x)


class VAEEncoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["encoder"]
        dims.insert(0, config["input_features"])
        activation = plugins.get_activation(config["activation"])
        batch_norm_pos = config["encoder_bn"]

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
        activation = plugins.get_activation(config["activation"])
        batchnorm_pos = config["decoder_bn"]

        net = []
        for i in range(len(dims) - 1):
            net.append((f"{i}_Linear", nn.Linear(dims[i], dims[i + 1])))
            if i + 1 != len(dims) - 1:
                net.append((f"{i}_Activation", activation))
            if i - 1 in batchnorm_pos:
                net.append((f"{i}_BatchNorm", nn.BatchNorm1d(dims[i + 1])))  # type: ignore
        self.net = nn.Sequential(OrderedDict(net))
        plugins.init(self.net, config["activation"])

    def forward(self, x):
        return self.net(x)


class NBDecoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["decoder"]
        dims.insert(0, latent_dim)
        activation = plugins.get_activation(config["activation"])
        self.activation = activation

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)
        self.net = nn.Sequential(*net)

        self.to_r = self.create_last_sequential(
            dims[-1], config["target_features"], self.activation
        )
        self.to_p = self.create_last_sequential(
            dims[-1], config["target_features"], self.activation
        )

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
        activation = plugins.get_activation(config["activation"])

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
        zinb_distribution = ZeroInflatedNegativeBinomial(
            zinb_r, probs=zinb_p, gate=dropout
        )
        return zinb_distribution.mean


class LogNormDecoder(pl.LightningModule):
    def __init__(self, config: dict, latent_dim: int):
        super().__init__()
        dims = config["decoder"]
        dims.insert(0, latent_dim)
        activation = plugins.get_activation(config["activation"])
        self.activation = activation

        net = []
        for i in range(len(dims) - 1):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(activation)
        self.net = nn.Sequential(*net)

        self.to_loc = self.create_last_sequential(
            dims[-1], config["target_features"], self.activation
        )
        self.to_scale = self.create_last_sequential(
            dims[-1], config["target_features"], self.activation
        )

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
    elif loss_name == "lognorm":
        return LogNormDecoder
    else:
        raise NotImplementedError
