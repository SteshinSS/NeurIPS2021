from collections import OrderedDict

import numpy as np
import anndata as ad
import pytorch_lightning as pl
import torch
from lab_scripts.metrics.je import metrics
from lab_scripts.models.common import plugins
from pytorch_lightning.callbacks import Callback
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer


class Encoder(pl.LightningModule):
    def __init__(self, dims, activation_name: str):
        super().__init__()
        self.net = construct_net(dims, activation_name)

    def forward(self, x):
        return self.net(x)


class Decoder(pl.LightningModule):
    def __init__(self, dims, activation_name: str, first_dim):
        super().__init__()
        rev_dims = dims[1::][::-1]
        rev_dims.insert(0, first_dim)
        self.net = construct_net(rev_dims, activation_name)
        self.to_last = nn.Linear(dims[1], dims[0])

    def forward(self, x):
        y = self.net(x)
        return self.to_last(y)


def construct_net(dims, activation_name: str):
    activation = plugins.get_activation(activation_name)

    net = []
    for i in range(len(dims) - 1):
        net.append(
            (
                f"{i}_Linear",
                nn.Linear(dims[i], dims[i + 1]),
            )
        )
        net.append((f"{i}_Actiavtion", activation))
    return nn.Sequential(OrderedDict(net))


class JEAutoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.lr = config["lr"]
        self.attack = config["attack"]
        self.sustain = config["sustain"]
        self.release = config["release"]

        self.first_encoder = Encoder(config["first_dim"], config["activation"])
        self.second_encoder = Encoder(config["second_dim"], config["activation"])
        self.net = construct_net(config["common_dim"], config["activation"])
        self.first_decoder = Decoder(
            config["first_dim"], config["activation"], config["common_dim"][-1]
        )
        self.second_decoder = Decoder(
            config["second_dim"], config["activation"], config["common_dim"][-1]
        )

    def forward(self, first, second):
        first_embed = self.first_encoder(first)
        second_embed = self.second_encoder(second)
        features = torch.cat([first_embed, second_embed], dim=1)
        embeddings = self.net(features)
        return embeddings

    def training_step(self, batch, batch_n):
        first, second, batch_idx = batch
        embeddings = self(first, second)
        first_reconstruction = self.first_decoder(embeddings)
        second_reconstruction = self.second_decoder(embeddings)
        loss = 0.0
        loss += self.reconstruction_loss(
            first, first_reconstruction, second, second_reconstruction
        )

        self.log("train_loss", loss, logger=True, prog_bar=False)
        return loss

    def reconstruction_loss(
        self, first, first_reconstruction, second, second_reconstruction
    ):
        loss = F.mse_loss(first, first_reconstruction)
        loss += F.mse_loss(second, second_reconstruction)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def lr_foo(epoch):
            epoch = epoch + 1
            if epoch < self.attack:
                lr_scale = epoch / self.attack
            elif epoch < (self.sustain + self.attack):
                lr_scale = 1.0
            else:
                lr_scale = 1.0 - (epoch - self.sustain - self.attack) / self.release
            return lr_scale

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class TargetCallback(pl.Callback):
    def __init__(self, solution: ad.AnnData, dataset, frequency: int):
        self.solution = solution
        self.dataset = dataset
        self.frequency = frequency
        self.current_epoch = 0
    
    def on_train_epoch_end(self, trainer, pl_module):
        logger = trainer.logger
        if not logger:
            return
        self.current_epoch += 1
        if self.current_epoch % self.frequency != 0:
            return
        pl_module.eval()
        embeddings = []
        device = pl_module.device
        with torch.no_grad():
            for i, batch in enumerate(self.dataset):
                first, second, batch_idx = batch
                embeddings.append(pl_module(first.to(device), second.to(device)).cpu())
        embeddings = torch.cat(embeddings, dim=0)
        prediction = metrics.create_anndata(self.solution, embeddings.numpy())
        all_metrics = metrics.calculate_metrics(prediction, self.solution)
        logger.experiment.log(all_metrics)
        

    

