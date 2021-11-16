from collections import OrderedDict

import numpy as np
import plotly.express as px
import pytorch_lightning as pl
import torch
from lab_scripts.metrics import mm
from lab_scripts.models.common import plugins
from torch import nn
from torch.nn import functional as F


def construct_net(dims, activation_name: str, latent_dim: int, dropout_pos, dropout, batchnorm_pos):
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
        if i in dropout_pos:
            net.append((f"{i}_Dropout", nn.Dropout(dropout)))  # type: ignore
        if i in batchnorm_pos:
            net.append((f"{i}_Batchnorm", nn.BatchNorm1d(dims[i+1])))  # type: ignore
    net.append((f"{len(dims) - 1}_Linear", nn.Linear(dims[-1], latent_dim)))
    return nn.Sequential(OrderedDict(net))


class Clip(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.first_net = construct_net(
            config["first_dim"],
            config["activation"],
            config["latent_dim"],
            config["first_dropout"],
            config["dropout"],
            config['first_batchnorm'],
        )
        self.second_net = construct_net(
            config["second_dim"],
            config["activation"],
            config["latent_dim"],
            config["second_dropout"],
            config["dropout"],
            config['second_batchnorm'],
        )
        self.lr = config["lr"]
        self.attack = config["attack"]
        self.sustain = config["sustain"]
        self.release = config["release"]
        self.train_per_batch = config["train_per_batch"]
        self.enter_dropout = config["enter_dropout"]
        if self.enter_dropout > 0.0:
            self.dropout = nn.Dropout(self.enter_dropout)
        if config["train_temperature"] == -1:
            self.learn_temperature = True
            self.first_to_temperature = construct_net(
                config['first_dim'],
                config['activation'],
                1,
                config['first_dropout'],
                config['dropout'],
                config['first_batchnorm']
            )
            self.second_to_temperature = construct_net(
                config['second_dim'],
                config['activation'],
                1,
                config['second_dropout'],
                config['dropout'],
                config['second_batchnorm']
            )
        else:
            self.learn_temperature = False
            self.register_buffer(
                "temperature", torch.tensor(config["train_temperature"])
            )
        self.l2_lambda = config["l2_lambda"]

    def forward(self, first, second):
        first_embed = self.first_net(first)
        first_embed = first_embed / torch.linalg.norm(first_embed, dim=1, keepdim=True)
        second_embed = self.second_net(second)
        second_embed = second_embed / torch.linalg.norm(
            second_embed, dim=1, keepdim=True
        )
        if self.learn_temperature:
            first_temp = self.first_to_temperature(first)
            second_temp = self.second_to_temperature(second)
            temp = (first_temp + second_temp) / 2.0
            temp = torch.sigmoid(temp) * 10.0
            return first_embed, second_embed, temp
        else:
            return first_embed, second_embed

    def training_step(self, batch, batch_n):
        first, second, batch_idx = batch
        if self.enter_dropout > 0.0:
            first = self.dropout(first)
            second = self.dropout(second)

        if self.learn_temperature:
            first_embed, second_embed, temp = self(first, second)
        else:
            first_embed, second_embed = self(first, second)

        if self.train_per_batch:
            loss = 0.0
            for b_index in torch.unique(batch_idx):
                idx = batch_idx == b_index
                if self.learn_temperature:
                    loss += self.calculate_loss(first_embed[idx], second_embed[idx], temp[idx])
                else:
                    loss += self.calculate_loss(first_embed[idx], second_embed[idx])
        else:
            if self.learn_temperature:
                loss = self.calculate_loss(first_embed, second_embed, temp)
            else:
                loss = self.calculate_loss(first_embed, second_embed)
                
        self.log("train_loss", loss, logger=True, prog_bar=False)
        if self.learn_temperature:
            self.log("temp", temp.mean(), logger=True, prog_bar=False)
        return loss

    def predict_step(self, batch, batch_n):
        first, second, batch_idx = batch
        return self(first, second)

    def calculate_loss(self, first_embed, second_embed, temp=None):
        if temp is not None:
            logits = (first_embed @ second_embed.t()) * torch.exp(temp)
        else:
            logits = (first_embed @ second_embed.t()) * torch.exp(self.temperature)
        labels = torch.arange(first_embed.shape[0], device=self.device)
        first_loss = F.cross_entropy(logits, labels)
        second_loss = F.cross_entropy(logits.t(), labels)
        return (first_loss + second_loss) / 2.0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2_lambda
        )

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
    def __init__(self, dataset, temperature, prefix, log_top=None, log_preds=False):
        self.dataset = dataset
        self.temperature = temperature
        self.prefix = prefix
        self.log_top = log_top
        if isinstance(self.log_top, int):
            self.log_top = [self.log_top]
        self.log_preds = log_preds

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        device = pl_module.device
        first_embed = []
        second_embed = []
        batch_idx = []
        temp = []
        for i, batch in enumerate(self.dataset):
            with torch.no_grad():
                first, second, idx = batch
                preds = pl_module(first.to(device), second.to(device))
                if pl_module.learn_temperature:
                    first_e, second_e, temp_e = preds
                    temp.append(temp_e.cpu())
                else:
                    first_e, second_e = preds 
                first_embed.append(first_e.cpu())
                second_embed.append(second_e.cpu())
                batch_idx.append(idx)
        first_embed = torch.cat(first_embed, dim=0)
        second_embed = torch.cat(second_embed, dim=0)
        batch_idx = torch.cat(batch_idx)
        temp = torch.cat(temp, dim=0)
        n = first_embed.shape[0]
        similiarity = (first_embed @ second_embed.t()) * np.exp(
            temp * self.temperature
        )
        unique_batches = torch.unique(batch_idx)
        for batch in unique_batches:
            idx = batch_idx == batch
            similiarity[idx][:, ~idx] = -1e9
        final_predictions = torch.softmax(similiarity, dim=1)
        init = torch.eye(n)
        score = mm.calculate_target(final_predictions.numpy(), init.numpy())
        pl_module.log(self.prefix, score, logger=True, prog_bar=False)

        logger = trainer.logger
        if logger is None:
            return
        if self.prefix == 'test':
            fig = px.histogram(temp.numpy(), nbins=20)
            trainer.logger.experiment.log({"temperature": fig})

        if self.log_top:
            (_, idx) = torch.sort(similiarity, descending=True)
            idx = idx.numpy()
            for top in self.log_top:
                good = 0
                frac_top = 0.01 * top

                for i in range(n):
                    good += i in idx[i][: int(frac_top * n)]
                prog_bar = top == 5
                self.log(
                    f"{self.prefix}_top{top}", good / n, logger=True, prog_bar=prog_bar
                )
