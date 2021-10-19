import logging
from typing import Callable, List

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from lab_scripts.metrics import mp
from lab_scripts.models.common import autoencoders, losses
from lab_scripts.utils import utils
from torch import nn
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("x_autoencoder")


class BatchCritic(pl.LightningModule):
    def __init__(self, latent_dim: int, total_batches: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, total_batches),
        )
        self.bias = np.log(total_batches)

    def forward(self, x):
        return self.net(x)

    def get_loss(self, predicted, true, shifted=False):
        cross_entropy = F.cross_entropy(predicted, true)
        if shifted:
            return max(self.bias - cross_entropy, 0.0)
        else:
            return cross_entropy


def get_critic(critic_name: str):
    if critic_name == "discriminator":
        return BatchCritic


class X_autoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.lr = config["lr"]
        self.attack = config["attack"]
        self.sustain = config["sustain"]
        self.release = config["release"]
        self.patience = config["patience"]
        self.vae = config["vae"]
        self.use_mmd = config["use_mmd_loss"]
        self.mmd_lambda = config["mmd_lambda"]
        self.use_sim_loss = config["use_sim_loss"]
        self.sim_lambda = config["sim_lambda"]
        self.use_critic = config["use_critic"]
        self.critic_lambda = config["critic_lambda"]
        self.critic_type = config["critic_type"]
        self.critic_lr = config["critic_lr"]
        self.critic_iterations = config["critic_iterations"]
        self.gradient_clip = config["gradient_clip"]
        self.balance_classes = config["balance_classes"]
        if self.balance_classes:
            self.register_buffer("batch_weights", torch.tensor(config["batch_weights"]))

        first_config = config["first"]
        self.first_config = first_config
        self.first_loss = losses.get_loss(first_config["loss"])
        self.first_encoder = autoencoders.get_encoder(self.vae)(
            first_config, config["latent_dim"]
        )
        self.first_decoder = autoencoders.get_decoder(first_config["loss"])(
            first_config, config["latent_dim"]
        )

        second_config = config["second"]
        self.second_config = second_config
        self.second_loss = losses.get_loss(second_config["loss"])
        self.second_encoder = autoencoders.get_encoder(self.vae)(
            second_config, config["latent_dim"]
        )
        self.second_decoder = autoencoders.get_decoder(second_config["loss"])(
            second_config, config["latent_dim"]
        )
        self.main_parameters = []
        self.main_parameters.extend(self.first_encoder.parameters())
        self.main_parameters.extend(self.first_decoder.parameters())
        self.main_parameters.extend(self.second_encoder.parameters())
        self.main_parameters.extend(self.second_decoder.parameters())

        if self.use_critic:
            self.first_critic = get_critic(self.critic_type)(
                config["latent_dim"], config["total_batches"]
            )
            self.second_critic = get_critic(self.critic_type)(
                config["latent_dim"], config["total_batches"]
            )
            self.critic_parameters = []
            self.critic_parameters.extend(self.first_critic.parameters())
            self.critic_parameters.extend(self.second_critic.parameters())

        self.automatic_optimization = False

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
            result["first_critic"] = self.first_critic(first_latent)
            result["second_critic"] = self.second_critic(second_latent)

        return result

    def critic_forward(self, first, second):
        first_latent = self.first_encoder(first)
        first_critic = self.first_critic(first_latent.detach())

        second_latent = self.second_encoder(second)
        second_critic = self.second_critic(second_latent.detach())
        return first_critic, second_critic

    def training_step(self, batch, batch_n):
        inputs, targets, batch_idx = batch
        first, second = inputs

        optimizers = self.optimizers()

        if self.use_critic:
            first_critic, second_critic = self.critic_forward(first, second)
            critic_optimizer = optimizers[1]
            critic_optimizer.zero_grad()
            critic_loss = self.calculate_critic_loss(
                first_critic, second_critic, batch_idx
            )
            self.manual_backward(critic_loss)
            torch.nn.utils.clip_grad_norm_(
                self.critic_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            critic_optimizer.step()
            main_optimizer = optimizers[0]

            if batch_n % self.critic_iterations != 0:
                return
        else:
            main_optimizer = optimizers

        result = self(first, second)
        main_optimizer.zero_grad()
        loss = self.calculate_loss(result, targets, batch_idx, batch_n)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.main_parameters, self.gradient_clip, error_if_nonfinite=True
        )
        main_optimizer.step()
        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, logger=True, on_epoch=False
        )
        return

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()

    def predict_step(self, batch, _):
        (first, second), _, _ = batch
        first = first.to(self.device)
        second = second.to(self.device)
        result = self(first, second)
        first_to_second = result["first_to_second"]
        second_to_first = result["second_to_first"]
        if self.second_config["loss"] != "mse":
            first_to_second = self.second_decoder.to_prediction(first_to_second)
        if self.first_config["loss"] != "mse":
            second_to_first = self.first_decoder.to_prediction(second_to_first)
        return first_to_second, second_to_first

    def calculate_loss(self, result: dict, targets, batch_idx, batch_n):
        first_target, second_target = targets
        weights = self.calculate_weights(batch_idx)
        loss = self.calculate_standard_loss(
            result, first_target, second_target, weights
        )
        if self.use_mmd:
            loss += self.calculate_mmd_loss(result, batch_idx)
        if self.vae:
            loss += self.calculate_kl_loss()
        if self.use_sim_loss:
            loss += self.calculate_sim_loss(result)
        if self.use_critic:
            critic_loss = self.first_critic.get_loss(
                result["first_critic"], batch_idx, shifted=True
            )
            critic_loss += self.second_critic.get_loss(
                result["second_critic"], batch_idx, shifted=True
            )
            critic_loss *= self.critic_lambda
            self.log(
                "train_critic",
                critic_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
            loss += critic_loss
        return loss

    def calculate_standard_loss(self, result, first_target, second_target, weights):
        first_to_first = self.first_loss(
            result["first_to_first"], first_target, weights
        )
        self.log("11", first_to_first)

        second_to_first = self.first_loss(
            result["second_to_first"], first_target, weights
        )
        self.log("21", second_to_first)

        first_to_second = self.second_loss(
            result["first_to_second"], second_target, weights
        )
        self.log("12", first_to_second)

        second_to_second = self.second_loss(
            result["second_to_second"], second_target, weights
        )
        self.log("22", second_to_second)
        return first_to_first + first_to_second + second_to_first + second_to_second

    def calculate_critic_loss(self, first_critic, second_critic, batch_idx):
        loss = self.first_critic.get_loss(first_critic, batch_idx)
        loss += self.second_critic.get_loss(second_critic, batch_idx)
        self.log(
            "critic",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        main_optimizer = torch.optim.Adam(
            self.main_parameters,
            lr=self.lr,
        )

        def lr_foo(epoch):
            if epoch < self.attack:
                lr_scale = epoch / self.attack
            elif epoch < (self.sustain + self.attack):
                lr_scale = 1.0
            else:
                lr_scale = 1.0 - (epoch - self.sustain - self.attack) / self.release

            return lr_scale

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            main_optimizer, lr_lambda=lr_foo
        )
        optimizers = [{"optimizer": main_optimizer, "lr_scheduler": lr_scheduler}]
        if self.use_critic:
            critic_optimizer = torch.optim.Adam(
                self.critic_parameters,
                lr=self.critic_lr,
            )
            optimizers.append({"optimizer": critic_optimizer})
        return tuple(optimizers)

    def calculate_mmd_loss(self, result, batch_idx):
        mmd_loss = losses.calculate_mmd_loss(result["first_latent"], batch_idx)
        mmd_loss += losses.calculate_mmd_loss(result["second_latent"], batch_idx)
        mmd_loss *= self.mmd_lambda
        self.log(
            "mmd",
            mmd_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return mmd_loss

    def calculate_kl_loss(self):
        kl_loss = self.first_encoder.kl_loss + self.second_encoder.kl_loss
        self.log(
            "kl", kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return kl_loss

    def calculate_sim_loss(self, result):
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
        return sim_loss

    def calculate_weights(self, batch_idx):
        if self.balance_classes:
            return self.batch_weights[batch_idx]
        else:
            return torch.ones_like(batch_idx, device=self.device)

    def get_metrics(self):
        items = super().get_metrics()
        items.pop('v_num', None)
        return items

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
    predictions = torch.cat(raw_predictions, dim=0)
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
        prefix=None,
    ):
        self.test_dataloader = test_dataloader
        self.first_inverse = first_inverse
        self.first_true_target = first_true_target
        self.second_inverse = second_inverse
        self.second_true_target = second_true_target
        if not prefix:
            prefix = "true"
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        first_to_second = []
        second_to_first = []
        for i, batch in enumerate(self.test_dataloader):
            first_to_second_batch, second_to_first_batch = pl_module.predict_step(
                batch, i
            )
            first_to_second.append(first_to_second_batch.cpu().detach())
            second_to_first.append(second_to_first_batch.cpu().detach())

        logger = trainer.logger
        if self.second_inverse is not None:
            predictions = torch.cat(first_to_second, dim=0)
            predictions = self.second_inverse(predictions)
            metric = mp.calculate_target(predictions, self.second_true_target)
            pl_module.log(self.prefix + "_1_to_2", metric, prog_bar=True)

            difference = predictions - self.second_true_target.X.toarray()
            if logger:
                logger.experiment.log(
                    {self.prefix + "_second_difference": wandb.Histogram(difference)}
                )

        if self.first_inverse is not None:
            predictions = torch.cat(second_to_first, dim=0)
            predictions = self.first_inverse(predictions)
            metric = mp.calculate_target(predictions, self.first_true_target)

            pl_module.log(self.prefix + "_2_to_1", metric, prog_bar=True)

            difference = predictions - self.first_true_target.X.toarray()
            if logger:
                logger.experiment.log(
                    {self.prefix + "_first_difference": wandb.Histogram(difference)}
                )
