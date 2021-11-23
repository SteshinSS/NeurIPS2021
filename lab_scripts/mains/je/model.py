from collections import OrderedDict

import pytorch_lightning as pl
import torch
from lab_scripts.models.common import losses, plugins
from torch import nn
from torch.nn import functional as F


class GaninCritic(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = config["common_dim"][-1]
        self.total_batches = config["total_correction_batches"]
        dims = config["critic_dims"]
        dims.insert(0, self.feature_dim)
        self.nets = nn.ModuleList(
            [
                construct_net(dims, config["activation"])
                for _ in range(self.total_batches)
            ]
        )
        self.to_prediction = nn.ModuleList(
            [nn.Linear(dims[-1], 1) for _ in range(self.total_batches)]
        )

    def forward(self, x):
        predictions = []
        for net, to_pred in zip(self.nets, self.to_prediction):
            prediction = net(x)
            predictions.append(torch.squeeze(to_pred(prediction)))
        return predictions

    def calculate_loss(self, predictions, batch_idx, inverse=False):
        loss = 0.0
        for i, prediction in enumerate(predictions):
            true = batch_idx == i
            if inverse:
                true = ~true
            weights = (~true).sum() / (true).sum()
            loss += F.binary_cross_entropy_with_logits(
                prediction, true.to(torch.float32).flatten(), pos_weight=weights
            )
        return loss


class Encoder(pl.LightningModule):
    def __init__(self, config, activation_name: str):
        super().__init__()
        self.dropout = nn.Dropout(config['dropout'])
        self.net = construct_net(config['dim'], activation_name)

    def forward(self, x):
        y = self.dropout(x)
        return self.net(y)


class Decoder(pl.LightningModule):
    def __init__(self, config, activation_name: str, first_dim):
        super().__init__()
        dims = config['dim']
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
        self.gradient_clip = config['gradient_clip']
        self.total_correction_batches = config["total_correction_batches"]

        self.first_encoder = Encoder(config["first"], config["activation"])
        self.second_encoder = Encoder(config["second"], config["activation"])
        self.net = construct_net(config["common_dim"], config["activation"])
        self.first_decoder = Decoder(
            config["first"], config["activation"], config["common_dim"][-1]
        )
        self.second_decoder = Decoder(
            config["second"], config["activation"], config["common_dim"][-1]
        )

        self.main_parameters = self.parameters()

        self.use_mmd_loss = config["use_mmd_loss"]
        self.mmd_lambda = config["mmd_lambda"]

        self.use_l2_loss = config["use_l2_loss"]
        self.l2_loss_lambda = config["l2_loss_lambda"]

        self.use_coral_loss = config["use_coral_loss"]
        self.coral_lambda = config["coral_lambda"]

        self.use_critic = config["use_critic"]
        if self.use_critic:
            self.critic = GaninCritic(config)
            self.critic_lambda = config["critic_lambda"]
            self.normal_iterations = config["normal_iterations"]
            self.critic_iterations = config["critic_iterations"]
            self.critic_lr = config["critic_lr"]
            self.critic_parameters = self.critic.parameters()
            self.do_all_steps = (
                self.normal_iterations == 0 and self.critic_iterations == 0
            )
            self.automatic_optimization = False
        self.current_batch = -1

    def forward(self, first, second):
        first_embed = self.first_encoder(first)
        second_embed = self.second_encoder(second)
        features = torch.cat([first_embed, second_embed], dim=1)
        embeddings = self.net(features)
        return embeddings

    def training_step(self, batch, batch_n):
        self.current_batch += 1
        if not self.use_critic:
            return self.automatic_step(batch, batch_n)
        else:
            self.manual_step(batch, batch_n)

    def automatic_step(self, batch, batch_n):
        loss = 0.0
        if self.total_correction_batches > 0:
            train_batch = batch[0]
            correct_batch = batch[1:]
        else:
            train_batch = batch
            correct_batch = None
        loss += self.target_step(train_batch)
        loss += self.div_step(correct_batch)
        return loss

    def manual_step(self, batch, batch_n):
        main_batch = batch[0]
        correction_batches = batch[1:]
        if self.do_all_steps:
            is_normal = True
            is_critic = True
        else:
            is_normal = self.current_batch % (
                self.normal_iterations + self.critic_iterations
            ) > (self.critic_iterations - 1)
            is_critic = ~is_normal

        optimizers = self.optimizers()
        if is_critic:
            optimizer = optimizers[1]
            loss = self.critic_step(correction_batches, batch_n)
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.critic_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()

        if is_normal:
            optimizer = optimizers[0]
            loss = self.target_step(main_batch)
            loss += self.div_step(correction_batches)
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.main_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()

    def critic_step(self, correction_batches, batch_n):
        first = []
        second = []
        cor_idx = []
        for i, cor_batch in enumerate(correction_batches):
            first.append(cor_batch[0])
            second.append(cor_batch[1])
            cor_idx.append(torch.ones((cor_batch[0].shape[0], 1), device=self.device) * i)
        first = torch.cat(first, dim=0)
        second = torch.cat(second, dim=0)
        features = self(first, second)
        critic_preds = self.critic(features)

        cor_idx = torch.cat(cor_idx, dim=0)
        critic_loss = self.critic.calculate_loss(critic_preds, cor_idx)
        self.log("critic", critic_loss, logger=True, prog_bar=True)
        return critic_loss

    def target_step(self, batch):
        first, second = batch
        embeddings = self(first, second)
        first_reconstruction = self.first_decoder(embeddings)
        second_reconstruction = self.second_decoder(embeddings)
        rec_loss = self.get_reconstruction_loss(
            first, first_reconstruction, second, second_reconstruction
        )
        self.log("rec", rec_loss, logger=True, prog_bar=False)
        return rec_loss

    def get_reconstruction_loss(
        self, first, first_reconstruction, second, second_reconstruction
    ):
        loss = F.mse_loss(first, first_reconstruction)
        loss += F.mse_loss(second, second_reconstruction)
        return loss

    def div_step(self, batch):
        if batch is None:
            return 0.0
        first_all = []
        second_all = []
        idx = []
        for i, cor_batch in enumerate(batch):
            first, second = cor_batch
            first_all.append(first)
            second_all.append(second)
            idx.append(torch.ones((first.shape[0], 1), device=self.device) * i)
        first_all = torch.cat(first_all, dim=0)
        second_all = torch.cat(second_all, dim=0)
        embeddings = self(first_all, second_all)
        idx = torch.cat(idx, dim=0).flatten()
        div_loss = self.get_div_loss(embeddings, idx)
        self.log('div', div_loss, logger=True, prog_bar=False)
        return div_loss

    def get_div_loss(self, features, idx):
        loss = 0.0
        if self.use_mmd_loss:
            loss += self.calculate_mmd_loss(features, idx)

        if self.use_l2_loss:
            loss += self.calulate_l2_loss(features, idx)

        if self.use_coral_loss:
            loss += self.calculate_coral_loss(features, idx)

        if self.use_critic:
            loss += self.calculate_critic_loss(features, idx)

        return loss

    def calculate_mmd_loss(self, features, batch_idx):
        mmd_loss = losses.calculate_mmd_loss(features, batch_idx) * self.mmd_lambda
        self.log("mmd", mmd_loss, logger=True)
        return mmd_loss

    def calulate_l2_loss(self, features, batch_idx):
        l2_loss = losses.calculate_l2_loss(features, batch_idx) * self.l2_loss_lambda
        self.log("l2", l2_loss, logger=True)
        return l2_loss

    def calculate_coral_loss(self, features, batch_idx):
        coral_loss = (
            losses.calculate_coral_loss(features, batch_idx) * self.coral_lambda
        )
        self.log("coral", coral_loss, logger=True)
        return coral_loss

    def calculate_critic_loss(self, features, batch_idx):
        critic_preds = self.critic(features)
        critic_loss = (
            self.critic.calculate_loss(critic_preds, batch_idx, inverse=True)
            * self.critic_lambda
        )

        self.log("critic_adv", critic_loss, logger=True, prog_bar=True)
        return critic_loss

    def configure_optimizers(self):
        main_optimizer = torch.optim.Adam(self.main_parameters, lr=self.lr)

        def lr_foo(epoch):
            epoch = epoch + 1
            if epoch < self.attack:
                lr_scale = epoch / self.attack
            elif epoch < (self.sustain + self.attack):
                lr_scale = 1.0
            else:
                lr_scale = 1.0 - (epoch - self.sustain - self.attack) / self.release

            return lr_scale

        main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            main_optimizer, lr_lambda=lr_foo
        )
        main_optimizer_dict = {
            "optimizer": main_optimizer,
            "lr_scheduler": main_lr_scheduler,
        }
        if not self.use_critic:
            return main_optimizer_dict

        optimizers = [main_optimizer_dict]
        critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=self.critic_lr)
        optimizers.append({"optimizer": critic_optimizer})
        return optimizers
