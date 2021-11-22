from collections import OrderedDict
from itertools import chain

import pandas as pd
from pandas.core.indexing import convert_missing_indexer
import plotly.express as px
import pytorch_lightning as pl
import torch
from lab_scripts.metrics import mp as mp_metrics
from lab_scripts.models.common import losses, plugins
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.distributions.uniform import Uniform
from torch.nn import functional as F


class BioDropout(pl.LightningModule):
    def __init__(self, b, k):
        super().__init__()
        self.b = b
        self.k = k

    def forward(self, x):
        sum_before = x.sum()
        p = self.k * x + self.b
        rand = torch.rand_like(x)
        is_zeroed = rand < p
        x[is_zeroed] = 0.0
        sum_after = x.sum()
        coef = sum_before / sum_after
        return x * coef

class StupidDropout(pl.LightningModule):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        rand = torch.rand_like(x)
        is_zeroed = rand < self.p
        x[is_zeroed] = 0.0
        return x


def get_entry_dropout(config):
    dropout_type = config['type']
    if dropout_type == 'bio':
        return BioDropout(config['b'], config['k'])
    elif dropout_type == 'uniform':
        return nn.Dropout(config['p'])
    elif dropout_type == 'stupid':
        return StupidDropout(config['p'])
    else:
        raise NotImplementedError()



class GaninCritic(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = config["feature_extractor_dims"][-1]
        self.total_batches = config["total_correction_batches"]
        dims = config["critic_dims"]
        dims.insert(0, self.feature_dim)
        self.nets = nn.ModuleList(
            [
                construct_net(dims, config["activation"], [], 0, [])
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


def get_critic(name):
    if name == "ganin":
        return GaninCritic
    else:
        raise NotImplementedError()


class ResidualBlock(pl.LightningModule):
    def __init__(self, input_dim, output_dim, activation, use_batchnorm, use_dropout, dropout, type):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = plugins.get_activation(activation)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(output_dim)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        self.block_type = type

    def forward(self, x):
        y = self.linear(x)
        if self.use_batchnorm:
            y = self.batchnorm(y)
        if self.use_dropout:
            y = self.dropout(y)
        if self.block_type == 'residual':
            return x + y
        else:
            raise NotImplementedError()


def construct_net(dims, activation_name: str, dropout_pos, dropout: float, batchnorm_pos, connections=None, last_activation=True, task_type=None):
    activation = plugins.get_activation(activation_name)

    net = []
    for i in range(len(dims) - 1):
        if connections is None or connections == 'None' or dims[i] != dims[i+1]:
            # Standard block without connections
            net.append(
                (
                    f"{i}_Linear",
                    nn.Linear(dims[i], dims[i + 1]),
                )
            )
            if not last_activation and i == len(dims) - 2:
                break
            net.append((f"{i}_Activation", activation))
            if i in batchnorm_pos:
                net.append((f"{i}_Batchnorm", nn.BatchNorm1d(dims[i+1])))  # type: ignore
            if i in dropout_pos:
                net.append((f"{i}_Dropout", nn.Dropout(dropout)))  # type: ignore
        elif connections == 'residual':
            net.append(
                (f"{i}_ResBlock", ResidualBlock(dims[i], dims[i+1], activation_name, i in batchnorm_pos, i in dropout_pos, dropout, connections))  # type: ignore
            )
    result = nn.Sequential(OrderedDict(net))
    plugins.init(result, activation_name)
    return result


class VariationalDropout(pl.LightningModule):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.zeros((1, n_features)))
        self.last_vi_loss = None

    def forward(self, x):
        eps = 1e-5
        w = torch.sigmoid(self.weight)
        u = Uniform(0.0, 1.0).sample(x.shape).to(self.device)
        z = (
            torch.log(w + eps)
            - torch.log(1 - w + eps)
            + torch.log(u + eps)
            - torch.log(1 - u + eps)
        )
        z = torch.sigmoid(10.0 * z)
        self.last_vi_loss = z.mean()
        return x * z


class Predictor(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_exctractor_dims = config["feature_extractor_dims"]
        self.regression_dims = config["regression_dims"]
        self.activation = config["activation"]
        self.lr = config["lr"]
        self.gradient_clip = config["gradient_clip"]
        self.attack = config["attack"]
        self.sustain = config["sustain"]
        self.release = config["release"]
        self.balance_classes = config["balance_classes"]
        self.total_correct_batches = config["total_correction_batches"]
        if config["batch_weights"] is not None:
            self.register_buffer("batch_weights", torch.tensor(config["batch_weights"]))
        else:
            self.batch_weights = None

        self.dropout = get_entry_dropout(config['entry_dropout'])
        self.random_scale = config['random_scale']
        self.log_transform = config['log_transform']
        self.l2_lambda = config["l2_lambda"]
        self.task_type = config['task_type']

        self.use_mmd_loss = config["use_mmd_loss"]
        self.mmd_lambda = config["mmd_lambda"]

        self.use_l2_loss = config["use_l2_loss"]
        self.l2_loss_lambda = config["l2_loss_lambda"]

        self.use_coral_loss = config["use_coral_loss"]
        self.coral_lambda = config["coral_lambda"]

        self.use_vi_dropout = config["use_vi_dropout"]
        if self.use_vi_dropout:
            self.vi_dropout = VariationalDropout(config["dims"][0])
            self.vi_lambda = config["vi_lambda"]
            self.vi_attack = config["vi_attack"]

        self.feature_extractor = construct_net(
            self.feature_exctractor_dims,
            self.activation,
            config["fe_dropout"],
            config["dropout"],
            config['fe_batchnorm'],
            config['connections'],
        )
        plugins.init(self.feature_extractor, self.activation)

        self.regression = construct_net(
            self.regression_dims,
            self.activation,
            config["regression_dropout"],
            config["dropout"],
            config['regression_batchnorm'],
            config['connections'],
            last_activation=False,
            task_type=self.task_type,
        )
        plugins.init(self.regression, self.activation)
        self.concat_input = config['concat_input']

        self.main_parameters = chain(
            self.feature_extractor.parameters(), self.regression.parameters()
        )

        self.use_critic = config["use_critic"]
        self.critic_type = config["critic_type"]
        if self.use_critic:
            self.automatic_optimization = False
            self.critic = get_critic(self.critic_type)(config)
            self.critic_parameters = self.critic.parameters()
            self.critic_lambda = config["critic_lambda"]
            self.critic_gamma = config["critic_gamma"]
            self.critic_lr = config["critic_lr"]
            self.normal_iterations = config["normal_iterations"]
            self.critic_iterations = config["critic_iterations"]
            self.do_all_steps = (self.normal_iterations == 0) and (
                self.critic_iterations == 0
            )
        self.current_batch = -1

        if 'prediction_weights' in config:
            self.register_buffer("prediction_weights", torch.tensor(config["prediction_weights"]))
            self.use_prediction_weights = True
        else:
            self.use_prediction_weights = False


    def forward(self, x):
        if self.use_vi_dropout:
            x = self.vi_dropout(x)
        features = self.feature_extractor(x)
        if self.concat_input:
            return torch.cat([features, x], dim=1)
        else:
            return features

    def calculate_weights(self, batch_idx):
        if self.balance_classes:
            return self.batch_weights[batch_idx]
        else:
            return torch.ones_like(batch_idx, device=self.device)

    def training_step(self, batch, batch_n):
        self.current_batch += 1
        if not self.use_critic:
            return self.automatic_step(batch, batch_n)
        elif self.critic_type == "ganin":
            self.ganin_training_step(batch, batch_n)
        else:
            raise NotImplementedError()

    def ganin_training_step(self, batch, batch_n):
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
            loss = self.normal_step(main_batch)
            loss += self.correction_step(correction_batches)
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.main_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()

    def critic_step(self, correction_batches, batch_n):
        cor_first = torch.cat(correction_batches, dim=0)
        cor_first = self.augment(cor_first)
        features = self.feature_extractor(cor_first)
        critic_preds = self.critic(features)

        cor_idx = []
        for i, batch in enumerate(correction_batches):
            cor_idx.append(torch.ones((batch.shape[0], 1), device=self.device) * i)
        cor_idx = torch.cat(cor_idx, dim=0)
        critic_loss = self.critic.calculate_loss(critic_preds, cor_idx)
        self.log("critic", critic_loss, logger=True, prog_bar=True)
        return critic_loss

    def automatic_step(self, batch, batch_n):
        loss = 0.0
        if self.total_correct_batches > 0:
            main_batch = batch[0]
            correction_batches = batch[1:]
            loss += self.correction_step(correction_batches)
        else:
            main_batch = batch

        loss += self.normal_step(main_batch)
        self.log("loss", loss, logger=True)
        return loss

    def normal_step(self, main_batch):
        first, second, batch_idx = main_batch
        first = self.augment(first)
        features = self.forward(first)
        predictions = self.regression(features)
        return self.calculate_standard_loss(predictions, second, batch_idx)
    
    def augment(self, first):
        if self.log_transform:
            first = torch.log(first + 1.0)
        first = self.dropout(first)
        if self.random_scale > 0.0:
            scale = torch.rand(first.shape[0], device=self.device) * 2 * self.random_scale
            scale = scale + (1 - self.random_scale)
            first = first * scale[:, None]

        return first

    def correction_step(self, correction_batches):
        cor_first = torch.cat(correction_batches, dim=0)
        cor_first = self.augment(cor_first)
        cor_features = self.forward(cor_first)
        cor_idx = []
        for i, batch in enumerate(correction_batches):
            cor_idx.append(torch.ones((batch.shape[0], 1), device=self.device) * i)
        cor_idx = torch.cat(cor_idx, dim=0).flatten()
        return self.calculate_div_loss(cor_features, cor_idx)

    def calculate_standard_loss(self, source_predictions, source_second, source_idx):
        weights = self.calculate_weights(source_idx)
        mse_loss = F.mse_loss(source_predictions, source_second, reduction="none")
        loss = torch.unsqueeze(weights, dim=-1) * mse_loss
        if self.use_prediction_weights:
            loss =  loss * torch.unsqueeze(self.prediction_weights, dim=0)
        loss = loss.mean()

        if self.use_vi_dropout:
            loss += self.calculate_vi_loss()

        self.log("reg", loss, logger=True, prog_bar=True)
        return loss

    def calculate_vi_loss(self):
        vi_loss = self.vi_dropout.last_vi_loss
        vi_loss *= self.vi_lambda
        if self.current_epoch < self.vi_attack:
            vi_loss *= self.current_epoch / self.vi_attack
        self.log("vi", vi_loss, logger=True)
        return vi_loss

    def calculate_div_loss(self, features, idx):
        loss = 0.0
        if self.use_mmd_loss:
            loss += self.calculate_mmd_loss(features, idx)

        if self.use_l2_loss:
            loss += self.calulate_l2_loss(features, idx)

        if self.use_coral_loss:
            loss += self.calculate_coral_loss(features, idx)

        if self.use_critic:
            loss += self.calculate_critic_loss(features, idx)

        self.log("div", loss, logger=True, prog_bar=True)
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
        if self.critic_type == "ganin":
            critic_loss = (
                self.critic.calculate_loss(critic_preds, batch_idx, inverse=True)
                * self.critic_lambda
            )
        else:
            critic_loss = (
                self.critic.calculate_loss(critic_preds, batch_idx) * self.critic_lambda
            )
        self.log("critic_adv", critic_loss, logger=True, prog_bar=True)
        return critic_loss

    def predict_step(self, batch, batch_n):
        if self.log_transform:
            batch = torch.log(batch + 1.0)
        features = self.forward(batch.to(self.device))
        prediction = self.regression(features)
        return prediction

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()

    def configure_optimizers(self):
        main_optimizer = torch.optim.Adam(
            self.main_parameters, lr=self.lr, weight_decay=self.l2_lambda
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


class TargetCallback(pl.Callback):
    def __init__(self, dataset, inverse_transform, target, prefix: str):
        self.dataset = dataset
        self.inverse_transform = inverse_transform
        self.target = target
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        second_pred = []
        with torch.no_grad():
            for i, batch in enumerate(self.dataset):
                prediction = pl_module.predict_step(batch, i)
                second_pred.append(prediction.cpu())
        second_pred = torch.cat(second_pred, dim=0)
        second_pred = self.inverse_transform(second_pred)
        metric = mp_metrics.calculate_target(second_pred, self.target)
        pl_module.log(self.prefix + "_m", metric, logger=True, prog_bar=True)


class BatchEffectCallback(pl.Callback):
    def __init__(self, train_dataset, test_dataset, frequency):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.current_epoch = -1
        self.frequency = frequency

    def on_train_epoch_end(self, trainer, pl_module):
        self.current_epoch += 1
        if self.current_epoch % self.frequency != 0:
            return
        pl_module.eval()
        all_features = []
        all_mse = []
        all_batch_idx = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_dataset):
                source_first, source_second, source_idx = batch
                source_features = pl_module.forward(source_first.to(pl_module.device))
                predictions = pl_module.regression(source_features)
                mse = F.mse_loss(
                    predictions.cpu(), source_second.cpu(), reduction="none"
                ).mean(dim=1)
                all_mse.append(mse)
                all_features.append(source_features.cpu())
                all_batch_idx.append(source_idx.cpu())
            for i, batch in enumerate(self.test_dataset):
                target_first, target_second, target_idx = batch
                target_features = pl_module.forward(target_first.to(pl_module.device))
                predictions = pl_module.regression(target_features)
                mse = F.mse_loss(
                    predictions.cpu(), target_second.cpu(), reduction="none"
                ).mean(dim=1)
                all_mse.append(mse)
                all_features.append(target_features.cpu())
                target_idx += 100
                all_batch_idx.append(target_idx.cpu())

        features = torch.cat(all_features, dim=0).numpy()
        batch_idx = torch.cat(all_batch_idx, dim=0).numpy()
        mse = torch.cat(all_mse, dim=0).numpy()

        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
        tsne = TSNE(n_jobs=-1)
        embed = tsne.fit_transform(features)

        test_idx = batch_idx > 90

        df = pd.DataFrame({"mse": mse, "batch": batch_idx, "is_test": batch_idx > 90})
        df["batch"] = df["batch"].astype("category")
        fig_1 = px.scatter(embed, x=0, y=1, color=df["batch"])
        fig_2 = px.scatter(embed, x=0, y=1, color=df["mse"], text=df["is_test"])
        fig_3 = px.scatter(
            embed[test_idx], x=0, y=1, color=df["mse"][test_idx], range_color=[0.0, 2.0]
        )
        fig_4 = px.scatter(
            embed[~test_idx],
            x=0,
            y=1,
            color=df["mse"][~test_idx],
            range_color=[0.0, 2.0],
        )
        trainer.logger.experiment.log(
            {
                "batch effect": fig_1,
                "mse": fig_2,
                "mse_test": fig_3,
                "mse_train": fig_4,
            }
        )
