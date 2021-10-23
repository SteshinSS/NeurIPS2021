import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lab_scripts.models.common import plugins, losses
from lab_scripts.metrics.mp import mp_metrics
from collections import OrderedDict
from torch.distributions.uniform import Uniform


def construct_net(dims, activation_name: str, dropout_pos, dropout: float):
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
    return nn.Sequential(OrderedDict(net))


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
        self.attack = config["attack"]
        self.sustain = config["sustain"]
        self.release = config["release"]
        self.balance_classes = config["balance_classes"]
        self.register_buffer("batch_weights", torch.tensor(config["batch_weights"]))

        self.l2_lambda = config["l2_lambda"]

        self.use_mmd_loss = config["use_mmd_loss"]
        self.mmd_lambda = config["mmd_lambda"]

        self.use_l2_loss = config['use_l2_loss']
        self.l2_loss_lambda = config['l2_loss_lambda']

        self.use_coral_loss = config['use_coral_loss']
        self.coral_lambda = config['coral_lambda']

        self.use_vi_dropout = config["use_vi_dropout"]
        if self.use_vi_dropout:
            self.vi_dropout = VariationalDropout(config["dims"][0])
            self.vi_lambda = config["vi_lambda"]
            self.vi_attack = config["vi_attack"]

        self.feature_extractor = construct_net(
            self.feature_exctractor_dims, self.activation, config['fe_dropout'], config['dropout']
        )
        plugins.init(self.feature_extractor, self.activation)

        self.regression = construct_net(self.regression_dims, self.activation, config['regression_dropout'], config['dropout'])
        plugins.init(self.regression, self.activation)

    def forward(self, x):
        if self.use_vi_dropout:
            x = self.vi_dropout(x)
        features = self.feature_extractor(x)
        return self.regression(features), features

    def calculate_weights(self, batch_idx):
        if self.balance_classes:
            return self.batch_weights[batch_idx]
        else:
            return torch.ones_like(batch_idx, device=self.device)

    def training_step(self, batch, batch_n):
        first, second, batch_idx = batch
        predictions, features = self.forward(first)
        loss = self.calculate_loss(predictions, features, second, batch_idx)
        self.log("loss", loss, logger=True)
        return loss

    def calculate_loss(self, predictions, features, target, batch_idx):
        weights = self.calculate_weights(batch_idx)
        loss = self.calculate_standard_loss(predictions, target, weights)

        if self.use_vi_dropout:
            loss += self.calculate_vi_loss()

        if self.use_mmd_loss:
            loss += self.calculate_mmd_loss(features, batch_idx)
        
        if self.use_l2_loss:
            loss += self.calulate_l2_loss(features, batch_idx)
        
        if self.use_coral_loss:
            loss += self.calculate_coral_loss(features, batch_idx)

        return loss

    def calculate_standard_loss(self, predictions, target, weights):
        mse_loss = F.mse_loss(predictions, target, reduction="none")
        mse_loss = torch.unsqueeze(weights, dim=-1) * mse_loss
        return mse_loss.mean()
    
    def calculate_coral_loss(self, features, batch_idx):
        coral_loss = losses.calculate_coral_loss(features, batch_idx) * self.coral_lambda
        self.log('coral', coral_loss, logger=True)
        return coral_loss
    
    def calulate_l2_loss(self, features, batch_idx):
        l2_loss = losses.calculate_l2_loss(features, batch_idx) * self.l2_loss_lambda
        self.log('l2', l2_loss, logger=True)
        return l2_loss

    def calculate_vi_loss(self):
        vi_loss = self.vi_dropout.last_vi_loss
        vi_loss *= self.vi_lambda
        if self.current_epoch < self.vi_attack:
            vi_loss *= self.current_epoch / self.vi_attack
        self.log("vi", vi_loss.item(), logger=True)
        return vi_loss

    def calculate_mmd_loss(self, features, batch_idx):
        mmd_loss = (
            losses.calculate_mmd_loss(features, batch_idx) * self.mmd_lambda
        )
        self.log("mmd", mmd_loss, logger=True)
        return mmd_loss

    def predict_step(self, batch, batch_n):
        first, _, _ = batch
        result, _ = self.forward(first.to(self.device))
        return result

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()

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
