import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lab_scripts.models.common import plugins
from lab_scripts.metrics.mp import mp_metrics
from collections import OrderedDict
from torch.distributions.uniform import Uniform

def construct_net(dims, bn, activation_name: str):
    activation = plugins.get_activation(activation_name)

    net = []
    for i in range(len(dims) - 1):
        net.append((
            f'{i}_Linear',
            nn.Linear(dims[i], dims[i+1]),
        ))
        net.append((
            f'{i}_Actiavtion',
            activation
        ))
        if i in bn:
            net.append((
                f'{i}_BatchNorm',
                nn.BatchNorm1d(dims[i+1])  # type: ignore
            ))
    return nn.Sequential(OrderedDict(net))


class VariationalDropout(pl.LightningModule):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.weight = nn.Parameter(torch.zeros((1, n_features)))
    
    def forward(self, x):
        eps = 1e-5
        w = torch.sigmoid(self.weight)
        u = Uniform(0., 1.).sample(x.shape).to(self.device)
        z = torch.log(w + eps) - torch.log(1 - w + eps) + torch.log(u + eps) - torch.log(1 - u + eps)
        z = torch.sigmoid(0.1 * z)
        vi_loss = z.mean()
        return x * z, vi_loss



class Predictor(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.dims = config['dims']
        self.bn = config['bn']
        self.lr = config['lr']
        self.l1_lambda = config['l1_lambda']
        self.l2_lambda = config['l2_lambda']
        self.attack = config['attack']
        self.sustain = config['sustain']
        self.release = config['release']
        self.balance_classes = config['balance_classes']
        self.register_buffer("batch_weights", torch.tensor(config["batch_weights"]))

        activation = config['activation']
        self.use_vi_dropout = config['use_vi_dropout']
        if self.use_vi_dropout:
            self.vi_dropout = VariationalDropout(config['dims'][0])
            self.vi_lambda = config['vi_lambda']
            self.vi_attack = config['vi_attack']
        self.net = construct_net(self.dims, self.bn, activation)
        plugins.init(self.net, activation)

    def forward(self, x):
        if self.use_vi_dropout:
            x_dropped, vi_loss = self.vi_dropout(x)
            return self.net(x_dropped), vi_loss
        else:
            return self.net(x)

    def calculate_weights(self, batch_idx):
        if self.balance_classes:
            return self.batch_weights[batch_idx]
        else:
            return torch.ones_like(batch_idx, device=self.device)


    def training_step(self, batch, batch_n):
        first, second, batch_idx = batch
        second_pred = self.forward(first)
        loss = self.calculate_loss(second_pred, second, batch_idx)
        self.log('loss', loss, logger=True)
        return loss
    
    def calculate_loss(self, second_pred, second, batch_idx):
        loss = 0.0
        if self.use_vi_dropout:
            second_pred, vi_loss = second_pred
            self.log('vi', vi_loss.item(), logger=True, prog_bar=True)
            vi_loss *= self.vi_lambda
            if self.current_epoch < self.vi_attack:
                vi_loss *= (self.current_epoch / self.vi_attack)
            loss += vi_loss
        weights = self.calculate_weights(batch_idx)
        mse_loss = F.mse_loss(second_pred, second, reduction='none')
        mse_loss = torch.unsqueeze(weights, dim=-1) * mse_loss
        loss += mse_loss.mean()

        l1_loss = torch.abs(self.net[0].weight).sum() * self.l1_lambda
        loss += l1_loss
        return loss
    
    def predict_step(self, batch, batch_n):
        first, _, _ = batch
        result = self.forward(first.to(self.device))
        if self.use_vi_dropout:
            result = result[0]
        return result

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_lambda)

        def lr_foo(epoch):
            epoch = epoch + 1
            if epoch < self.attack:
                lr_scale = epoch / self.attack
            elif epoch < (self.sustain + self.attack):
                lr_scale = 1.0
            else:
                lr_scale = 1.0 - (epoch - self.sustain - self.attack) / self.release

            return lr_scale

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_foo
        )
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
        pl_module.log(self.prefix + '_m', metric, logger=True, prog_bar=True)
