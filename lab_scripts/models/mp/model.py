import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from lab_scripts.models.common import plugins, losses
from lab_scripts.metrics.mp import mp_metrics
from collections import OrderedDict
from torch.distributions.uniform import Uniform
from itertools import chain
from torch.autograd import grad


class WDGRLCritic(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = config['feature_extractor_dims'][-1]
        self.total_batches = config['total_batches']
        dims = config['critic_dims']
        dims.insert(0, self.feature_dim)
        self.nets = nn.ModuleList([construct_net(dims, config['activation'], [], 0) for _ in range(self.total_batches)])       
        self.to_prediction = nn.ModuleList([
            nn.Linear(dims[-1], 1) for _ in range(self.total_batches)
        ])
    
    def forward(self, x):
        predictions = []
        for net, to_pred in zip(self.nets, self.to_prediction):
            prediction = net(x)
            predictions.append(torch.squeeze(to_pred(prediction)))
        return predictions

    def gradient_penalty(self, features, batch_idx):
        batch_idx = torch.squeeze(batch_idx)
        unique_batches = torch.unique(batch_idx)
        gradient_penalty = 0.0
        for batch in unique_batches:
            reference_batch = features[batch_idx == batch]
            other_batch = features[batch_idx != batch]
            ref_idx = torch.randint(low=0, high=reference_batch.shape[0], size=[other_batch.shape[0]])
            difference = reference_batch[ref_idx] - other_batch
            alpha = torch.rand(difference.shape[0], 1).to(self.device)
            interpolates = other_batch + alpha * difference
            interpolates = torch.cat([interpolates, reference_batch, other_batch], dim=0).requires_grad_()

            preds = self.nets[batch](interpolates)
            preds = self.to_prediction[batch](preds)
            gradients = grad(preds, interpolates,
                            grad_outputs=torch.ones_like(preds),
                            retain_graph=True, create_graph=True)[0]
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty += ((gradient_norm - 1)**2).mean()
        return gradient_penalty / unique_batches.shape[0]
    
    def calculate_loss(self, predictions, batch_idx):
        batch_idx = torch.squeeze(batch_idx)
        unique_batches = torch.unique(batch_idx)
        wasserstain_distance = 0.0
        for batch in unique_batches:
            preds = predictions[batch]
            reference_batch = preds[batch_idx == batch]
            other_batch = preds[batch_idx != batch]
            wasserstain_distance += reference_batch.mean() - other_batch.mean()
        return wasserstain_distance / unique_batches.shape[0]
    

class GaninCritic(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = config['feature_extractor_dims'][-1]
        self.total_batches = config['total_batches']
        dims = config['critic_dims']
        dims.insert(0, self.feature_dim)
        self.nets = nn.ModuleList([construct_net(dims, config['activation'], [], 0) for _ in range(self.total_batches)])
        self.to_prediction = nn.ModuleList([
            nn.Linear(dims[-1], 1) for _ in range(self.total_batches)
        ])
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
            loss += F.binary_cross_entropy_with_logits(prediction, true.to(torch.float32), pos_weight=weights)
        return loss


def get_critic(name):
    if name == 'ganin':
        return GaninCritic
    elif name == 'wdgrl':
        return WDGRLCritic
    else:
        raise NotImplementedError()


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
        self.gradient_clip = config['gradient_clip']
        self.attack = config["attack"]
        self.sustain = config["sustain"]
        self.release = config["release"]
        self.balance_classes = config["balance_classes"]
        self.register_buffer("batch_weights", torch.tensor(config["batch_weights"]))
        self.inject_test = config['inject_test']
        self.black_and_white = config['black_and_white']
        self.total_batches = config['total_batches']

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

        self.main_parameters = chain(self.feature_extractor.parameters(), self.regression.parameters())

        self.use_critic = config['use_critic']
        self.critic_type = config['critic_type']
        if self.use_critic:
            self.automatic_optimization = False
            self.critic = get_critic(self.critic_type)(config)
            self.critic_parameters = self.critic.parameters()
            self.critic_lambda = config['critic_lambda']
            self.critic_gamma = config['critic_gamma']
            self.normal_iterations = config['normal_iterations']
            self.critic_iterations = config['critic_iterations']
            self.critic_lr = config['critic_lr']
        
        self.current_batch = -1

    def forward(self, x):
        if self.use_vi_dropout:
            x = self.vi_dropout(x)
        features = self.feature_extractor(x)
        return features

    def calculate_weights(self, batch_idx):
        if self.balance_classes:
            return self.batch_weights[batch_idx]
        else:
            return torch.ones_like(batch_idx, device=self.device)
    
    def training_step(self, batch, batch_n):
        self.current_batch += 1
        if not self.use_critic:
            return self.normal_step(batch, batch_n)
        elif self.critic_type == 'ganin':
            self.ganin_training_step(batch, batch_n)
        elif self.critic_type == 'wdgrl':
            self.wdgrl_training_step(batch, batch_n)

    def wdgrl_training_step(self, batch, batch_n):
        optimizers = self.optimizers()
        first, second, batch_idx = batch
        for _ in range(self.critic_iterations):
            optimizer = optimizers[1]
            with torch.no_grad():
                features = self.feature_extractor(first)

            gradient_penalty = self.critic.gradient_penalty(features, batch_idx) * self.critic_gamma
            critic_preds = self.critic(features)
            critic_loss = gradient_penalty - self.critic.calculate_loss(critic_preds, batch_idx)
            self.log('gp', gradient_penalty, logger=True)
            optimizer.zero_grad()
            self.manual_backward(critic_loss)
            torch.nn.utils.clip_grad_norm_(
                self.critic_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            self.log('critic', critic_loss, logger=True, prog_bar=True)

        for _ in range(self.normal_iterations):
            optimizer = optimizers[0]
            loss = self.normal_step(batch, batch_n)
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.main_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()

    
    def ganin_training_step(self, batch, batch_n):
        critic_step = (self.current_batch % (self.normal_iterations + self.critic_iterations)) > (self.normal_iterations - 1)
        optimizers = self.optimizers()
        if critic_step:
            optimizer = optimizers[1]
            loss = self.critic_step(batch, batch_n)
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.critic_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
        else:
            optimizer = optimizers[0]
            loss = self.normal_step(batch, batch_n)
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.main_parameters, self.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
    
    def get_features(self, source_features, source_idx, target_first, target_idx):
        if self.inject_test:
            target_features = self.feature_extractor(target_first)
            if self.black_and_white:
                zero_source_idx = torch.ones_like(source_idx)
                one_target_idx = torch.zeros_like(target_idx)
                idx = torch.cat([zero_source_idx, one_target_idx], dim=0)
                features = torch.cat([source_features, target_features], dim=0)
            else:
                # Shift source batch idx by number of target indices
                source_idx += self.total_batches - len(self.batch_weights)
                idx = torch.cat([source_idx, target_idx], dim=0)
                features = torch.cat([source_features, target_features], dim=0)
        else:
            features = source_features
            if self.black_and_white:
                idx = torch.ones_like(source_idx)
                idx[source_idx == 0] = 0
            else:
                idx = source_idx
        return features, idx
    
    def critic_step(self, batch, batch_n):
        source_first, _, source_idx = batch[0]
        source_features = self.feature_extractor(source_first)
        target_first, _, target_idx = batch[1]
        features, idx = self.get_features(source_features, source_idx, target_first, target_idx)
        critic_preds = self.critic(features)
        critic_loss = self.critic.calculate_loss(critic_preds, idx)
        self.log('critic', critic_loss, logger=True, prog_bar=True)
        return critic_loss

    def normal_step(self, batch, batch_n):
        source_first, source_second, source_idx = batch[0]
        source_features = self.forward(source_first)
        source_predictions = self.regression(source_features)
        loss = self.calculate_standard_loss(source_predictions, source_second, source_idx)

        target_first, _ , target_idx = batch[1]
        features, idx = self.get_features(source_features, source_idx, target_first, target_idx)
                
        loss += self.calculate_div_loss(features, idx)
        self.log("loss", loss, logger=True)
        return loss

    def calculate_standard_loss(self, source_predictions, source_second, source_idx):
        weights = self.calculate_weights(source_idx)
        mse_loss = F.mse_loss(source_predictions, source_second, reduction="none")
        loss = (torch.unsqueeze(weights, dim=-1) * mse_loss).mean()

        if self.use_vi_dropout:
            loss += self.calculate_vi_loss()
        
        self.log('reg', loss, logger=True, prog_bar=True)
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
        
        self.log('div', loss, logger=True, prog_bar=True)
        return loss

    def calculate_mmd_loss(self, features, batch_idx):
        mmd_loss = (
            losses.calculate_mmd_loss(features, batch_idx) * self.mmd_lambda
        )
        self.log("mmd", mmd_loss, logger=True)
        return mmd_loss

    def calulate_l2_loss(self, features, batch_idx):
        l2_loss = losses.calculate_l2_loss(features, batch_idx) * self.l2_loss_lambda
        self.log('l2', l2_loss, logger=True)
        return l2_loss

    def calculate_coral_loss(self, features, batch_idx):
        coral_loss = losses.calculate_coral_loss(features, batch_idx) * self.coral_lambda
        self.log('coral', coral_loss, logger=True)
        return coral_loss
    
    def calculate_critic_loss(self, features, batch_idx):
        critic_preds = self.critic(features)
        if self.critic_type == 'ganin':
            critic_loss = self.critic.calculate_loss(critic_preds, batch_idx, inverse=True) * self.critic_lambda
        else:
            critic_loss = self.critic.calculate_loss(critic_preds, batch_idx) * self.critic_lambda
        self.log('critic_adv', critic_loss, logger=True, prog_bar=True)
        return critic_loss

    def predict_step(self, batch, batch_n):
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

        main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(main_optimizer, lr_lambda=lr_foo)
        main_optimizer_dict = {"optimizer": main_optimizer, "lr_scheduler": main_lr_scheduler}
        if not self.use_critic:
            return main_optimizer_dict
        
        optimizers = [main_optimizer_dict]
        critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=self.critic_lr
        )
        optimizers.append({'optimizer': critic_optimizer})
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
                first = batch[0]
                prediction = pl_module.predict_step(first, i)
                second_pred.append(prediction.cpu())
        second_pred = torch.cat(second_pred, dim=0)
        second_pred = self.inverse_transform(second_pred)
        metric = mp_metrics.calculate_target(second_pred, self.target)
        pl_module.log(self.prefix + "_m", metric, logger=True, prog_bar=True)
