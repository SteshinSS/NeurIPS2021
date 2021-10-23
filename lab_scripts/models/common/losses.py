from pyro.distributions import zero_inflated
import torch
import torch.nn.functional as F
from lab_scripts.utils import utils
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial, ZeroInflatedDistribution
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
from einops import repeat


def lognorm_loss(predictied_parameters, targets, weights):
    eps = 1e-6
    loc, scale = predictied_parameters
    exp_distribution = LogNormal(loc, scale)
    exp_loss = exp_distribution.log_prob(targets + eps).mean()
    return -exp_loss


def zinb_loss(predicted_parameters, targets, weights):
    zinb_r, zinb_p, dropout = predicted_parameters
    zinb_distribution = ZeroInflatedNegativeBinomial(zinb_r, probs=zinb_p, gate=dropout)
    log_loss = torch.unsqueeze(weights, dim=-1) * zinb_distribution.log_prob(targets)
    log_loss = log_loss.mean()
    return -log_loss


def nb_loss(predicted_parameters, targets, weights):
    nb_r, nb_p = predicted_parameters
    nb_distribution = NegativeBinomial(nb_r, nb_p)
    log_loss = torch.unsqueeze(weights, dim=-1) * nb_distribution.log_prob(targets)
    log_loss = log_loss.mean()
    return -log_loss


def weighted_mse(predicted, true, weights, is_ce_phase=None):
    difference = (predicted - true) ** 2
    difference = torch.unsqueeze(weights, dim=-1) * difference
    return difference.mean()


def zero_mse(predicted, true, weights, is_ce_phase):
    y, is_zero = predicted
    #if is_ce_phase:
    #    true_zero = true == 0.0
    #    weights_ce = torch.ones_like(true_zero).to(torch.float32)
    #    weights_ce[~true_zero] *= 20
    #    loss = F.binary_cross_entropy_with_logits(is_zero, true_zero.to(torch.float32), weight=weights_ce)
    #else:
    #    predicted_not_zero = torch.sigmoid(is_zero) < 0.5
    #    loss = F.mse_loss(y[predicted_not_zero], true[predicted_not_zero])
    dist = ZeroInflatedDistribution(Normal(y, 1), gate_logits=is_zero)
    loss = -dist.log_prob(true)
    loss = torch.unsqueeze(weights, dim=-1) * loss
    return loss.mean()


def get_loss(loss_name: str):
    if loss_name == "mse":
        return weighted_mse
    elif loss_name == "nb":
        return nb_loss
    elif loss_name == "zinb":
        return zinb_loss
    elif loss_name == "lognorm":
        return lognorm_loss
    elif loss_name == "zero_mse":
        return zero_mse


def calculate_coral_loss(features, batch_idx):
    coral_loss = 0.0
    reference_batch = features[batch_idx == 0]
    unique_batches = torch.unique(batch_idx)
    for batch in unique_batches:
        if batch == 0:
            continue
        other_batch = features[batch_idx == batch]
        coral_loss += compute_one_coral_loss(reference_batch, other_batch)
    return coral_loss    


def compute_one_coral_loss(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / d
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    device = input_data.device

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
    return c


def calculate_l2_loss(X, batch_idx):
    l2_loss = 0.0
    reference_batch = X[batch_idx == 0].mean(dim=0)
    unique_batches = torch.unique(batch_idx)
    for batch in unique_batches:
        if batch == 0:
            continue
        other_batch = X[batch_idx == batch].mean(dim=0)
        l2_loss += torch.norm(reference_batch - other_batch)
    return l2_loss


def calculate_mmd_loss(X, batch_idx):
    mmd_loss = 0.0
    reference_batch = X[batch_idx == 0]
    unique_batches = torch.unique(batch_idx)
    for batch in unique_batches:
        if batch == 0:
            continue
        other_batch = X[batch_idx == batch]
        loss = mmd_for_two_batches(reference_batch, reference_batch)
        loss -= 2 * mmd_for_two_batches(reference_batch, other_batch)
        loss += mmd_for_two_batches(other_batch, other_batch)
        mmd_loss += loss

    return mmd_loss


def mmd_for_two_batches(first, second):
    first_tiled = repeat(first, 'x_cell features -> x_cell y_cell features', y_cell=second.shape[0])
    second_tiled = repeat(second, 'y_cell features -> x_cell y_cell features', x_cell=first.shape[0])
    diff = -torch.mean((first_tiled - second_tiled)**2, dim=2)
    sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    result = 0.0
    for sigma in sigmas:
        result += torch.exp(diff / sigma).mean()

    return result / len(sigmas)


def _generate_sample(loc, std, shape):
    first_distribution = torch.distributions.normal.Normal(loc, std)
    return first_distribution.sample(shape)


if __name__ == "__main__":
    print("Testing calculate MMD loss...")
    utils.set_deafult_seed()
    first = _generate_sample(10.0, 1, [100, 20])
    second = _generate_sample(0.0, 1, [50, 20])
    third = _generate_sample(4.0, 0.01, [200, 20])
    X = torch.cat([first, second], dim=0)
    batch_idx = []
    for _ in range(first.shape[0]):
        batch_idx.append(0)
    for _ in range(second.shape[0]):
        batch_idx.append(1)
    #for _ in range(third.shape[0]):
    #    batch_idx.append(2)
    batch_idx = torch.tensor(batch_idx)  # type: ignore
    #new_idx = torch.randperm(X.shape[0])
    #X = X[new_idx]
    #batch_idx = batch_idx[new_idx]  # type: ignore
    print(calculate_mmd_loss(X, batch_idx))
