import torch
from lab_scripts.utils import utils
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from torch.distributions.log_normal import LogNormal
from torch.distributions.negative_binomial import NegativeBinomial


def lognorm_loss(predictied_parameters, targets):
    eps = 1e-6
    loc, scale = predictied_parameters
    exp_distribution = LogNormal(loc, scale)
    exp_loss = exp_distribution.log_prob(targets + eps).mean()
    return -exp_loss


def zinb_loss(predicted_parameters, targets):
    zinb_r, zinb_p, dropout = predicted_parameters
    zinb_distribution = ZeroInflatedNegativeBinomial(zinb_r, probs=zinb_p, gate=dropout)
    log_loss = zinb_distribution.log_prob(targets).mean()
    return -log_loss


def nb_loss(predicted_parameters, targets):
    nb_r, nb_p = predicted_parameters
    nb_distribution = NegativeBinomial(nb_r, nb_p)
    log_loss = nb_distribution.log_prob(targets).mean()
    return -log_loss


def weighted_mse(predicted, true, weights):
    difference = (predicted - true) ** 2
    difference = torch.unsqueeze(weights, dim=-1) * difference
    return difference.mean()


def get_loss(loss_name: str):
    if loss_name == "mse":
        return weighted_mse
    elif loss_name == "nb":
        return nb_loss
    elif loss_name == "zinb":
        return zinb_loss
    elif loss_name == "lognorm":
        return lognorm_loss


def calculate_mmd_loss(X, batch_idx):
    mmd_loss = 0.0
    reference_batch = X[batch_idx == 0]
    reference_mmd = mmd_for_two_batches(reference_batch, reference_batch)
    unique_batches = torch.unique(batch_idx)
    for batch in unique_batches:
        if batch == 0:
            continue
        other_batch = X[batch_idx == batch]
        loss = reference_mmd
        loss -= 2 * mmd_for_two_batches(reference_batch, other_batch)
        loss += mmd_for_two_batches(other_batch, other_batch)
        mmd_loss += loss

    return -mmd_loss


def mmd_for_two_batches(first, second):
    result = 0.0
    if first.shape[0] == 0 or second.shape[0] == 0:
        return result
    for first_row in first:
        diff = second - first_row
        dist = (diff ** 2).sum(axis=1)  # **(0.5)
        result += dist.sum()
        # result += (diff ** 2).sum()  # squared distance between first_row and each row
    result = result / (first.shape[0] * second.shape[0])
    return result


def _generate_sample(loc, std, shape):
    first_distribution = torch.distributions.normal.Normal(loc, std)
    return first_distribution.sample(shape)


if __name__ == "__main__":
    print("Testing calculate MMD loss...")
    utils.set_deafult_seed()
    first = _generate_sample(0.0, 0.01, [50, 20])
    second = _generate_sample(10.0, 0.01, [100, 20])
    third = _generate_sample(4.0, 0.01, [200, 20])
    X = torch.cat([first, second], dim=0)
    batch_idx = []
    for _ in range(first.shape[0]):
        batch_idx.append(0)
    for _ in range(second.shape[0]):
        batch_idx.append(1)
    for _ in range(third.shape[0]):
        batch_idx.append(2)
    batch_idx = torch.tensor(batch_idx)  # type: ignore
    new_idx = torch.randperm(X.shape[0])
    X = X[new_idx]
    batch_idx = batch_idx[new_idx]  # type: ignore
    print(calculate_mmd_loss(X, batch_idx))
