import numpy as np
import torch


def make_quantile_grid(M, device=None, dtype=torch.float32):
    j = torch.arange(1, M + 1, device=device, dtype=dtype)
    return (j - 0.5) / M


def huber_loss(u, k):
    abs_u = torch.abs(u)
    quad = 0.5 * u * u
    lin = k * (abs_u - 0.5 * k)
    return torch.where(abs_u <= k, quad, lin)


def quantile_huber_loss(y, z, taus, k):
    # y: (B, M), z: (B, M)
    u = y.unsqueeze(1) - z.unsqueeze(2)  # (B, M, M)
    indicator = (u < 0).to(u.dtype)
    taus = taus.view(1, -1, 1).to(u.device).to(u.dtype)
    weight = torch.abs(taus - indicator)
    L = huber_loss(u, k)
    return (weight * L).mean()


def sort_quantiles(quantiles):
    if quantiles.dim() == 1:
        return torch.sort(quantiles, dim=0).values
    return torch.sort(quantiles, dim=1).values


def var_from_quantiles(quantiles, alpha=0.95):
    q = quantiles
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = sort_quantiles(q)

    _, M = q.shape
    idx = int(np.ceil(alpha * M) - 1)
    idx = max(0, min(M - 1, idx))
    var = q[:, idx]
    return var.squeeze(0) if quantiles.dim() == 1 else var


def cvar_from_quantiles(quantiles, alpha=0.95):
    q = quantiles
    if q.dim() == 1:
        q = q.unsqueeze(0)
    q = sort_quantiles(q)

    _, M = q.shape
    idx = int(np.ceil(alpha * M) - 1)
    idx = max(0, min(M - 1, idx))
    tail = q[:, idx:]
    cvar = tail.mean(dim=1)
    return cvar.squeeze(0) if quantiles.dim() == 1 else cvar


def mean_std_objective_from_quantiles(quantiles, lambda_std=1.645):
    if quantiles.dim() == 1:
        quantiles = quantiles.unsqueeze(0)
    mean = quantiles.mean(dim=1)
    var = (quantiles - mean.unsqueeze(1)).pow(2).mean(dim=1)
    std = torch.sqrt(var + 1e-12)
    obj = mean + lambda_std * std
    return obj.squeeze(0) if obj.shape[0] == 1 else obj


def objective_from_quantiles(quantiles, objective, alpha=0.95, lambda_std=1.645):
    obj = objective.lower()
    if obj == "mean_std":
        return mean_std_objective_from_quantiles(quantiles, lambda_std=lambda_std)
    if obj == "var":
        return var_from_quantiles(quantiles, alpha=alpha)
    if obj == "cvar":
        return cvar_from_quantiles(quantiles, alpha=alpha)
    raise ValueError("Unknown objective. Choose from {'mean_std','var','cvar'}.")
