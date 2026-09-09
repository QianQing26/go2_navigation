"""Evaluation metrics for local closing and scalar drift predictions."""

import math

import numpy as np
import torch


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _corr(x, y):
    x, y = x.reshape(-1).float(), y.reshape(-1).float()
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    if x.numel() < 2:
        return {'count': int(x.numel()), 'pearson': None, 'spearman': None}
    xc, yc = x - x.mean(), y - y.mean()
    denominator = torch.sqrt(xc.square().sum() * yc.square().sum())
    pearson = (xc * yc).sum() / denominator if denominator > 0 else torch.tensor(float('nan'))
    spearman = None
    try:
        from scipy.stats import spearmanr
        spearman = spearmanr(x.numpy(), y.numpy()).statistic
    except (ImportError, AttributeError, ValueError):
        pass
    return {
        'count': int(x.numel()),
        'pearson': _number(pearson),
        'spearman': _number(spearman) if spearman is not None else None,
    }


def regression_stats(pred, target, sign_epsilon=0.05, delta=0.1):
    pred, target = pred.reshape(-1).float(), target.reshape(-1).float()
    finite = torch.isfinite(pred) & torch.isfinite(target)
    pred, target = pred[finite], target[finite]
    if pred.numel() == 0:
        return {'count': 0}
    error = pred - target
    sign_mask = target.abs() > sign_epsilon
    dangerous = target > sign_epsilon
    return {
        'count': int(pred.numel()),
        'mae': _number(error.abs().mean()),
        'rmse': _number(torch.sqrt(error.square().mean())),
        'bias_pred_minus_gt': _number(error.mean()),
        'correlation': _corr(pred, target),
        'sign_accuracy': (
            _number((torch.sign(pred[sign_mask]) == torch.sign(target[sign_mask])).float().mean())
            if sign_mask.any() else None
        ),
        'dangerous_underestimation_rate': (
            _number((pred[dangerous] < target[dangerous] - delta).float().mean())
            if dangerous.any() else None
        ),
        'static_leakage_mean_abs_pred': (
            _number(pred[~sign_mask].abs().mean()) if (~sign_mask).any() else None
        ),
        'static_leakage_count': int((~sign_mask).sum()),
    }


def scalar_stats(pred, target, sign_epsilon=0.05, delta=0.1):
    result = regression_stats(pred, target, sign_epsilon, delta)
    if not result.get('count'):
        return result
    pred, target = pred.reshape(-1).float(), target.reshape(-1).float()
    danger = target < -sign_epsilon
    result['false_safe_rate'] = (
        _number((pred[danger] >= 0.0).float().mean()) if danger.any() else None
    )
    result['optimistic_danger_rate'] = (
        _number((pred[danger] > target[danger] + delta).float().mean())
        if danger.any() else None
    )
    result['dangerous_count'] = int(danger.sum())
    return result


def zero_baseline(batch_iter, sign_epsilon=0.05, delta=0.1):
    closing, drift = [], []
    for batch in batch_iter:
        closing.append(batch['closing_gt'])
        drift.append(batch['lse_drift_gt'])
    closing, drift = torch.cat(closing), torch.cat(drift)
    return {
        'closing_zero': regression_stats(torch.zeros_like(closing), closing, sign_epsilon, delta),
        'drift_zero': scalar_stats(torch.zeros_like(drift), drift, sign_epsilon, delta),
    }
