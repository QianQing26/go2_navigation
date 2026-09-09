"""Evaluation metrics for local closing and scalar drift predictions."""

import math

import torch


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def error_quantiles(pred, target, quantiles=(0.50, 0.90, 0.95, 0.99, 0.995, 0.999)):
    pred, target = pred.reshape(-1).float(), target.reshape(-1).float()
    finite = torch.isfinite(pred) & torch.isfinite(target)
    error = (pred[finite] - target[finite]).abs()
    if not error.numel():
        return {('p{:g}'.format(q * 100)): None for q in quantiles}
    values = torch.quantile(error, torch.as_tensor(quantiles, dtype=error.dtype))
    return {
        'p{:g}'.format(q * 100): _number(value)
        for q, value in zip(quantiles, values)
    }


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
    optimistic = danger & (pred > target)
    optimistic_error = (pred - target)[optimistic]
    result['optimistic_danger_rate_any'] = (
        _number(optimistic.float().mean()) if danger.any() else None
    )
    result['optimistic_error_mean'] = (
        _number(optimistic_error.mean()) if optimistic_error.numel() else None
    )
    result['optimistic_error_p95'] = (
        _number(torch.quantile(optimistic_error, 0.95))
        if optimistic_error.numel() else None
    )
    result['dangerous_count'] = int(danger.sum())
    result['error_quantiles'] = error_quantiles(pred, target)
    return result


def local_dynamic_stats(pred, target, mask):
    """Report error and underestimation statistics for a local subset."""
    pred, target = pred.reshape(-1).float(), target.reshape(-1).float()
    mask = mask.reshape(-1).bool()
    pred, target = pred[mask], target[mask]
    result = regression_stats(pred, target, sign_epsilon=0.05, delta=0.1)
    if not result.get('count'):
        result.update({
            'underestimation_rate': None,
            'mean_underestimation_magnitude': None,
            'p95_absolute_error': None,
            'error_quantiles': error_quantiles(pred, target),
        })
        return result
    error = pred - target
    under = error < 0.0
    result['underestimation_rate'] = _number(under.float().mean())
    result['mean_underestimation_magnitude'] = (
        _number((-error[under]).mean()) if under.any() else 0.0
    )
    result['p95_absolute_error'] = _number(torch.quantile(error.abs(), 0.95))
    result['error_quantiles'] = error_quantiles(pred, target)
    return result


def scalar_danger_breakdown(pred, target, danger_threshold=-0.05, severe_threshold=-0.5,
                            sign_epsilon=0.05, delta=0.1):
    pred, target = pred.reshape(-1).float(), target.reshape(-1).float()
    masks = {
        'safe_neutral': target >= danger_threshold,
        'mild_danger': (target < danger_threshold) & (target > severe_threshold),
        'severe_danger': target <= severe_threshold,
    }
    result = {}
    for name, mask in masks.items():
        result[name] = scalar_stats(pred[mask], target[mask], sign_epsilon, delta)
    return result


def speed_danger_breakdown(pred, target, speed, danger_threshold=-0.05,
                           severe_threshold=-0.5, sign_epsilon=0.05, delta=0.1):
    """Break scalar safety metrics down by max obstacle speed and danger level."""
    pred, target = pred.reshape(-1).float(), target.reshape(-1).float()
    speed = speed.reshape(-1).float()
    bins = {
        'low_speed': (speed < 0.5),
        'medium_speed': (speed >= 0.5) & (speed < 1.0),
        'high_speed': speed >= 1.0,
    }
    danger = {
        'mild_danger': (target < danger_threshold) & (target > severe_threshold),
        'severe_danger': target <= severe_threshold,
    }
    result = {}
    for speed_name, speed_mask in bins.items():
        for danger_name, danger_mask in danger.items():
            mask = speed_mask & danger_mask
            stats = scalar_stats(pred[mask], target[mask], sign_epsilon, delta)
            result['{}__{}'.format(speed_name, danger_name)] = stats
        danger_mask = speed_mask & (target < danger_threshold)
        result['{}__danger'.format(speed_name)] = scalar_stats(
            pred[danger_mask], target[danger_mask], sign_epsilon, delta
        )
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
