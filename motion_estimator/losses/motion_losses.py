"""Masked local and safety-aware scalar losses."""

import torch
import torch.nn.functional as F


def _mean_or_zero(values, reference):
    return values.mean() if values.numel() else reference.sum() * 0.0


def motion_loss(predictions, batch, cfg):
    loss_cfg = cfg
    pred_closing = predictions['closing']
    pred_drift = predictions['drift']
    closing = batch['closing_target']
    drift = batch['drift_target']
    continuous = batch['continuous_mask'].bool()
    zero_threshold = float(loss_cfg['zero_threshold'])
    zero_weight = float(loss_cfg['zero_weight'])
    drift_threshold = float(loss_cfg['drift_danger_threshold'])

    data_weight = torch.where(
        closing.abs() < zero_threshold,
        torch.full_like(closing, zero_weight),
        torch.ones_like(closing),
    )
    local_element = F.smooth_l1_loss(pred_closing, closing, reduction='none')
    local_weights = data_weight * continuous.float()
    local = (local_element * local_weights).sum() / (local_weights.sum() + 1.0e-6)

    under_mask = continuous & (closing > zero_threshold)
    under_penalty = torch.relu(closing - pred_closing)
    local_under = _mean_or_zero(
        F.smooth_l1_loss(under_penalty[under_mask], torch.zeros_like(under_penalty[under_mask]), reduction='none'),
        pred_closing,
    )

    drift_base = F.smooth_l1_loss(pred_drift, drift, reduction='mean')
    danger_mask = drift < -drift_threshold
    optimistic_penalty = torch.relu(pred_drift - drift)
    drift_optimistic = _mean_or_zero(
        F.smooth_l1_loss(
            optimistic_penalty[danger_mask],
            torch.zeros_like(optimistic_penalty[danger_mask]),
            reduction='none',
        ),
        pred_drift,
    )
    total = (
        float(loss_cfg['lambda_closing']) * local
        + float(loss_cfg['lambda_drift']) * drift_base
        + float(loss_cfg['lambda_closing_under']) * local_under
        + float(loss_cfg['lambda_drift_optimistic']) * drift_optimistic
    )
    return {
        'total': total,
        'closing': local,
        'drift': drift_base,
        'closing_under': local_under,
        'drift_optimistic': drift_optimistic,
        'continuous_ray_count': continuous.sum().detach(),
        'dangerous_sample_count': danger_mask.sum().detach(),
    }
