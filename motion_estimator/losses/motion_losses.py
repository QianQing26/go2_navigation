"""Masked local and safety-aware scalar losses."""

import torch
import torch.nn.functional as F


def _mean_or_zero(values, reference):
    return values.mean() if values.numel() else reference.sum() * 0.0


def _weighted_mean_or_zero(values, weights, reference):
    if not values.numel():
        return reference.sum() * 0.0
    weights = weights.to(dtype=values.dtype)
    return (values * weights).sum() / (weights.sum() + 1.0e-6)


def _drift_weights(drift_gt, cfg):
    drift_cfg = cfg.get('drift_weight', {})
    danger_threshold = float(
        drift_cfg.get('danger_threshold', -float(cfg.get('drift_danger_threshold', 0.05)))
    )
    severe_threshold = float(drift_cfg.get('severe_threshold', -float('inf')))
    normal_weight = float(drift_cfg.get('normal_weight', 1.0))
    danger_weight = float(drift_cfg.get('danger_weight', 1.0))
    severe_weight = float(drift_cfg.get('severe_weight', danger_weight))
    weights = torch.full_like(drift_gt, normal_weight)
    danger_mask = drift_gt < danger_threshold
    severe_mask = drift_gt <= severe_threshold
    weights = torch.where(danger_mask, torch.full_like(weights, danger_weight), weights)
    weights = torch.where(severe_mask, torch.full_like(weights, severe_weight), weights)
    return weights, danger_mask


def _danger_speed_weights(drift_gt, danger_mask, batch, cfg):
    speed_cfg = cfg.get('speed_weight', {})
    if not bool(speed_cfg.get('enabled', False)):
        return torch.ones_like(drift_gt)
    velocity = batch.get('trajectory_velocity_world')
    if velocity is None:
        return torch.ones_like(drift_gt)
    velocity = velocity.to(device=drift_gt.device, dtype=drift_gt.dtype)
    if velocity.ndim < 3 or velocity.shape[-1] != 2:
        return torch.ones_like(drift_gt)
    sample_speed = torch.linalg.vector_norm(velocity, dim=-1).amax(dim=-1)
    medium_threshold = float(speed_cfg.get('medium_threshold', 0.5))
    high_threshold = float(speed_cfg.get('high_threshold', 1.0))
    medium_weight = float(speed_cfg.get('medium', 1.0))
    high_weight = float(speed_cfg.get('high', 1.0))
    speed_weight = torch.where(
        sample_speed >= high_threshold,
        torch.full_like(sample_speed, high_weight),
        torch.where(
            sample_speed >= medium_threshold,
            torch.full_like(sample_speed, medium_weight),
            torch.ones_like(sample_speed),
        ),
    )
    # Privileged speed metadata affects only dangerous scalar samples.
    return torch.where(danger_mask, speed_weight, torch.ones_like(speed_weight))


def motion_loss(predictions, batch, cfg):
    loss_cfg = cfg
    pred_closing = predictions['closing']
    pred_drift = predictions['drift']
    # Regression targets are normalized for optimization, while all safety
    # masks must remain in their physical units (m/s).
    closing = batch['closing_target']
    drift = batch['drift_target']
    closing_gt = batch['closing_gt'].to(dtype=closing.dtype)
    drift_gt = batch['lse_drift_gt'].to(dtype=drift.dtype)
    continuous = batch['continuous_mask'].bool()
    zero_threshold = float(loss_cfg['zero_threshold'])
    zero_weight = float(loss_cfg['zero_weight'])

    zero_mask = closing_gt.abs() < zero_threshold
    approaching_mask = continuous & (closing_gt > zero_threshold)
    approaching_weight = float(loss_cfg.get('approaching_weight', 1.0))
    data_weight = torch.where(
        zero_mask,
        torch.full_like(closing, zero_weight),
        torch.ones_like(closing),
    )
    local_element = F.smooth_l1_loss(pred_closing, closing, reduction='none')
    local_weights = data_weight * continuous.to(dtype=closing.dtype)
    local_weights = local_weights * torch.where(
        approaching_mask,
        torch.full_like(local_weights, approaching_weight),
        torch.ones_like(local_weights),
    )
    local = (local_element * local_weights).sum() / (local_weights.sum() + 1.0e-6)

    under_mask = approaching_mask
    under_penalty = torch.relu(closing - pred_closing)
    under_element = F.smooth_l1_loss(
        under_penalty[under_mask],
        torch.zeros_like(under_penalty[under_mask]),
        reduction='none',
    )
    local_under = _mean_or_zero(
        under_element * approaching_weight,
        pred_closing,
    )

    drift_element = F.smooth_l1_loss(pred_drift, drift, reduction='none')
    drift_weight, danger_mask = _drift_weights(drift_gt, loss_cfg)
    speed_weight = _danger_speed_weights(drift_gt, danger_mask, batch, loss_cfg)
    scalar_weight = drift_weight * speed_weight
    drift_base = (drift_element * scalar_weight).sum() / (scalar_weight.sum() + 1.0e-6)
    optimistic_penalty = torch.relu(pred_drift - drift)
    optimistic_element = F.smooth_l1_loss(
        optimistic_penalty[danger_mask],
        torch.zeros_like(optimistic_penalty[danger_mask]),
        reduction='none',
    )
    drift_optimistic = _weighted_mean_or_zero(
        optimistic_element,
        scalar_weight[danger_mask],
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
        'zero_sample_count': zero_mask.sum().detach(),
        'approaching_ray_count': approaching_mask.sum().detach(),
        'continuous_ray_count': continuous.sum().detach(),
        'dangerous_sample_count': danger_mask.sum().detach(),
    }
