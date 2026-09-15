"""Pure execution-path helpers for the Phase-4.5 causal audit.

The simulator evaluator owns all Isaac Gym state and terminal bookkeeping.  This
module keeps the stage reconstruction and residual/event definitions CPU-testable
and exactly aligned with :class:`ExactLSECBFLayer`.
"""

import math

import torch


STAGE_NAMES = ('cbf', 'rawclip', 'filter', 'cmd')
DEGRADATION_NAMES = ('rawclip', 'filter', 'bound')
EVENT_NAMES = ('raw_clamp', 'filter_change', 'bound_clip', 'track')


def reconstruct_execution_stages(u_cbf, previous_filter, beta, nav_clip_min, nav_clip_max):
    """Reconstruct the exact navigation execution chain without mutating env state."""

    beta = float(beta)
    u_rawclip = torch.clamp(u_cbf, min=-3.0, max=3.0)
    u_filter = beta * u_rawclip + (1.0 - beta) * previous_filter
    u_cmd = torch.maximum(torch.minimum(u_filter, nav_clip_max), nav_clip_min)
    return {
        'cbf': u_cbf,
        'rawclip': u_rawclip,
        'filter': u_filter,
        'cmd': u_cmd,
    }


def stage_deltas(stages):
    """Return vector and scalar deltas introduced by each execution stage."""

    return {
        'raw_clamp': stages['rawclip'] - stages['cbf'],
        'filter': stages['filter'] - stages['rawclip'],
        'bound': stages['cmd'] - stages['filter'],
    }


def barrier_residual(drift, lgh, alpha_h, action):
    """Evaluate r(u,d)=d + Lgh*u_xy + alpha*h for one action stage."""

    drift = drift.reshape(-1, 1)
    alpha_h = alpha_h.reshape(-1, 1)
    return drift + torch.sum(lgh * action[:, :2], dim=-1, keepdim=True) + alpha_h


def stage_residuals(drift, lgh, alpha_h, stages):
    return {
        name: barrier_residual(drift, lgh, alpha_h, stages[name])
        for name in STAGE_NAMES
    }


def violation_events(residuals):
    """Return one-way safety-margin losses at raw/filter/bound stages."""

    return {
        'raw_clamp': (residuals['cbf'] >= 0.0) & (residuals['rawclip'] < 0.0),
        'filter': (residuals['rawclip'] >= 0.0) & (residuals['filter'] < 0.0),
        'bound': (residuals['filter'] >= 0.0) & (residuals['cmd'] < 0.0),
    }


def tracking_violation(residual_cmd, residual_actual_post):
    return (residual_cmd >= 0.0) & (residual_actual_post < 0.0)


def vector_norm(value):
    return torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def percentile(values, fraction):
    """Small dependency-free linear percentile helper."""

    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return float('nan')
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def distribution(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {
            'count': 0, 'mean': float('nan'), 'median': float('nan'),
            'p10': float('nan'), 'p50': float('nan'), 'p90': float('nan'),
            'min': float('nan'), 'max': float('nan'),
        }
    return {
        'count': len(values),
        'mean': sum(values) / len(values),
        'median': percentile(values, 0.5),
        'p10': percentile(values, 0.1),
        'p50': percentile(values, 0.5),
        'p90': percentile(values, 0.9),
        'min': min(values),
        'max': max(values),
    }


def shadow_reprojection(layer, nominal_filter, shield_rays, alpha, drift, nav_clip_min, nav_clip_max):
    """Stateless CBF re-projection used only for diagnosis, never execution."""

    projected = layer.project(
        nominal_filter, shield_rays, alpha, safety_drift=drift
    )
    action = projected['safe_action']
    residual = barrier_residual(
        drift, projected['Lgh'], projected['alpha_h'], action
    )
    out_of_bounds = (action < nav_clip_min) | (action > nav_clip_max)
    return {
        'action': action,
        'residual': residual,
        'intervention_norm': projected['intervention_norm'],
        'out_of_bounds': out_of_bounds.any(dim=-1, keepdim=True),
    }
