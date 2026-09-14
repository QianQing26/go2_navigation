"""Pure helpers for Phase-4 residual-failure and oracle analysis."""

import math

import torch


PHASE4_MODES = (
    'predictive_learned',
    'oracle_gt_0p1',
    'oracle_gt_0p2',
    'oracle_gt_0p3',
    'oracle_gt_0p5',
    'oracle_gt_multi',
)
ORACLE_HORIZONS = (0.1, 0.2, 0.3, 0.5)


def mode_spec(mode):
    """Return the drift source and horizon set for one Phase-4 mode."""

    if mode not in PHASE4_MODES:
        raise ValueError('unknown Phase-4 mode: {}'.format(mode))
    if mode == 'predictive_learned':
        return {
            'safety_drift_source': 'learned',
            'oracle_horizon_s': None,
            'oracle_multi_horizon': False,
            'oracle_horizon_set': [],
        }
    if mode == 'oracle_gt_multi':
        return {
            'safety_drift_source': 'gt',
            'oracle_horizon_s': None,
            'oracle_multi_horizon': True,
            'oracle_horizon_set': list(ORACLE_HORIZONS),
        }
    horizon = {
        'oracle_gt_0p1': 0.1,
        'oracle_gt_0p2': 0.2,
        'oracle_gt_0p3': 0.3,
        'oracle_gt_0p5': 0.5,
    }[mode]
    return {
        'safety_drift_source': 'gt',
        'oracle_horizon_s': horizon,
        'oracle_multi_horizon': False,
        'oracle_horizon_set': [horizon],
    }


def select_multi_horizon_drift(drift_by_horizon):
    """Return the minimum finite-horizon drift and its selected horizon."""

    if not drift_by_horizon:
        raise ValueError('drift_by_horizon must not be empty')
    horizon, drift = min(
        drift_by_horizon.items(), key=lambda item: float(item[1])
    )
    return drift, float(horizon)


def lse_drift_from_fused_rays(current_fused_rays, future_fused_rays, horizon, d_safe, kappa):
    """Compute the finite-horizon drift used by the live GT label."""

    horizon = float(horizon)
    if horizon <= 0.0:
        raise ValueError('horizon must be positive')
    current_h = current_fused_rays - float(d_safe)
    future_h = future_fused_rays - float(d_safe)
    current_lse = -torch.logsumexp(-float(kappa) * current_h, dim=-1, keepdim=True) / float(kappa)
    future_lse = -torch.logsumexp(-float(kappa) * future_h, dim=-1, keepdim=True) / float(kappa)
    return (future_lse - current_lse) / horizon


def false_safe_decision(r_hat, r_gt):
    """Whether learned CBF residual is safe while GT residual is unsafe."""

    return float(r_hat) >= 0.0 and float(r_gt) < 0.0


def warning_time(collision_time_s, rows, residual_field):
    """Return collision time minus first sample with negative residual."""

    candidates = [
        float(row['time_s']) for row in rows
        if float(row.get(residual_field, 0.0)) < 0.0
    ]
    if not candidates:
        return float('nan')
    return float(collision_time_s) - min(candidates)


def extract_pre_collision_window(rows, collision_time_s, max_samples=100):
    """Copy the last control samples before a collision and annotate timing."""

    selected = list(rows)[-int(max_samples):]
    result = []
    for row in selected:
        item = dict(row)
        item['time_to_collision_s'] = (
            float(collision_time_s) - float(item['time_s'])
        )
        result.append(item)
    return result


def _safe_mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else float('nan')


def classify_failure(metrics):
    """Assign non-exclusive descriptive Phase-4 failure tags.

    The thresholds are intentionally conservative heuristics.  Continuous
    metrics remain authoritative; these labels are only a compact summary for
    the final report.
    """

    tags = []
    if metrics.get('num_false_safe_updates', 0) > 0 and metrics.get(
        'learned_detection_delay_s', float('nan')
    ) > 0.0:
        tags.append('F1_estimator_decision_miss')
    if (
        metrics.get('dangerous_drift_mae', 0.0) > 0.05
        and metrics.get('mean_oracle_extra_intervention_0p1', 0.0) > 1.0e-3
    ):
        tags.append('F2_estimator_magnitude_underestimate')
    warning_01 = metrics.get('warning_time_0p1', float('nan'))
    warning_03 = metrics.get('warning_time_0p3', float('nan'))
    if (
        (math.isnan(warning_01) or warning_01 < 0.1)
        and not math.isnan(warning_03)
        and warning_03 >= 0.1
    ):
        tags.append('F3_short_horizon')
    if (
        metrics.get('clip_fraction', 0.0) > 0.05
        or metrics.get('mean_tracking_error', 0.0) > 0.15
    ):
        tags.append('F4_execution_control_authority')
    if not tags:
        tags.append('F5_geometry_or_residual')
    return tags


def primary_failure_label(tags):
    """Return the requested heuristic precedence summary."""

    precedence = (
        'F1_estimator_decision_miss',
        'F2_estimator_magnitude_underestimate',
        'F3_short_horizon',
        'F4_execution_control_authority',
        'F5_geometry_or_residual',
    )
    for label in precedence:
        if label in tags:
            return label
    return 'F5_geometry_or_residual'


def aggregate_numeric(rows, field):
    """Return mean/std/range for a numeric field without numpy dependency."""

    values = [
        float(row[field]) for row in rows
        if field in row and str(row[field]).lower() not in {'', 'nan'}
    ]
    if not values:
        return {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan')}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        'mean': mean,
        'std': math.sqrt(variance),
        'min': min(values),
        'max': max(values),
    }


def mean_or_nan(values):
    return _safe_mean(values)
