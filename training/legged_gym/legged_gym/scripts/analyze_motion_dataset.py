"""Research analysis for raw and refined motion-estimation targets.

The script never changes the formal training target.  In particular,
``closing_rate_gt`` remains the raw hard-min finite-difference target.  v2
datasets additionally contain analytic trajectory velocity, active source
ids, source transitions, and radial velocity diagnostics.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


BASE_REQUIRED_KEYS = {
    'rays_hist', 'motion_ego_hist', 'closing_rate_gt',
    'current_fused_rays', 'static_rays', 'dynamic_rays',
    'future_fused_rays', 'future_dynamic_rays',
    'dynamic_ray_hit_mask', 'obstacle_velocity_world',
    'obstacle_position_world', 'robot_xy_world', 'robot_yaw',
    'gt_horizon', 'env_id', 'episode_id', 'episode_step',
    'global_policy_step',
}
V2_REQUIRED_KEYS = {
    'obstacle_trajectory_velocity_world',
    'obstacle_physics_velocity_world',
    'active_dynamic_obstacle_id', 'future_active_dynamic_obstacle_id',
    'current_fused_source_id', 'future_fused_source_id',
    'source_switch_mask', 'same_dynamic_source_mask',
    'source_switch_type', 'active_obstacle_radial_velocity_gt',
    'active_obstacle_radial_velocity_valid_mask',
}
SWITCH_NAMES = {
    0: 'no_switch',
    1: 'static_to_dynamic',
    2: 'dynamic_to_static',
    3: 'dynamic_to_dynamic',
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--d_safe', type=float, default=0.20)
    parser.add_argument('--kappa', type=float, default=10.0)
    parser.add_argument(
        '--speed_bins', type=float, nargs='+',
        default=[0.2, 0.5, 1.0, 1.5],
        help='Bin edges; the final bin includes its right edge.',
    )
    return parser.parse_args()


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _distribution(values, percentiles=None):
    values = values.reshape(-1).float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {'count': 0}
    if percentiles is None:
        percentiles = [90.0, 95.0, 99.0, 99.9]
    quantile_values = torch.quantile(
        values, torch.tensor(percentiles, dtype=values.dtype) / 100.0
    )
    result = {
        'count': int(values.numel()),
        'mean': _json_number(values.mean()),
        'std': _json_number(values.std(unbiased=False)),
        'min': _json_number(values.min()),
        'max': _json_number(values.max()),
        'median': _json_number(values.median()),
    }
    for percentile, quantile in zip(percentiles, quantile_values):
        name = (
            str(int(percentile))
            if float(percentile).is_integer()
            else str(percentile).replace('.', '_')
        )
        result['p{}'.format(name)] = _json_number(quantile)
    return result


def _ratio_stats(values, epsilon):
    values = values.reshape(-1).float()
    count = max(int(values.numel()), 1)
    positive = values > epsilon
    negative = values < -epsilon
    near_zero = values.abs() <= epsilon
    return {
        'epsilon': float(epsilon),
        'positive_ratio': float(positive.sum()) / count,
        'negative_ratio': float(negative.sum()) / count,
        'near_zero_ratio': float(near_zero.sum()) / count,
    }


def _load_dataset(dataset_dir):
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    manifest_path = os.path.join(dataset_dir, 'manifest.json')
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path) as file:
            manifest = json.load(file)
    if manifest is not None and manifest.get('shards'):
        paths = [
            os.path.join(dataset_dir, shard['path'])
            for shard in manifest['shards']
        ]
    else:
        paths = sorted(glob.glob(os.path.join(dataset_dir, 'shard_*.pt')))
    if not paths:
        raise FileNotFoundError('No dataset shards found in {}'.format(dataset_dir))

    chunks = {key: [] for key in BASE_REQUIRED_KEYS}
    optional_chunks = {key: [] for key in V2_REQUIRED_KEYS}
    total = 0
    reference_shapes = None
    optional_presence = None
    for path in paths:
        shard = torch.load(path, map_location='cpu')
        missing = BASE_REQUIRED_KEYS.difference(shard.keys())
        if missing:
            raise RuntimeError('{} is missing keys {}'.format(path, sorted(missing)))
        num_samples = int(shard['closing_rate_gt'].shape[0])
        if num_samples < 1:
            raise RuntimeError('Empty shard: {}'.format(path))
        current_optional = V2_REQUIRED_KEYS.intersection(shard.keys())
        if optional_presence is None:
            optional_presence = current_optional
        elif current_optional != optional_presence:
            raise RuntimeError('{} has inconsistent v2 field presence'.format(path))
        for key in BASE_REQUIRED_KEYS:
            if int(shard[key].shape[0]) != num_samples:
                raise RuntimeError('{} has inconsistent leading shapes'.format(path))
            chunks[key].append(shard[key])
        for key in current_optional:
            if int(shard[key].shape[0]) != num_samples:
                raise RuntimeError('{} has inconsistent leading shapes'.format(path))
            optional_chunks[key].append(shard[key])
        sample_shapes = {
            key: list(shard[key].shape[1:]) for key in BASE_REQUIRED_KEYS
        }
        if reference_shapes is None:
            reference_shapes = sample_shapes
        elif sample_shapes != reference_shapes:
            raise RuntimeError('{} has inconsistent tensor shapes'.format(path))
        total += num_samples

    data = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    for key, values in optional_chunks.items():
        if values:
            data[key] = torch.cat(values, dim=0)
    if int(data['closing_rate_gt'].shape[0]) != total:
        raise RuntimeError('Dataset sample count mismatch')
    num_rays = int(data['closing_rate_gt'].shape[1])
    if num_rays != 41 or tuple(data['rays_hist'].shape[1:]) != (
        int(data['rays_hist'].shape[1]), 41
    ):
        raise RuntimeError('rays_hist does not have [N, H, 41] layout')
    if tuple(data['motion_ego_hist'].shape[1:]) != (
        int(data['motion_ego_hist'].shape[1]), 3
    ):
        raise RuntimeError('motion_ego_hist does not have [N, H, 3] layout')
    for key in [
        'current_fused_rays', 'static_rays', 'dynamic_rays',
        'future_fused_rays', 'future_dynamic_rays',
        'dynamic_ray_hit_mask',
    ]:
        if tuple(data[key].shape[1:]) != (41,):
            raise RuntimeError('{} does not have [N, 41] layout'.format(key))
    return dataset_dir, manifest, data, paths, optional_presence or set()


def _check_sensor_frame_sampling(data, manifest):
    if manifest is None:
        return {'checked': False, 'reason': 'manifest.json not found'}
    interval = int(manifest.get('exteroception_update_interval', 0))
    if interval < 1:
        return {'checked': False, 'reason': 'update interval not recorded'}
    grouped = {}
    for env_id, episode_id, step in zip(
        data['env_id'].tolist(), data['episode_id'].tolist(),
        data['global_policy_step'].tolist(),
    ):
        grouped.setdefault((int(env_id), int(episode_id)), []).append(int(step))
    deltas = []
    violations = []
    for key, values in grouped.items():
        values = sorted(values)
        group_deltas = np.diff(values)
        deltas.extend(group_deltas.tolist())
        if np.any(group_deltas < interval):
            violations.append(key)
    result = {
        'checked': True,
        'expected_policy_step_interval': interval,
        'num_trajectory_groups': len(grouped),
        'num_interval_violations': len(violations),
        'min_policy_step_delta': int(min(deltas)) if deltas else None,
        'median_policy_step_delta': float(np.median(deltas)) if deltas else None,
    }
    if violations:
        raise RuntimeError(
            'dataset contains repeated/non-10Hz frames in {} groups'.format(
                len(violations)
            )
        )
    return result


def _check_splits(dataset_dir, data):
    path = os.path.join(dataset_dir, 'splits.json')
    if not os.path.isfile(path):
        return {'checked': False, 'reason': 'splits.json not found'}
    with open(path) as file:
        splits = json.load(file)
    owners = {}
    for split, ids in splits.get('episode_ids', {}).items():
        for episode_id in ids:
            episode_id = int(episode_id)
            if episode_id in owners:
                raise RuntimeError('episode appears in multiple dataset splits')
            owners[episode_id] = split
    present = set(int(value) for value in torch.unique(data['episode_id']))
    if present != set(owners):
        raise RuntimeError('splits.json does not cover exactly the dataset episodes')
    return {
        'checked': True,
        'episode_counts': splits.get('episode_counts', {}),
        'sample_counts': splits.get('sample_counts', {}),
    }


def _derive_source_fields(data, epsilon):
    """Derive source topology available in v1; obstacle identity is unknown."""
    current_dynamic = data['dynamic_rays'] < data['static_rays'] - epsilon
    future_dynamic = data['future_dynamic_rays'] < data['static_rays'] - epsilon
    zeros = torch.zeros_like(data['closing_rate_gt'], dtype=torch.long)
    minus_ones = torch.full_like(data['closing_rate_gt'], -1, dtype=torch.long)
    current = torch.where(current_dynamic, zeros, minus_ones)
    future = torch.where(future_dynamic, zeros, minus_ones)
    switch = current != future
    same_dynamic = current_dynamic & future_dynamic & (current == future)
    switch_type = torch.zeros_like(current)
    switch_type[(~current_dynamic) & future_dynamic] = 1
    switch_type[current_dynamic & (~future_dynamic)] = 2
    switch_type[current_dynamic & future_dynamic & switch] = 3
    return current, future, switch, same_dynamic, switch_type


def _source_fields(data, epsilon):
    if V2_REQUIRED_KEYS.issubset(data):
        return (
            data['current_fused_source_id'].long(),
            data['future_fused_source_id'].long(),
            data['source_switch_mask'].bool(),
            data['same_dynamic_source_mask'].bool(),
            data['source_switch_type'].long(),
            True,
        )
    current, future, switch, same_dynamic, switch_type = _derive_source_fields(
        data, epsilon
    )
    return current, future, switch, same_dynamic, switch_type, False


def _per_angle_stats(closing, epsilon):
    stats = []
    for angle_index in range(closing.shape[1]):
        values = closing[:, angle_index]
        positive = values > epsilon
        negative = values < -epsilon
        nonzero = values.abs() > epsilon
        positive_values = values[positive]
        stats.append({
            'ray_index': angle_index,
            'nonzero_rate': float(nonzero.float().mean()),
            'positive_rate': float(positive.float().mean()),
            'negative_rate': float(negative.float().mean()),
            'mean_positive_closing_rate': (
                _json_number(positive_values.mean())
                if positive_values.numel() else None
            ),
            'p95_abs': _json_number(
                torch.quantile(values.abs(), torch.tensor(0.95))
            ),
        })
    return stats


def _corr_stats(x, y):
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    if x.numel() < 2:
        return {
            'count': int(x.numel()), 'pearson': None,
            'spearman': None, 'spearman_status': 'insufficient samples',
        }
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt(x_centered.square().sum() * y_centered.square().sum())
    pearson = _json_number((x_centered * y_centered).sum() / denominator) \
        if denominator > 0 else None
    try:
        from scipy.stats import spearmanr
        spearman = spearmanr(x.numpy(), y.numpy()).statistic
        spearman_status = 'scipy'
    except (ImportError, AttributeError, ValueError):
        spearman = None
        spearman_status = 'scipy unavailable'
    return {
        'count': int(x.numel()),
        'pearson': pearson,
        'spearman': _json_number(spearman) if spearman is not None else None,
        'spearman_status': spearman_status,
    }


def _agreement_stats(raw, radial):
    raw = raw.reshape(-1).float()
    radial = radial.reshape(-1).float()
    finite = torch.isfinite(raw) & torch.isfinite(radial)
    raw, radial = raw[finite], radial[finite]
    if raw.numel() == 0:
        return {'count': 0}
    error = raw - radial
    return {
        'count': int(raw.numel()),
        'correlation': _corr_stats(raw, radial),
        'mae': _json_number(error.abs().mean()),
        'rmse': _json_number(torch.sqrt(error.square().mean())),
        'bias_raw_minus_radial': _json_number(error.mean()),
        'raw_distribution': _distribution(raw),
        'radial_distribution': _distribution(radial),
    }


def _trajectory_speed_validation(data, manifest):
    if 'obstacle_trajectory_velocity_world' not in data:
        physical = torch.linalg.vector_norm(
            data['obstacle_velocity_world'].float(), dim=-1
        ).reshape(-1)
        return {
            'checked': False,
            'reason': 'v1 dataset has no obstacle_trajectory_velocity_world',
            'legacy_obstacle_velocity_semantics': 'PhysX actor root-state velocity',
            'legacy_physical_root_velocity_distribution': _distribution(physical),
            'legacy_physical_root_velocity_observed_max': _json_number(physical.max()),
        }
    trajectory_velocity = data['obstacle_trajectory_velocity_world'].float()
    speeds = torch.linalg.vector_norm(trajectory_velocity, dim=-1)
    observed = speeds.reshape(-1)
    final_range = None
    envelope = None
    if manifest is not None:
        if manifest.get('obstacle_speed_range'):
            final_range = [float(x) for x in manifest['obstacle_speed_range']]
        curriculum = manifest.get('obstacle_speed_curriculum', {})
        ranges = [final_range] if final_range else []
        if curriculum.get('enabled') and curriculum.get('speed_start'):
            ranges.append([float(x) for x in curriculum['speed_start']])
        if ranges:
            envelope = [min(x[0] for x in ranges), max(x[1] for x in ranges)]
    result = {
        'checked': True,
        'observed_speed_distribution': _distribution(observed),
        'observed_min': _json_number(observed.min()),
        'observed_max': _json_number(observed.max()),
        'configured_final_range': final_range,
        'configured_curriculum_envelope': envelope,
    }
    for name, bounds in [
        ('final_range', final_range), ('curriculum_envelope', envelope)
    ]:
        if bounds is None:
            continue
        below = observed < bounds[0] - 1e-5
        above = observed > bounds[1] + 1e-5
        result[name] = {
            'strictly_within': not bool((below | above).any()),
            'below_count': int(below.sum()),
            'above_count': int(above.sum()),
            'violation_count': int((below | above).sum()),
        }
    if 'obstacle_physics_velocity_world' in data:
        physical = torch.linalg.vector_norm(
            data['obstacle_physics_velocity_world'].float(), dim=-1
        ).reshape(-1)
        result['physical_root_velocity_distribution'] = _distribution(physical)
        result['physical_root_velocity_observed_max'] = _json_number(physical.max())
        result['physical_vs_trajectory_max_abs_difference'] = _json_number(
            (physical - observed).abs().max()
        )
    return result


def _speed_conditioned_stats(
    closing, radial, same_dynamic, source_ids,
    trajectory_velocity, speed_bins,
):
    if trajectory_velocity is None:
        return {'checked': False, 'reason': 'trajectory velocity is unavailable'}
    speed = torch.linalg.vector_norm(trajectory_velocity.float(), dim=-1)
    selected_ids = source_ids.clamp(min=0)
    selected_speed = torch.gather(speed, 1, selected_ids)
    mask = same_dynamic & (source_ids >= 0)
    edges = list(speed_bins)
    observed_max = float(selected_speed[mask].max()) if mask.any() else 0.0
    if observed_max > edges[-1] + 1e-6:
        edges.append(observed_max)
    bins = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (selected_speed >= low) & (
            selected_speed <= high if index == len(edges) - 2
            else selected_speed < high
        )
        valid = mask & in_bin
        raw_values = closing[valid]
        radial_values = radial[valid]
        error = raw_values - radial_values
        bins.append({
            'low': low,
            'high': high,
            'count_ray_values': int(raw_values.numel()),
            'mean_abs_closing_gt': (
                _json_number(raw_values.abs().mean())
                if raw_values.numel() else None
            ),
            'mean_abs_radial_gt': (
                _json_number(radial_values.abs().mean())
                if radial_values.numel() else None
            ),
            'mae': _json_number(error.abs().mean()) if error.numel() else None,
            'correlation': _corr_stats(raw_values, radial_values),
        })
    return {
        'checked': True,
        'conditioning': 'active obstacle trajectory speed for same-dynamic-source rays',
        'bin_edges': edges,
        'bins': bins,
    }


def _switching_stats(closing, source_switch_type):
    abs_closing = closing.abs()
    top_threshold = torch.quantile(abs_closing.reshape(-1), torch.tensor(0.99))
    top_one_percent = abs_closing >= top_threshold
    result = {
        'top_1_percent_abs_threshold': _json_number(top_threshold),
        'top_1_percent_total_count': int(top_one_percent.sum()),
        'types': {},
    }
    total = max(int(closing.numel()), 1)
    top_total = max(int(top_one_percent.sum()), 1)
    for type_id, name in SWITCH_NAMES.items():
        mask = source_switch_type == type_id
        values = closing[mask]
        result['types'][name] = {
            'type_id': type_id,
            'count': int(values.numel()),
            'ratio': float(values.numel()) / total,
            'distribution': _distribution(values),
            'absolute_distribution': _distribution(values.abs()),
            'top_1_percent_extreme_count': int((mask & top_one_percent).sum()),
            'top_1_percent_extreme_contribution': (
                float((mask & top_one_percent).sum()) / top_total
            ),
        }
    return result


def _smooth_min(first, second, beta):
    return -torch.logaddexp(-beta * first, -beta * second) / beta


def _smooth_min_analysis(data, closing, source_switch, same_dynamic, radial, horizon):
    result = {}
    for beta in [5.0, 10.0, 20.0]:
        current_soft = _smooth_min(
            data['static_rays'].float(), data['dynamic_rays'].float(), beta
        )
        future_soft = _smooth_min(
            data['static_rays'].float(), data['future_dynamic_rays'].float(), beta
        )
        soft_closing = (current_soft - future_soft) / horizon
        error = soft_closing - closing
        result[str(int(beta))] = {
            'beta': beta,
            'overall_abs_distribution': _distribution(soft_closing.abs()),
            'switching_abs_distribution': _distribution(
                soft_closing[source_switch].abs()
            ),
            'non_switching_abs_distribution': _distribution(
                soft_closing[~source_switch].abs()
            ),
            'difference_from_raw_distribution': _distribution(error),
            'mean_abs_difference_from_raw': _json_number(error.abs().mean()),
            'same_dynamic_vs_radial': (
                _agreement_stats(soft_closing[same_dynamic], radial[same_dynamic])
                if same_dynamic.any() else {'count': 0}
            ),
        }
    return result


def _lse_temporal_target(data, source_switch, horizon, d_safe, kappa):
    current_h = data['current_fused_rays'].float() - d_safe
    future_h = data['future_fused_rays'].float() - d_safe
    current_lse = -torch.logsumexp(-kappa * current_h, dim=1) / kappa
    future_lse = -torch.logsumexp(-kappa * future_h, dim=1) / kappa
    target = (future_lse - current_lse) / horizon
    return {
        'd_safe': float(d_safe),
        'kappa': float(kappa),
        'overall': _distribution(target),
        'absolute': _distribution(target.abs()),
        'switching': _distribution(target[source_switch.any(dim=1)]),
        'non_switching': _distribution(target[~source_switch.any(dim=1)]),
        'target': target,
    }


def _write_figures(
    output_dir, angles_deg, closing, per_angle, speed_stats,
    switching, source_switch, same_dynamic, radial, soft_data, lse_target,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    flat = closing.reshape(-1).numpy()
    display_limit = float(np.quantile(np.abs(flat), 0.995)) if flat.size else 1.0
    display_limit = max(display_limit, 1e-3)

    plt.figure(figsize=(8, 5))
    plt.hist(np.clip(flat, -display_limit, display_limit), bins=100)
    plt.xlabel('raw closing_rate_gt [m/s] (display clipped at P99.5)')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'gt_distribution.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, [item['positive_rate'] for item in per_angle], label='positive')
    plt.plot(angles_deg, [item['negative_rate'] for item in per_angle], label='negative')
    plt.plot(angles_deg, [item['p95_abs'] for item in per_angle], label='P95 |c|')
    plt.xlabel('ray angle [deg]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'angle_statistics.png'), dpi=140)
    plt.close()

    if speed_stats.get('bins'):
        labels = [
            '[{:.1f},{:.1f}]'.format(item['low'], item['high'])
            for item in speed_stats['bins']
        ]
        plt.figure(figsize=(8, 5))
        plt.bar(
            labels,
            [item['mean_abs_closing_gt'] or 0.0 for item in speed_stats['bins']],
        )
        plt.ylabel('mean |c| [m/s]')
        plt.xlabel('active obstacle speed bin [m/s]')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'speed_bins.png'), dpi=140)
        plt.close()

    switch_names = list(SWITCH_NAMES.values())
    switch_counts = [switching['types'][name]['count'] for name in switch_names]
    plt.figure(figsize=(8, 5))
    plt.bar(switch_names, switch_counts)
    plt.xticks(rotation=20, ha='right')
    plt.ylabel('ray count')
    plt.title('source transition types')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'switching_type_distribution.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, mask in [
        ('all', torch.ones_like(closing, dtype=torch.bool)),
        ('switching', source_switch),
        ('non-switching', ~source_switch),
    ]:
        values = closing[mask].abs().reshape(-1).numpy()
        if values.size:
            values = np.sort(values)
            plt.plot(values, np.linspace(0.0, 1.0, values.size, endpoint=False), label=name)
    plt.xscale('log')
    plt.xlabel('|closing rate| [m/s]')
    plt.ylabel('empirical CDF')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'boundary_switching.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(6, 6))
    if radial is not None and same_dynamic.any():
        x = radial[same_dynamic].reshape(-1).numpy()
        y = closing[same_dynamic].reshape(-1).numpy()
        if x.size > 100000:
            keep = np.linspace(0, x.size - 1, 100000).astype(np.int64)
            x, y = x[keep], y[keep]
        plt.hexbin(x, y, gridsize=60, mincnt=1, bins='log')
        limit = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-3)
        plt.plot([-limit, limit], [-limit, limit], 'r--', label='y=x')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'v2 radial GT unavailable', ha='center', va='center')
    plt.xlabel('active obstacle radial velocity GT [m/s]')
    plt.ylabel('raw closing_rate_gt [m/s]')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'continuous_closing_vs_radial_velocity.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(9, 5))
    raw_values = np.sort(closing.reshape(-1).abs().numpy())
    if raw_values.size:
        plt.plot(raw_values, np.linspace(0.0, 1.0, raw_values.size, endpoint=False), label='raw')
    for beta, values in soft_data.items():
        flat_values = np.sort(values.reshape(-1).abs().numpy())
        if flat_values.size:
            plt.plot(flat_values, np.linspace(0.0, 1.0, flat_values.size, endpoint=False), label='beta={}'.format(beta))
    plt.xscale('log')
    plt.xlabel('|closing rate| [m/s]')
    plt.ylabel('empirical CDF')
    plt.title('raw vs smooth-min candidate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'raw_vs_soft_closing_distribution.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(lse_target.numpy(), bins=100)
    plt.xlabel('L_f h^cf,GT [m/s]')
    plt.ylabel('count')
    plt.title('composite LSE-CBF temporal target')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lse_temporal_target_distribution.png'), dpi=140)
    plt.close()
    return figures_dir


def analyze(args):
    if args.epsilon < 0.0:
        raise ValueError('--epsilon must be non-negative')
    if args.d_safe < 0.0 or args.kappa <= 0.0:
        raise ValueError('--d_safe must be non-negative and --kappa positive')
    if len(args.speed_bins) < 2 or any(
        right <= left for left, right in zip(args.speed_bins, args.speed_bins[1:])
    ):
        raise ValueError('--speed_bins must be strictly increasing')
    dataset_dir, manifest, data, shard_paths, optional_presence = _load_dataset(
        args.dataset_dir
    )
    output_dir = args.output_dir or os.path.join(dataset_dir, 'analysis')
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    closing = data['closing_rate_gt'].float()
    horizon = data['gt_horizon'].float()
    if horizon.ndim == 0:
        horizon = horizon.expand(closing.shape[0])
    overall = _distribution(closing)
    abs_overall = _distribution(closing.abs())
    ratio = _ratio_stats(closing, args.epsilon)
    if manifest is not None and len(manifest.get('ray_angles_deg', [])) == closing.shape[1]:
        angles = np.asarray(manifest['ray_angles_deg'], dtype=np.float32)
    else:
        angles = np.linspace(-120.0, 120.0, closing.shape[1])
    per_angle = _per_angle_stats(closing, args.epsilon)

    current_source, future_source, source_switch, same_dynamic, switch_type, is_v2 = (
        _source_fields(data, args.epsilon)
    )
    radial = data.get('active_obstacle_radial_velocity_gt')
    if radial is not None:
        radial = radial.float()
    valid_radial = data.get('active_obstacle_radial_velocity_valid_mask')
    if valid_radial is not None:
        same_dynamic = same_dynamic & valid_radial.bool()

    trajectory_velocity = data.get('obstacle_trajectory_velocity_world')
    physical_validation = _trajectory_speed_validation(data, manifest)
    speed_stats = _speed_conditioned_stats(
        closing, radial if radial is not None else torch.zeros_like(closing),
        same_dynamic, current_source, trajectory_velocity, args.speed_bins,
    )
    switching = _switching_stats(closing, switch_type)

    if radial is not None and same_dynamic.any():
        radial_agreement = _agreement_stats(
            closing[same_dynamic], radial[same_dynamic]
        )
    else:
        radial_agreement = {
            'checked': False,
            'reason': 'v1 dataset has no active_obstacle_radial_velocity_gt',
        }

    smooth_data = {}
    for beta in [5.0, 10.0, 20.0]:
        current_soft = _smooth_min(
            data['static_rays'].float(), data['dynamic_rays'].float(), beta
        )
        future_soft = _smooth_min(
            data['static_rays'].float(), data['future_dynamic_rays'].float(), beta
        )
        smooth_data[str(int(beta))] = (
            current_soft - future_soft
        ) / horizon[:, None]
    smooth_summary = _smooth_min_analysis(
        data, closing, source_switch, same_dynamic,
        radial if radial is not None else torch.zeros_like(closing), horizon[:, None],
    )
    lse = _lse_temporal_target(
        data, source_switch, horizon, args.d_safe, args.kappa
    )
    lse_target = lse.pop('target')

    clipping = {
        str(limit): float((closing.abs() > limit).float().mean())
        for limit in [2.0, 5.0, 10.0, 20.0]
    }
    sampling_check = _check_sensor_frame_sampling(data, manifest)
    split_check = _check_splits(dataset_dir, data)
    summary = {
        'dataset_dir': dataset_dir,
        'dataset_version': manifest.get('dataset_version', 'v1') if manifest else 'v1',
        'num_shards_loaded': len(shard_paths),
        'num_samples': int(closing.shape[0]),
        'num_ray_values': int(closing.numel()),
        'v2_fields_present': sorted(optional_presence),
        'overall': overall,
        'absolute_value_distribution': abs_overall,
        'sign_ratios': ratio,
        'per_angle': {'angles_deg': angles.tolist(), 'statistics': per_angle},
        'trajectory_velocity_validation': physical_validation,
        'source_semantics': {
            'is_v2': is_v2,
            'encoding': {'static_environment': -1, 'dynamic_obstacles': '0..M-1'},
            'source_switch_ray_ratio': float(source_switch.float().mean()),
            'continuous_boundary_ray_ratio': float((~source_switch).float().mean()),
            'same_dynamic_source_ray_ratio': float(same_dynamic.float().mean()),
        },
        'speed_conditioned_same_dynamic_source': speed_stats,
        'continuous_same_dynamic_vs_radial': radial_agreement,
        'switching_types': switching,
        'raw_gt_clipping_report': clipping,
        'smooth_min_candidates': smooth_summary,
        'lse_temporal_target': {key: value for key, value in lse.items()},
        'sensor_frame_sampling_check': sampling_check,
        'split_check': split_check,
        'epsilon': float(args.epsilon),
        'manifest': manifest,
    }
    figures_dir = _write_figures(
        output_dir, angles, closing, per_angle, speed_stats,
        switching, source_switch, same_dynamic, radial, smooth_data,
        lse_target,
    )
    summary['figures_dir'] = figures_dir
    with open(os.path.join(output_dir, 'summary.json'), 'w') as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    with open(os.path.join(output_dir, 'summary.txt'), 'w') as file:
        file.write('Motion dataset research analysis\n')
        file.write('================================\n')
        file.write('dataset version: {}\n'.format(summary['dataset_version']))
        file.write('samples: {}\n'.format(summary['num_samples']))
        file.write('cGT mean/std: {:.6f} / {:.6f} m/s\n'.format(
            overall.get('mean', float('nan')), overall.get('std', float('nan')),
        ))
        file.write('cGT P99.9 |c|: {}\n'.format(abs_overall.get('p99_9')))
        file.write('source switch ray ratio: {:.6f}\n'.format(
            summary['source_semantics']['source_switch_ray_ratio']
        ))
        file.write('continuous same-dynamic agreement: {}\n'.format(radial_agreement))
        file.write('trajectory velocity validation: {}\n'.format(physical_validation))
        file.write('switching types: {}\n'.format(switching['types']))
        file.write('smooth-min candidates: {}\n'.format(smooth_summary))
        file.write('LSE temporal target: {}\n'.format(summary['lse_temporal_target']))
        file.write('sensor frame check: {}\n'.format(sampling_check))
        file.write('split check: {}\n'.format(split_check))
        file.write('raw clipping fractions: {}\n'.format(clipping))
    print('[analysis] summary={}'.format(os.path.join(output_dir, 'summary.json')))
    print('[analysis] figures={}'.format(figures_dir))


if __name__ == '__main__':
    analyze(_parse_args())
