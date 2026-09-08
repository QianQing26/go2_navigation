"""Analyze raw motion-estimation dataset shards without training an estimator."""

import argparse
import json
import os
import glob

import numpy as np
import torch


REQUIRED_KEYS = {
    'rays_hist', 'motion_ego_hist', 'closing_rate_gt',
    'current_fused_rays', 'static_rays', 'dynamic_rays',
    'future_fused_rays', 'future_dynamic_rays',
    'dynamic_ray_hit_mask', 'obstacle_velocity_world',
    'obstacle_position_world', 'robot_xy_world', 'robot_yaw', 'gt_horizon',
    'env_id', 'episode_id', 'episode_step', 'global_policy_step',
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument(
        '--speed_bins', type=float, nargs='+',
        default=[0.0, 0.5, 1.0, 1.5],
        help='Bin edges; the final bin includes its right edge.',
    )
    return parser.parse_args()


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _distribution(values, percentiles=None):
    values = values.reshape(-1).float()
    if values.numel() == 0:
        return {'count': 0}
    if percentiles is None:
        percentiles = [90.0, 95.0, 99.0, 99.5, 99.9]
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
        percentile_name = (
            str(int(percentile))
            if float(percentile).is_integer()
            else str(percentile).replace('.', '_')
        )
        result['p{}'.format(percentile_name)] = _json_number(quantile)
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


def _conditional_positive(values, epsilon=0.0):
    positive = values[values > epsilon]
    if positive.numel() == 0:
        return {
            'positive_count': 0,
            'positive_ratio': 0.0,
            'mean_positive': None,
            'p95_positive': None,
        }
    return {
        'positive_count': int(positive.numel()),
        'positive_ratio': float(positive.numel()) / max(int(values.numel()), 1),
        'mean_positive': _json_number(positive.mean()),
        'p95_positive': _json_number(torch.quantile(positive, torch.tensor(0.95))),
    }


def _load_dataset(dataset_dir):
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    manifest_path = os.path.join(dataset_dir, 'manifest.json')
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path) as file:
            manifest = json.load(file)
    if manifest is not None:
        paths = [
            os.path.join(dataset_dir, shard['path'])
            for shard in manifest.get('shards', [])
        ]
    else:
        paths = sorted(glob.glob(os.path.join(dataset_dir, 'shard_*.pt')))
    if not paths:
        raise FileNotFoundError('No dataset shards found in {}'.format(dataset_dir))

    chunks = {key: [] for key in REQUIRED_KEYS}
    total = 0
    reference_shapes = None
    for path in paths:
        shard = torch.load(path, map_location='cpu')
        missing = REQUIRED_KEYS.difference(shard.keys())
        if missing:
            raise RuntimeError('{} is missing keys {}'.format(path, sorted(missing)))
        num_samples = int(shard['closing_rate_gt'].shape[0])
        if num_samples < 1:
            raise RuntimeError('Empty shard: {}'.format(path))
        for key in REQUIRED_KEYS:
            if int(shard[key].shape[0]) != num_samples:
                raise RuntimeError('{} has inconsistent leading shapes'.format(path))
            chunks[key].append(shard[key])
        sample_shapes = {
            key: list(shard[key].shape[1:]) for key in REQUIRED_KEYS
        }
        if reference_shapes is None:
            reference_shapes = sample_shapes
        elif sample_shapes != reference_shapes:
            raise RuntimeError('{} has inconsistent tensor shapes'.format(path))
        total += num_samples
    data = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
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
        'future_fused_rays', 'future_dynamic_rays', 'dynamic_ray_hit_mask',
    ]:
        if tuple(data[key].shape[1:]) != (41,):
            raise RuntimeError('{} does not have [N, 41] layout'.format(key))
    return dataset_dir, manifest, data, paths


def _check_sensor_frame_sampling(data, manifest):
    if manifest is None:
        return {'checked': False, 'reason': 'manifest.json not found'}
    interval = int(manifest.get('exteroception_update_interval', 0))
    if interval < 1:
        return {'checked': False, 'reason': 'update interval not recorded'}
    env_ids = data['env_id'].tolist()
    episode_ids = data['episode_id'].tolist()
    steps = data['global_policy_step'].tolist()
    grouped = {}
    for env_id, episode_id, step in zip(env_ids, episode_ids, steps):
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
            if episode_id in owners:
                raise RuntimeError('episode appears in multiple dataset splits')
            owners[episode_id] = split
    present = set(int(value) for value in torch.unique(data['episode_id']))
    if present != set(int(value) for value in owners):
        raise RuntimeError('splits.json does not cover exactly the dataset episodes')
    return {
        'checked': True,
        'episode_counts': splits.get('episode_counts', {}),
        'sample_counts': splits.get('sample_counts', {}),
    }


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
            'p95_abs': _json_number(torch.quantile(values.abs(), torch.tensor(0.95))),
        })
    return stats


def _speed_stats(closing, obstacle_velocity, speed_bins):
    scene_speed = torch.linalg.vector_norm(obstacle_velocity, dim=-1).amax(dim=-1)
    speed_stats = []
    for index in range(len(speed_bins) - 1):
        low, high = speed_bins[index:index + 2]
        if index == len(speed_bins) - 2:
            mask = (scene_speed >= low) & (scene_speed <= high)
        else:
            mask = (scene_speed >= low) & (scene_speed < high)
        values = closing[mask].reshape(-1)
        positive = values[values > 0.0]
        speed_stats.append({
            'low': low,
            'high': high,
            'count_samples': int(mask.sum()),
            'count_ray_values': int(values.numel()),
            'positive_ratio': float((values > 0.0).float().mean()) if values.numel() else 0.0,
            'mean_positive': _json_number(positive.mean()) if positive.numel() else None,
            'p95_positive': (
                _json_number(torch.quantile(positive, torch.tensor(0.95)))
                if positive.numel() else None
            ),
        })
    return scene_speed, speed_stats


def _switching_stats(data, epsilon):
    current_dynamic_source = data['dynamic_rays'] < data['static_rays'] - epsilon
    future_dynamic_source = data['future_dynamic_rays'] < data['static_rays'] - epsilon
    switching = current_dynamic_source ^ future_dynamic_source
    closing = data['closing_rate_gt']
    values_switching = closing[switching]
    values_static = closing[~switching]
    abs_closing = closing.abs()
    top_threshold = torch.quantile(abs_closing.reshape(-1), torch.tensor(0.99))
    top_one_percent = abs_closing >= top_threshold
    extreme_switching = switching & top_one_percent
    return {
        'source_epsilon': float(epsilon),
        'switching_ray_ratio': float(switching.float().mean()),
        'samples_with_switching_ratio': float(switching.any(dim=1).float().mean()),
        'switching_distribution': _distribution(values_switching),
        'non_switching_distribution': _distribution(values_static),
        'switching_abs_distribution': _distribution(values_switching.abs()),
        'non_switching_abs_distribution': _distribution(values_static.abs()),
        'top_1_percent_abs_threshold': _json_number(top_threshold),
        'top_1_percent_extreme_count': int(top_one_percent.sum()),
        'top_1_percent_extreme_switching_ratio': (
            float(extreme_switching.sum()) / max(int(top_one_percent.sum()), 1)
        ),
        'current_dynamic_source_ratio': float(current_dynamic_source.float().mean()),
        'future_dynamic_source_ratio': float(future_dynamic_source.float().mean()),
    }


def _write_figures(output_dir, angles_deg, closing, per_angle, speed_stats, switching):
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

    labels = ['[{:.1f},{:.1f}]'.format(item['low'], item['high']) for item in speed_stats]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, [item['positive_ratio'] for item in speed_stats])
    plt.ylabel('P(c > 0)')
    plt.xlabel('max obstacle speed bin [m/s]')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'speed_bins.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    labels = ['non-switching', 'switching']
    p95 = [
        switching['non_switching_abs_distribution'].get('p95', 0.0),
        switching['switching_abs_distribution'].get('p95', 0.0),
    ]
    plt.bar(labels, p95)
    plt.ylabel('P95 |c| [m/s]')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'boundary_switching.png'), dpi=140)
    plt.close()
    return figures_dir


def analyze(args):
    if args.epsilon < 0.0:
        raise ValueError('--epsilon must be non-negative')
    if len(args.speed_bins) < 2 or any(
        right <= left for left, right in zip(args.speed_bins, args.speed_bins[1:])
    ):
        raise ValueError('--speed_bins must be strictly increasing')
    dataset_dir, manifest, data, shard_paths = _load_dataset(args.dataset_dir)
    output_dir = args.output_dir or os.path.join(dataset_dir, 'analysis')
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    closing = data['closing_rate_gt'].float()
    overall = _distribution(closing)
    abs_overall = _distribution(closing.abs())
    ratio = _ratio_stats(closing, args.epsilon)
    if manifest is not None and len(manifest.get('ray_angles_deg', [])) == closing.shape[1]:
        angles = np.asarray(manifest['ray_angles_deg'], dtype=np.float32)
    else:
        angles = np.linspace(-120.0, 120.0, closing.shape[1])
    per_angle = _per_angle_stats(closing, args.epsilon)
    speed_bins = list(args.speed_bins)
    observed_speed = torch.linalg.vector_norm(
        data['obstacle_velocity_world'].float(), dim=-1
    ).amax(dim=-1)
    observed_max_speed = float(observed_speed.max())
    if observed_max_speed > speed_bins[-1] + 1e-6:
        speed_bins.append(observed_max_speed)
    scene_speed, speed_stats = _speed_stats(
        closing, data['obstacle_velocity_world'].float(), speed_bins
    )
    switching = _switching_stats(data, args.epsilon)
    clipping = {
        str(limit): float((closing.abs() > limit).float().mean())
        for limit in [2.0, 5.0, 10.0, 20.0]
    }
    sampling_check = _check_sensor_frame_sampling(data, manifest)
    split_check = _check_splits(dataset_dir, data)
    summary = {
        'dataset_dir': dataset_dir,
        'num_shards_loaded': len(shard_paths),
        'num_samples': int(closing.shape[0]),
        'num_ray_values': int(closing.numel()),
        'overall': overall,
        'absolute_value_distribution': abs_overall,
        'sign_ratios': ratio,
        'per_angle': {
            'angles_deg': angles.tolist(),
            'statistics': per_angle,
        },
        'speed_conditioned': {
            'bin_edges': speed_bins,
            'scene_speed_min': _json_number(scene_speed.min()),
            'scene_speed_max': _json_number(scene_speed.max()),
            'bins': speed_stats,
        },
        'boundary_switching': switching,
        'raw_gt_clipping_report': clipping,
        'sensor_frame_sampling_check': sampling_check,
        'split_check': split_check,
        'epsilon': float(args.epsilon),
        'manifest': manifest,
    }
    figures_dir = _write_figures(
        output_dir, angles, closing, per_angle, speed_stats, switching
    )
    summary['figures_dir'] = figures_dir
    with open(os.path.join(output_dir, 'summary.json'), 'w') as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    with open(os.path.join(output_dir, 'summary.txt'), 'w') as file:
        file.write('Motion dataset analysis\n')
        file.write('=======================\n')
        file.write('samples: {}\n'.format(summary['num_samples']))
        file.write('cGT mean/std: {:.6f} / {:.6f} m/s\n'.format(
            overall['mean'], overall['std'],
        ))
        file.write('cGT min/max: {:.6f} / {:.6f} m/s\n'.format(
            overall['min'], overall['max'],
        ))
        file.write('P(c > eps): {:.4f}\n'.format(ratio['positive_ratio']))
        file.write('P(c < -eps): {:.4f}\n'.format(ratio['negative_ratio']))
        file.write('P(|c| <= eps): {:.4f}\n'.format(ratio['near_zero_ratio']))
        file.write('switching ray ratio: {:.4f}\n'.format(
            switching['switching_ray_ratio'],
        ))
        file.write('top-1%% extreme switching ratio: {:.4f}\n'.format(
            switching['top_1_percent_extreme_switching_ratio'],
        ))
        file.write('sensor frame check: {}\n'.format(sampling_check))
        file.write('split check: {}\n'.format(split_check))
        file.write('raw clipping fractions: {}\n'.format(clipping))
    print('[analysis] summary={}'.format(os.path.join(output_dir, 'summary.json')))
    print('[analysis] figures={}'.format(figures_dir))


if __name__ == '__main__':
    analyze(_parse_args())
