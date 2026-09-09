"""Evaluate MotionEstimator checkpoints with safety-focused diagnostics."""

import argparse
import json
import os
import sys

# Keep direct execution (``python motion_estimator/evaluate.py``) self-contained.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import MotionDataset, NormalizationStats, load_dataset_metadata
from models import MotionEstimator
from utils.checkpoint import load_checkpoint
from utils.runtime import project_path
from utils.metrics import (
    error_quantiles, local_dynamic_stats, regression_stats,
    scalar_danger_breakdown, scalar_stats, speed_danger_breakdown,
)
from utils.visualization import (
    plot_error_distribution, plot_scatter, plot_test_example,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--num_examples', type=int, default=8)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    return parser.parse_args()


def _config_from_checkpoint(checkpoint, config_path):
    if config_path:
        with open(project_path(config_path)) as file:
            return yaml.safe_load(file)
    return checkpoint.get('config')


def _normalization_from_checkpoint(checkpoint):
    data = checkpoint['normalization']
    return NormalizationStats(
        ray_max=data['ray_max'], ego_mean=data['ego_mean'], ego_std=data['ego_std'],
        closing_scale=data['closing_scale'], drift_scale=data['drift_scale'],
        d_safe=data['d_safe'], kappa=data['kappa'], horizon=data['horizon'],
        hit_epsilon=data.get('hit_epsilon', 1.0e-4),
    )


def _model_from_checkpoint(checkpoint, device):
    config = checkpoint['model_config']
    return MotionEstimator(
        history_length=config['history_length'], num_rays=config['num_rays'],
        hidden_dim=config['hidden_dim'], temporal_pool=config['temporal_pool'],
        use_ego_motion=config['use_ego_motion'], ray_angles=config['ray_angles_deg'],
        ray_max=config['ray_max'], hit_epsilon=config['hit_epsilon'],
        d_safe=config['d_safe'], kappa=config['kappa'],
    ).to(device)


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _switch_breakdown(pred, target, source_switch, sign_epsilon, delta):
    sample_switch = source_switch.bool().any(dim=1)
    return {
        'switching': scalar_stats(pred[sample_switch], target[sample_switch], sign_epsilon, delta),
        'non_switching': scalar_stats(pred[~sample_switch], target[~sample_switch], sign_epsilon, delta),
    }


def collect_predictions(checkpoint_path, config_path=None, dataset_arg=None, split='test',
                        device=None, max_batches=None, num_workers=None, num_examples=0):
    """Run a checkpoint and return CPU tensors used by evaluation/calibration."""
    checkpoint_path = project_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = _config_from_checkpoint(checkpoint, config_path)
    dataset_arg = dataset_arg or config['dataset']['path']
    dataset_dir, manifest, splits, _ = load_dataset_metadata(project_path(dataset_arg))
    normalization = _normalization_from_checkpoint(checkpoint)
    model = _model_from_checkpoint(
        checkpoint,
        torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu')),
    )
    device = next(model.parameters()).device
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    dataset = MotionDataset(
        dataset_dir, split, normalization=normalization,
        target_cfg={
            'd_safe': normalization.d_safe, 'kappa': normalization.kappa,
            'horizon': normalization.horizon, 'hit_epsilon': normalization.hit_epsilon,
        }, history_length=checkpoint['model_config']['history_length'],
        ray_max=normalization.ray_max,
    )
    loader = DataLoader(
        dataset, batch_size=config['train']['batch_size'], shuffle=False,
        num_workers=int(num_workers if num_workers is not None else config['dataset']['num_workers']),
        pin_memory=bool(config['dataset'].get('pin_memory', True)),
    )
    closing_pred, closing_gt, closing_mask = [], [], []
    drift_pred, drift_gt, switches, speeds, current_rays, examples = [], [], [], [], [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            cpu_batch = batch
            device_batch = _to_device(batch, device)
            prediction = model(
                device_batch['rays_hist'], device_batch['ego_hist'],
                device_batch['ray_hit_hist'], device_batch['current_fused_rays'],
            )
            closing_pred_batch = prediction['closing'].cpu() * normalization.closing_scale
            drift_pred_batch = prediction['drift'].cpu() * normalization.drift_scale
            closing_pred.append(closing_pred_batch)
            closing_gt.append(cpu_batch['closing_gt'].float())
            closing_mask.append(cpu_batch['continuous_mask'].bool())
            drift_pred.append(drift_pred_batch)
            drift_gt.append(cpu_batch['lse_drift_gt'].float())
            switches.append(cpu_batch['source_switch_mask'].bool())
            current_rays.append(cpu_batch['current_fused_rays'].float())
            if 'trajectory_velocity_world' in cpu_batch:
                speeds.append(cpu_batch['trajectory_velocity_world'].float())
            if len(examples) < num_examples:
                for item in range(min(
                    int(cpu_batch['closing_gt'].shape[0]), num_examples - len(examples)
                )):
                    examples.append({
                        'closing_gt': closing_gt[-1][item].tolist(),
                        'closing_pred': closing_pred_batch[item].tolist(),
                        'current_rays': cpu_batch['current_fused_rays'][item].tolist(),
                        'drift_gt': float(drift_gt[-1][item]),
                        'drift_pred': float(drift_pred_batch[item]),
                    })
    if not drift_gt:
        raise RuntimeError('No batches processed')
    pred_c = torch.cat(closing_pred)
    gt_c = torch.cat(closing_gt)
    continuous = torch.cat(closing_mask)
    pred_h = torch.cat(drift_pred)
    gt_h = torch.cat(drift_gt)
    source_switch = torch.cat(switches)
    current_rays = torch.cat(current_rays)
    speed = (
        torch.linalg.vector_norm(torch.cat(speeds).float(), dim=-1).amax(dim=-1)
        if speeds else None
    )
    weights = torch.softmax(
        -normalization.kappa * (current_rays - normalization.d_safe), dim=-1
    )
    derived_h = -(weights * pred_c).sum(dim=-1)
    return {
        'checkpoint': checkpoint_path,
        'config': config,
        'dataset_dir': dataset_dir,
        'manifest': manifest,
        'splits': splits,
        'split': split,
        'normalization': normalization,
        'closing_pred': pred_c,
        'closing_gt': gt_c,
        'continuous_mask': continuous,
        'source_switch_mask': source_switch,
        'drift_pred': pred_h,
        'drift_gt': gt_h,
        'sample_switch_mask': source_switch.any(dim=1),
        'speed': speed,
        'derived_drift': derived_h,
        'examples': examples,
    }


def _build_report(data):
    config = data['config']
    normalization = data['normalization']
    pred_c, gt_c = data['closing_pred'], data['closing_gt']
    continuous = data['continuous_mask']
    source_switch = data['source_switch_mask']
    pred_h, gt_h = data['drift_pred'], data['drift_gt']
    sign_epsilon = float(config.get('evaluation', {}).get('sign_epsilon', 0.05))
    delta = float(config.get('evaluation', {}).get('underestimation_delta', 0.1))
    danger_cfg = config.get('loss', {}).get('drift_weight', {})
    danger_threshold = float(danger_cfg.get('danger_threshold', -sign_epsilon))
    severe_threshold = float(danger_cfg.get('severe_threshold', -0.5))
    local_masks = {
        'abs_gt_0.05': continuous & (gt_c.abs() > 0.05),
        'abs_gt_0.10': continuous & (gt_c.abs() > 0.10),
        'approaching_gt_0.05': continuous & (gt_c > 0.05),
        'approaching_gt_0.20': continuous & (gt_c > 0.20),
    }
    local_dynamic = {
        name: local_dynamic_stats(pred_c, gt_c, mask)
        for name, mask in local_masks.items()
    }
    local_switch = {
        'switching': local_dynamic_stats(pred_c, gt_c, source_switch),
        'non_switching': local_dynamic_stats(pred_c, gt_c, ~source_switch),
    }
    scalar = scalar_stats(pred_h, gt_h, sign_epsilon, delta)
    sample_switch = data['sample_switch_mask']
    error_masks = {
        'all': torch.ones_like(gt_h, dtype=torch.bool),
        'dangerous': gt_h < danger_threshold,
        'switching': sample_switch,
    }
    speed = data['speed']
    if speed is not None:
        error_masks['high_speed_danger'] = (speed >= 1.0) & (gt_h < danger_threshold)
    error_quantiles_report = {
        name: error_quantiles(pred_h[mask], gt_h[mask])
        for name, mask in error_masks.items()
    }
    result = {
        'checkpoint': data['checkpoint'],
        'dataset_dir': data['dataset_dir'],
        'split': data['split'],
        'num_samples': int(gt_h.numel()),
        'normalization': normalization.to_dict(),
        'local_closing_no_switch': regression_stats(
            pred_c[continuous], gt_c[continuous], sign_epsilon, delta
        ),
        'local_dynamic_breakdown': local_dynamic,
        'local_switch_breakdown': local_switch,
        'scalar_drift': scalar,
        'scalar_danger_breakdown': scalar_danger_breakdown(
            pred_h, gt_h, danger_threshold, severe_threshold, sign_epsilon, delta
        ),
        'scalar_switch_breakdown': _switch_breakdown(
            pred_h, gt_h, source_switch, sign_epsilon, delta
        ),
        'error_quantiles': error_quantiles_report,
        'derived_scalar_drift': scalar_stats(
            data['derived_drift'], gt_h, sign_epsilon, delta
        ),
        'zero_baseline': {
            'closing': regression_stats(
                torch.zeros_like(gt_c[continuous]), gt_c[continuous], sign_epsilon, delta
            ),
            'drift': scalar_stats(torch.zeros_like(gt_h), gt_h, sign_epsilon, delta),
        },
        'manifest_split_sample_counts': data['splits'].get('sample_counts'),
        'speed_metadata': {'available': speed is not None},
    }
    if speed is not None:
        result['speed_danger_breakdown'] = speed_danger_breakdown(
            pred_h, gt_h, speed, danger_threshold, severe_threshold, sign_epsilon, delta
        )
        result['speed_breakdown'] = {
            name: scalar_stats(pred_h[mask], gt_h[mask], sign_epsilon, delta)
            for name, mask in {
                'low_speed': speed < 0.5,
                'medium_speed': (speed >= 0.5) & (speed < 1.0),
                'high_speed': speed >= 1.0,
            }.items()
        }
        result['speed_metadata'].update({
            'min': float(speed.min()), 'max': float(speed.max()), 'mean': float(speed.mean()),
        })
    else:
        result['speed_danger_breakdown'] = {
            'checked': False,
            'reason': 'trajectory velocity unavailable',
        }
        result['speed_breakdown'] = result['speed_danger_breakdown']
    return result


def main(args):
    data = collect_predictions(
        args.checkpoint, config_path=args.config, dataset_arg=args.dataset,
        split=args.split, device=args.device, max_batches=args.max_batches,
        num_workers=args.num_workers, num_examples=args.num_examples,
    )
    result = _build_report(data)
    output_dir = project_path(
        args.output_dir or os.path.join(os.path.dirname(data['checkpoint']), 'evaluation')
    )
    os.makedirs(os.path.join(output_dir, 'figures', 'test_examples'), exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as file:
        json.dump(result, file, indent=2)
    with open(os.path.join(output_dir, 'drift_safety_breakdown.json'), 'w') as file:
        json.dump({
            'scalar_danger_breakdown': result['scalar_danger_breakdown'],
            'speed_danger_breakdown': result['speed_danger_breakdown'],
            'error_quantiles': result['error_quantiles'],
        }, file, indent=2)
    pred_c, gt_c = data['closing_pred'], data['closing_gt']
    mask = data['continuous_mask']
    pred_h, gt_h = data['drift_pred'], data['drift_gt']
    figures = os.path.join(output_dir, 'figures')
    plot_scatter(
        pred_c[mask], gt_c[mask], os.path.join(figures, 'closing_pred_vs_gt.png'),
        'GT c [m/s]', 'Pred c [m/s]', 'continuous closing'
    )
    plot_scatter(
        pred_h, gt_h, os.path.join(figures, 'drift_pred_vs_gt.png'),
        'GT d_h [m/s]', 'Pred d_h [m/s]', 'scalar drift'
    )
    plot_error_distribution(
        pred_c[mask], gt_c[mask], os.path.join(figures, 'closing_error_distribution.png'),
        'closing error', 'Pred - GT [m/s]'
    )
    plot_error_distribution(
        pred_h, gt_h, os.path.join(figures, 'drift_error_distribution.png'),
        'drift error', 'Pred - GT [m/s]'
    )
    angles = np.asarray(data['manifest'].get(
        'ray_angles_deg', np.linspace(-120, 120, gt_c.shape[1])
    ))
    for index, example in enumerate(data['examples']):
        plot_test_example(
            example, angles,
            os.path.join(figures, 'test_examples', 'example_{:03d}.png'.format(index)),
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main(_parse_args())
