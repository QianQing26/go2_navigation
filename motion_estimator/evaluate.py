"""Evaluate a trained MotionEstimator on the episode-level test split."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import MotionDataset, NormalizationStats, load_dataset_metadata
from models import MotionEstimator
from utils.checkpoint import load_checkpoint
from utils.metrics import regression_stats, scalar_stats, zero_baseline
from utils.visualization import (
    plot_error_distribution, plot_scatter, plot_test_example,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _path(value):
    value = os.path.expanduser(value)
    return value if os.path.isabs(value) else os.path.join(ROOT, value)


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
        with open(_path(config_path)) as file:
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


def _speed_breakdown(pred, target, trajectory_velocity, bins, sign_epsilon, delta):
    if trajectory_velocity is None:
        return {'checked': False, 'reason': 'trajectory velocity unavailable'}
    speed = torch.linalg.vector_norm(trajectory_velocity.float(), dim=-1).mean(dim=1)
    result = []
    for index, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (speed >= low) & (speed <= high if index == len(bins) - 2 else speed < high)
        result.append({
            'name': ['low', 'medium', 'high'][index] if index < 3 else 'bin_{}'.format(index),
            'low': low,
            'high': high,
            'sample_count': int(mask.sum()),
            'drift': scalar_stats(pred[mask], target[mask], sign_epsilon, delta),
        })
    return {'checked': True, 'bins': result}


def _switch_breakdown(pred, target, source_switch, sign_epsilon, delta):
    sample_switch = source_switch.bool().any(dim=1)
    return {
        'switching': scalar_stats(pred[sample_switch], target[sample_switch], sign_epsilon, delta),
        'non_switching': scalar_stats(pred[~sample_switch], target[~sample_switch], sign_epsilon, delta),
    }


def main(args):
    checkpoint_path = _path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = _config_from_checkpoint(checkpoint, args.config)
    dataset_arg = args.dataset or config['dataset']['path']
    dataset_dir, manifest, splits, _ = load_dataset_metadata(_path(dataset_arg))
    normalization = _normalization_from_checkpoint(checkpoint)
    model = _model_from_checkpoint(checkpoint, torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu')))
    device = next(model.parameters()).device
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    dataset = MotionDataset(
        dataset_dir, args.split, normalization=normalization,
        target_cfg={
            'd_safe': normalization.d_safe, 'kappa': normalization.kappa,
            'horizon': normalization.horizon, 'hit_epsilon': normalization.hit_epsilon,
        }, history_length=checkpoint['model_config']['history_length'],
        ray_max=normalization.ray_max,
    )
    loader = DataLoader(
        dataset, batch_size=config['train']['batch_size'], shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else config['dataset']['num_workers']),
        pin_memory=bool(config['dataset'].get('pin_memory', True)),
    )
    output_dir = _path(args.output_dir or os.path.join(os.path.dirname(checkpoint_path), 'evaluation'))
    os.makedirs(os.path.join(output_dir, 'figures', 'test_examples'), exist_ok=True)
    closing_pred, closing_gt, closing_mask = [], [], []
    drift_pred, drift_gt, switches, speeds, current_rays, examples = [], [], [], [], [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            cpu_batch = batch
            device_batch = _to_device(batch, device)
            prediction = model(
                device_batch['rays_hist'], device_batch['ego_hist'],
                device_batch['ray_hit_hist'], device_batch['current_fused_rays'],
            )
            # Keep all report tensors on CPU.
            closing_pred.append(prediction['closing'].cpu() * normalization.closing_scale)
            closing_gt.append(cpu_batch['closing_gt'].float())
            closing_mask.append(cpu_batch['continuous_mask'].bool())
            drift_pred.append(prediction['drift'].cpu() * normalization.drift_scale)
            drift_gt.append(cpu_batch['lse_drift_gt'].float())
            switches.append(cpu_batch['source_switch_mask'].bool())
            current_rays.append(cpu_batch['current_fused_rays'].float())
            if 'trajectory_velocity_world' in cpu_batch:
                speeds.append(cpu_batch['trajectory_velocity_world'].float())
            if len(examples) < args.num_examples:
                for item in range(min(int(cpu_batch['closing_gt'].shape[0]), args.num_examples - len(examples))):
                    examples.append({
                        'closing_gt': closing_gt[-1][item].tolist(),
                        'closing_pred': closing_pred[-1][item].tolist(),
                        'current_rays': cpu_batch['current_fused_rays'][item].tolist(),
                        'drift_gt': float(drift_gt[-1][item]),
                        'drift_pred': float(drift_pred[-1][item]),
                    })
    pred_c, gt_c, mask = torch.cat(closing_pred), torch.cat(closing_gt), torch.cat(closing_mask)
    pred_h, gt_h = torch.cat(drift_pred), torch.cat(drift_gt)
    switch = torch.cat(switches)
    sign_epsilon = float(config['evaluation']['sign_epsilon'])
    delta = float(config['evaluation']['underestimation_delta'])
    local = regression_stats(pred_c[mask], gt_c[mask], sign_epsilon, delta)
    scalar = scalar_stats(pred_h, gt_h, sign_epsilon, delta)
    current_rays = torch.cat(current_rays)
    weights = torch.softmax(-normalization.kappa * (current_rays - normalization.d_safe), dim=-1)
    derived_h = -(weights * pred_c).sum(dim=-1)
    result = {
        'checkpoint': checkpoint_path,
        'dataset_dir': dataset_dir,
        'split': args.split,
        'num_samples': int(gt_h.numel()),
        'normalization': normalization.to_dict(),
        'local_closing_no_switch': local,
        'scalar_drift': scalar,
        'scalar_switch_breakdown': _switch_breakdown(pred_h, gt_h, switch, sign_epsilon, delta),
        'derived_scalar_drift': scalar_stats(derived_h, gt_h, sign_epsilon, delta),
        'zero_baseline': {
            'closing': regression_stats(torch.zeros_like(gt_c[mask]), gt_c[mask], sign_epsilon, delta),
            'drift': scalar_stats(torch.zeros_like(gt_h), gt_h, sign_epsilon, delta),
        },
        'manifest_split_sample_counts': splits.get('sample_counts'),
    }
    if speeds:
        result['speed_breakdown'] = _speed_breakdown(
            pred_h, gt_h, torch.cat(speeds), [0.2, 0.5, 1.0, 1.5], sign_epsilon, delta
        )
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as file:
        json.dump(result, file, indent=2)
    figures = os.path.join(output_dir, 'figures')
    plot_scatter(pred_c[mask], gt_c[mask], os.path.join(figures, 'closing_pred_vs_gt.png'), 'GT c [m/s]', 'Pred c [m/s]', 'continuous closing')
    plot_scatter(pred_h, gt_h, os.path.join(figures, 'drift_pred_vs_gt.png'), 'GT d_h [m/s]', 'Pred d_h [m/s]', 'scalar drift')
    plot_error_distribution(pred_c[mask], gt_c[mask], os.path.join(figures, 'closing_error_distribution.png'), 'closing error', 'Pred - GT [m/s]')
    plot_error_distribution(pred_h, gt_h, os.path.join(figures, 'drift_error_distribution.png'), 'drift error', 'Pred - GT [m/s]')
    angles = np.asarray(manifest.get('ray_angles_deg', np.linspace(-120, 120, gt_c.shape[1])))
    for index, example in enumerate(examples):
        plot_test_example(example, angles, os.path.join(figures, 'test_examples', 'example_{:03d}.png'.format(index)))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(__file__))
    main(_parse_args())
