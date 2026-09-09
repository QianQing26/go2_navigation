"""Validation-only conservative scalar drift calibration analysis."""

import argparse
import json
import os
import sys

# Keep direct execution (``python motion_estimator/calibrate.py``) self-contained.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from evaluate import collect_predictions
from utils.runtime import project_path
from utils.metrics import scalar_stats


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    return parser.parse_args()


def _plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    delta = [row['delta'] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(delta, [row['false_safe_rate'] for row in rows], marker='o', label='false-safe')
    plt.plot(delta, [row['optimistic_danger_rate'] for row in rows], marker='o', label='optimistic danger')
    plt.plot(delta, [row['mae'] for row in rows], marker='o', label='MAE')
    plt.xlabel('conservative bias delta [m/s]')
    plt.ylabel('rate / error [m/s]')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main(args):
    data = collect_predictions(
        args.checkpoint, config_path=args.config, dataset_arg=args.dataset,
        split='val', device=args.device, max_batches=args.max_batches,
        num_workers=args.num_workers,
    )
    pred, target = data['drift_pred'], data['drift_gt']
    config = data['config']
    sign_epsilon = float(config.get('evaluation', {}).get('sign_epsilon', 0.05))
    metric_delta = float(config.get('evaluation', {}).get('underestimation_delta', 0.1))
    static = target.abs() <= sign_epsilon
    rows = []
    for bias in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15):
        calibrated = pred - bias
        stats = scalar_stats(calibrated, target, sign_epsilon, metric_delta)
        rows.append({
            'delta': bias,
            'mae': stats.get('mae'),
            'false_safe_rate': stats.get('false_safe_rate'),
            'optimistic_danger_rate': stats.get('optimistic_danger_rate'),
            'static_bias': float(calibrated[static].mean()) if static.any() else None,
            'percentage_predicted_dangerous': float((calibrated < -sign_epsilon).float().mean()),
            'optimistic_danger_rate_any': stats.get('optimistic_danger_rate_any'),
        })
    output_dir = project_path(
        args.output_dir or os.path.join(os.path.dirname(data['checkpoint']), 'calibration')
    )
    os.makedirs(output_dir, exist_ok=True)
    result = {
        'checkpoint': data['checkpoint'],
        'split': 'val',
        'num_samples': int(target.numel()),
        'rows': rows,
        'definition': 'calibrated_drift = predicted_drift - delta; forward pass unchanged',
    }
    with open(os.path.join(output_dir, 'calibration_curve.json'), 'w') as file:
        json.dump(result, file, indent=2)
    _plot(rows, os.path.join(output_dir, 'calibration_curve.png'))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main(_parse_args())
