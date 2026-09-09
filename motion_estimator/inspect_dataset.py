"""Inspect final MotionEstimator dataset schema and episode splits."""

import argparse
import json
import os
import sys

# Keep direct execution (``python motion_estimator/inspect_dataset.py``) self-contained.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import MotionDataset, load_dataset_metadata
from utils.runtime import project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='datasets/motion_dataset_final')
    args = parser.parse_args()
    dataset = project_path(args.dataset)
    dataset_dir, manifest, splits, owners = load_dataset_metadata(dataset)
    print(json.dumps({
        'dataset_dir': dataset_dir,
        'dataset_version': manifest.get('dataset_version'),
        'total_samples_manifest': manifest.get('total_samples'),
        'num_shards': manifest.get('num_shards'),
        'gt_horizon': manifest.get('gt_horizon'),
        'ray_max': (manifest.get('dynamic_ray_range') or [None, None])[1],
        'split_sample_counts_manifest': splits.get('sample_counts'),
        'split_episode_counts': splits.get('episode_counts'),
        'num_episode_ids': len(owners),
        'episode_overlap': False,
    }, indent=2))
    for split in ('train', 'val', 'test'):
        view = MotionDataset(dataset, split, normalization=None, target_cfg={
            'd_safe': 0.20, 'kappa': 10.0, 'horizon': manifest.get('gt_horizon', 0.1),
        })
        print('{} samples={}'.format(split, len(view)))


if __name__ == '__main__':
    main()
