"""Inspect final MotionEstimator dataset schema and episode splits."""

import argparse
import json
import os
import sys

from data import MotionDataset, load_dataset_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='datasets/motion_dataset_final')
    args = parser.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset = args.dataset if os.path.isabs(args.dataset) else os.path.join(root, args.dataset)
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
    sys.path.insert(0, os.path.dirname(__file__))
    main()
