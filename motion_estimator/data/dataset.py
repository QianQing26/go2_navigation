"""Shard-backed episode-split dataset for offline estimator training."""

import glob
import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, Sampler

from .normalization import NormalizationStats


CORE_KEYS = {
    'rays_hist', 'motion_ego_hist', 'closing_rate_gt',
    'source_switch_mask', 'current_fused_rays', 'future_fused_rays',
    'episode_id',
}


def _load_shard(path):
    """Load tensor-only dataset shards without the pickle warning."""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        return torch.load(path, map_location='cpu')


def _load_json(path):
    with open(path) as file:
        return json.load(file)


def load_dataset_metadata(dataset_dir):
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    manifest_path = os.path.join(dataset_dir, 'manifest.json')
    splits_path = os.path.join(dataset_dir, 'splits.json')
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError('Missing dataset manifest: {}'.format(manifest_path))
    if not os.path.isfile(splits_path):
        raise FileNotFoundError('Missing episode split file: {}'.format(splits_path))
    manifest = _load_json(manifest_path)
    splits = _load_json(splits_path)
    if not splits.get('episode_ids'):
        raise RuntimeError('splits.json has no episode_ids')
    owners = {}
    for split, episode_ids in splits['episode_ids'].items():
        for episode_id in episode_ids:
            episode_id = int(episode_id)
            if episode_id in owners:
                raise RuntimeError(
                    'Episode {} appears in multiple splits'.format(episode_id)
                )
            owners[episode_id] = split
    return dataset_dir, manifest, splits, owners


def _lse_drift(current, future, horizon, d_safe, kappa):
    current_h = current.float() - float(d_safe)
    future_h = future.float() - float(d_safe)
    current_lse = -torch.logsumexp(-float(kappa) * current_h, dim=-1) / float(kappa)
    future_lse = -torch.logsumexp(-float(kappa) * future_h, dim=-1) / float(kappa)
    return (future_lse - current_lse) / float(horizon)


class MotionDataset(Dataset):
    """Map-style view over one episode split.

    Dataset tensors are loaded one shard at a time and cached per worker.  The
    full dataset is never concatenated into RAM.
    """

    def __init__(
        self, dataset_dir, split, normalization=None, target_cfg=None,
        history_length=10, ray_max=None, max_cached_shards=1,
    ):
        self.dataset_dir, self.manifest, self.splits, self.episode_owners = (
            load_dataset_metadata(dataset_dir)
        )
        if split not in self.splits['episode_ids']:
            raise KeyError('Unknown split {}'.format(split))
        self.split = split
        self.allowed_episode_ids = set(
            int(value) for value in self.splits['episode_ids'][split]
        )
        self.history_length = int(history_length)
        self.target_cfg = target_cfg or {
            'd_safe': 0.20,
            'kappa': 10.0,
            'horizon': 0.10,
            'hit_epsilon': 1.0e-4,
        }
        self.normalization = normalization
        self.ray_max = float(ray_max or self._manifest_ray_max())
        self.max_cached_shards = int(max_cached_shards)
        self.shard_paths = self._manifest_shard_paths()
        self.index = self._build_index()
        self._cache = OrderedDict()

    def _manifest_ray_max(self):
        dynamic_range = self.manifest.get('dynamic_ray_range')
        if dynamic_range:
            return float(dynamic_range[1])
        raw_range = self.manifest.get('ray_range_semantics', {}).get(
            'raw_dataset_range_checked'
        )
        if raw_range:
            return float(raw_range[1])
        raise RuntimeError('Cannot determine ray_max from dataset manifest')

    def _manifest_shard_paths(self):
        entries = self.manifest.get('shards') or []
        paths = [
            os.path.join(self.dataset_dir, entry['path']) for entry in entries
        ]
        if not paths:
            paths = sorted(glob.glob(os.path.join(self.dataset_dir, 'shard_*.pt')))
        paths = [path for path in paths if os.path.isfile(path)]
        if not paths:
            raise FileNotFoundError('No dataset shards in {}'.format(self.dataset_dir))
        return paths

    def _build_index(self):
        index = []
        for shard_number, path in enumerate(self.shard_paths):
            shard = _load_shard(path)
            missing = CORE_KEYS.difference(shard.keys())
            if missing:
                raise RuntimeError('{} missing {}'.format(path, sorted(missing)))
            episode_ids = shard['episode_id'].reshape(-1).tolist()
            for local_index, episode_id in enumerate(episode_ids):
                if int(episode_id) in self.allowed_episode_ids:
                    index.append((shard_number, local_index))
        if not index:
            raise RuntimeError('Split {} contains no samples'.format(self.split))
        return index

    def __len__(self):
        return len(self.index)

    def _get_shard(self, shard_number):
        if shard_number in self._cache:
            shard = self._cache.pop(shard_number)
            self._cache[shard_number] = shard
            return shard
        shard = _load_shard(self.shard_paths[shard_number])
        self._cache[shard_number] = shard
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return shard

    def _prepare(self, shard, local_index):
        rays = shard['rays_hist'][local_index].float()
        ego = shard['motion_ego_hist'][local_index].float()
        closing = shard['closing_rate_gt'][local_index].float()
        current = shard['current_fused_rays'][local_index].float()
        future = shard['future_fused_rays'][local_index].float()
        source_switch = shard['source_switch_mask'][local_index].bool()
        if rays.shape[0] < self.history_length:
            raise RuntimeError('history_length exceeds stored history')
        rays = rays[-self.history_length:]
        ego = ego[-self.history_length:]
        if self.normalization is not None:
            ego_input = self.normalization.normalize_ego(ego)
            rays_input = rays / self.normalization.ray_max
            closing_target = closing / self.normalization.closing_scale
            drift_target = _lse_drift(
                current, future, self.normalization.horizon,
                self.normalization.d_safe, self.normalization.kappa,
            ) / self.normalization.drift_scale
            hit_epsilon = self.normalization.hit_epsilon
        else:
            ego_input = ego
            rays_input = rays / self.ray_max
            closing_target = closing
            drift_target = _lse_drift(
                current, future, self.target_cfg['horizon'],
                self.target_cfg['d_safe'], self.target_cfg['kappa'],
            )
            hit_epsilon = self.target_cfg.get('hit_epsilon', 1.0e-4)
        sample = {
            'rays_hist': rays_input,
            'ego_hist': ego_input,
            'closing_gt': closing,
            'closing_target': closing_target,
            'continuous_mask': ~source_switch,
            'source_switch_mask': source_switch,
            'current_fused_rays': current,
            'future_fused_rays': future,
            'lse_drift_gt': _lse_drift(
                current, future,
                self.normalization.horizon if self.normalization else self.target_cfg['horizon'],
                self.normalization.d_safe if self.normalization else self.target_cfg['d_safe'],
                self.normalization.kappa if self.normalization else self.target_cfg['kappa'],
            ),
            'drift_target': drift_target,
            'ray_hit_hist': (rays < self.ray_max - hit_epsilon),
            'episode_id': shard['episode_id'][local_index].long(),
            'episode_step': shard['episode_step'][local_index].long(),
        }
        if 'obstacle_trajectory_velocity_world' in shard:
            sample['trajectory_velocity_world'] = shard[
                'obstacle_trajectory_velocity_world'
            ][local_index].float()
        return sample

    def __getitem__(self, item):
        shard_number, local_index = self.index[item]
        return self._prepare(self._get_shard(shard_number), local_index)

    def iter_raw_batches(self, batch_size=4096):
        """Yield raw tensors for train-only statistics without per-frame I/O."""
        batch_size = int(batch_size)
        for path in self.shard_paths:
            shard = _load_shard(path)
            mask = torch.as_tensor([
                int(value) in self.allowed_episode_ids
                for value in shard['episode_id'].tolist()
            ], dtype=torch.bool)
            selected = {key: value[mask] for key, value in shard.items() if key in {
                'motion_ego_hist', 'closing_rate_gt', 'source_switch_mask',
                'current_fused_rays', 'future_fused_rays',
            }}
            selected['lse_drift_gt'] = _lse_drift(
                selected['current_fused_rays'], selected['future_fused_rays'],
                self.target_cfg['horizon'], self.target_cfg['d_safe'],
                self.target_cfg['kappa'],
            )
            for start in range(0, int(selected['closing_rate_gt'].shape[0]), batch_size):
                yield {
                    key: value[start:start + batch_size]
                    for key, value in selected.items()
                }


def build_datasets(dataset_dir, normalization, target_cfg, history_length, ray_max):
    return {
        split: MotionDataset(
            dataset_dir, split, normalization=normalization,
            target_cfg=target_cfg, history_length=history_length,
            ray_max=ray_max,
        )
        for split in ('train', 'val', 'test')
    }


class ShardBatchSampler(Sampler):
    """Shuffle batches while keeping every batch inside one shard.

    Random frame-level sampling causes each worker to repeatedly load many
    56 MB shards.  This sampler preserves stochastic training while making
    shard access sequential inside each batch.
    """

    def __init__(self, dataset, batch_size, shuffle, seed=1):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        groups = {}
        for global_index, (shard_number, _) in enumerate(dataset.index):
            groups.setdefault(shard_number, []).append(global_index)
        self.groups = list(groups.values())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        group_order = list(range(len(self.groups)))
        if self.shuffle:
            order = torch.randperm(len(group_order), generator=generator).tolist()
            group_order = [group_order[index] for index in order]
        for group_index in group_order:
            indices = self.groups[group_index]
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in order]
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start:start + self.batch_size])
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return iter(batches)

    def __len__(self):
        return sum(
            (len(group) + self.batch_size - 1) // self.batch_size
            for group in self.groups
        )
