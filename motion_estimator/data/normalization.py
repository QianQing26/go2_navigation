"""Train-only normalization statistics for MotionEstimator."""

import json
import os

import torch


class NormalizationStats:
    """Serializable normalization constants computed only from train data."""

    def __init__(
        self, ray_max, ego_mean, ego_std, closing_scale, drift_scale,
        d_safe, kappa, horizon, hit_epsilon=1.0e-4,
    ):
        self.ray_max = float(ray_max)
        self.ego_mean = torch.as_tensor(ego_mean, dtype=torch.float32).reshape(3)
        self.ego_std = torch.as_tensor(ego_std, dtype=torch.float32).reshape(3)
        self.closing_scale = float(max(float(closing_scale), 1.0e-6))
        self.drift_scale = float(max(float(drift_scale), 1.0e-6))
        self.d_safe = float(d_safe)
        self.kappa = float(kappa)
        self.horizon = float(horizon)
        self.hit_epsilon = float(hit_epsilon)

    def to_dict(self):
        return {
            'ray_max': self.ray_max,
            'ego_mean': self.ego_mean.tolist(),
            'ego_std': self.ego_std.tolist(),
            'closing_scale': self.closing_scale,
            'drift_scale': self.drift_scale,
            'd_safe': self.d_safe,
            'kappa': self.kappa,
            'horizon': self.horizon,
            'hit_epsilon': self.hit_epsilon,
            'source': 'train split only; P95 absolute target scales',
        }

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as file:
            data = json.load(file)
        return cls(
            ray_max=data['ray_max'],
            ego_mean=data['ego_mean'],
            ego_std=data['ego_std'],
            closing_scale=data['closing_scale'],
            drift_scale=data['drift_scale'],
            d_safe=data['d_safe'],
            kappa=data['kappa'],
            horizon=data['horizon'],
            hit_epsilon=data.get('hit_epsilon', 1.0e-4),
        )

    @classmethod
    def from_dataset(cls, dataset, target_cfg, ray_max, chunk_size=4096):
        """Compute statistics by streaming train samples shard by shard."""
        sum_ego = torch.zeros(3, dtype=torch.float64)
        sum_ego_sq = torch.zeros(3, dtype=torch.float64)
        ego_count = 0
        closing_abs = []
        drift_abs = []
        for batch in dataset.iter_raw_batches(chunk_size):
            ego = batch['motion_ego_hist'].double().reshape(-1, 3)
            sum_ego += ego.sum(dim=0)
            sum_ego_sq += ego.square().sum(dim=0)
            ego_count += int(ego.shape[0])
            continuous = ~batch['source_switch_mask'].bool()
            closing_abs.append(batch['closing_rate_gt'][continuous].abs().float())
            drift_abs.append(batch['lse_drift_gt'].abs().float().reshape(-1))
        if ego_count == 0:
            raise RuntimeError('Cannot compute normalization from an empty train split')
        mean = sum_ego / ego_count
        variance = (sum_ego_sq / ego_count - mean.square()).clamp_min(1.0e-8)
        closing_values = torch.cat(closing_abs) if closing_abs else torch.zeros(1)
        drift_values = torch.cat(drift_abs) if drift_abs else torch.zeros(1)
        return cls(
            ray_max=ray_max,
            ego_mean=mean.float(),
            ego_std=torch.sqrt(variance).float(),
            closing_scale=torch.quantile(closing_values, 0.95),
            drift_scale=torch.quantile(drift_values, 0.95),
            d_safe=target_cfg['d_safe'],
            kappa=target_cfg['kappa'],
            horizon=target_cfg['horizon'],
            hit_epsilon=target_cfg.get('hit_epsilon', 1.0e-4),
        )

    def normalize_ego(self, ego):
        shape = [1] * (ego.ndim - 1) + [3]
        return (ego - self.ego_mean.to(ego.device).view(*shape)) / (
            self.ego_std.to(ego.device).view(*shape) + 1.0e-6
        )
