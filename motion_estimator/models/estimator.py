"""MotionEstimator model, independent of Isaac Gym and navigation policy code."""

import torch
import torch.nn as nn

from .encoder import SpatioTemporalEncoder
from .heads import ClosingHead, DriftHead


class MotionEstimator(nn.Module):
    def __init__(
        self, history_length=10, num_rays=41, hidden_dim=64,
        temporal_pool='attention', use_ego_motion=True,
        angular_dilations=(1, 2, 4), ray_angles=None,
        ray_max=3.0, hit_epsilon=1.0e-4, d_safe=0.20, kappa=10.0,
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.num_rays = int(num_rays)
        self.use_ego_motion = bool(use_ego_motion)
        self.ray_max = float(ray_max)
        self.hit_epsilon = float(hit_epsilon)
        self.d_safe = float(d_safe)
        self.kappa = float(kappa)
        if ray_angles is None:
            ray_angles = torch.linspace(-120.0, 120.0, self.num_rays)
        self.register_buffer('ray_angles', torch.as_tensor(ray_angles).float())
        input_channels = 7 if self.use_ego_motion else 4
        self.encoder = SpatioTemporalEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            temporal_pool=temporal_pool,
            dilations=angular_dilations,
        )
        self.closing_head = ClosingHead(hidden_dim)
        self.drift_head = DriftHead(hidden_dim)

    def build_features(self, rays_hist, ego_hist, ray_hit_hist):
        # rays_hist/ego_hist are already normalized by the dataset.
        batch, history, rays = rays_hist.shape
        angles = torch.deg2rad(self.ray_angles).to(rays_hist.device)
        sin_angle = torch.sin(angles).view(1, 1, rays).expand(batch, history, -1)
        cos_angle = torch.cos(angles).view(1, 1, rays).expand(batch, history, -1)
        if self.use_ego_motion:
            ego = ego_hist.unsqueeze(-1).expand(-1, -1, -1, rays)
            channels = [
                rays_hist.unsqueeze(1),
                ego[:, :, 0:1].permute(0, 2, 1, 3),
                ego[:, :, 1:2].permute(0, 2, 1, 3),
                ego[:, :, 2:3].permute(0, 2, 1, 3),
                sin_angle.unsqueeze(1),
                cos_angle.unsqueeze(1),
                ray_hit_hist.float().unsqueeze(1),
            ]
        else:
            channels = [
                rays_hist.unsqueeze(1),
                sin_angle.unsqueeze(1),
                cos_angle.unsqueeze(1),
                ray_hit_hist.float().unsqueeze(1),
            ]
        return torch.cat(channels, dim=1)

    def forward(self, rays_hist, ego_hist, ray_hit_hist, current_fused_rays):
        features = self.build_features(rays_hist, ego_hist, ray_hit_hist)
        encoded = self.encoder(features)
        closing = self.closing_head(encoded)
        drift = self.drift_head(
            encoded, current_fused_rays, self.d_safe, self.kappa
        )
        return {'closing': closing, 'drift': drift}

    def config_dict(self):
        return {
            'history_length': self.history_length,
            'num_rays': self.num_rays,
            'hidden_dim': self.encoder.hidden_dim,
            'temporal_pool': self.encoder.temporal_pool,
            'use_ego_motion': self.use_ego_motion,
            'ray_max': self.ray_max,
            'hit_epsilon': self.hit_epsilon,
            'd_safe': self.d_safe,
            'kappa': self.kappa,
            'ray_angles_deg': self.ray_angles.detach().cpu().tolist(),
        }
