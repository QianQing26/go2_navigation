"""Frozen online runtime for the predictive CBF drift estimator.

This module is intentionally independent from Isaac Gym.  The navigation
runner only supplies the small tensor interface exposed by ``update`` and
``reset``; estimator architecture and checkpoint details stay in
``motion_estimator``.
"""

import os

import torch

from .data.normalization import NormalizationStats
from .models.estimator import MotionEstimator


class PredictiveDriftRuntime:
    """Run a frozen MotionEstimator on asynchronous exteroception history.

    The returned drift is physical m/s and has shape ``[num_envs, 1]``.  It is
    updated only for environments whose sensor history advanced this control
    step.  Other environments retain their previous value (ZOH).
    """

    def __init__(
        self,
        checkpoint_path,
        device='cpu',
        calibration_delta=0.0,
        use_warmup_gate=True,
    ):
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                'MotionEstimator checkpoint not found: {}'.format(
                    checkpoint_path
                )
            )

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.calibration_delta = float(calibration_delta)
        self.use_warmup_gate = bool(use_warmup_gate)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_config = dict(checkpoint.get('model_config', {}))
        if 'ray_angles_deg' in model_config and 'ray_angles' not in model_config:
            model_config['ray_angles'] = model_config.pop('ray_angles_deg')
        # Older checkpoints may not contain a full model config.  The model
        # defaults are the same values used by the dataset pipeline.
        self.model = MotionEstimator(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.requires_grad_(False)

        normalization = checkpoint.get('normalization')
        if normalization is None:
            raise KeyError(
                'MotionEstimator checkpoint has no train normalization stats'
            )
        self.normalization = NormalizationStats(
            ray_max=normalization['ray_max'],
            ego_mean=normalization['ego_mean'],
            ego_std=normalization['ego_std'],
            closing_scale=normalization['closing_scale'],
            drift_scale=normalization['drift_scale'],
            d_safe=normalization['d_safe'],
            kappa=normalization['kappa'],
            horizon=normalization['horizon'],
            hit_epsilon=normalization.get('hit_epsilon', 1.0e-4),
        )

        self.cached_drift = None
        self.last_inference_mask = None
        self.inference_count = 0

    @property
    def history_length(self):
        return self.model.history_length

    @property
    def num_rays(self):
        return self.model.num_rays

    def _ensure_cache(self, num_envs, device, dtype):
        if (
            self.cached_drift is None
            or self.cached_drift.shape != (num_envs, 1)
            or self.cached_drift.device != device
            or self.cached_drift.dtype != dtype
        ):
            self.cached_drift = torch.zeros(
                num_envs, 1, device=device, dtype=dtype
            )

    @torch.inference_mode()
    def _predict_physical_drift(
        self, rays_hist, motion_ego_hist, current_fused_rays,
        ray_hit_hist=None,
    ):
        rays_hist = rays_hist.to(self.device, dtype=torch.float32)
        motion_ego_hist = motion_ego_hist.to(self.device, dtype=torch.float32)
        current_fused_rays = current_fused_rays.to(
            self.device, dtype=torch.float32
        )
        if ray_hit_hist is None:
            ray_hit_hist = rays_hist < (
                self.normalization.ray_max - self.normalization.hit_epsilon
            )
        else:
            ray_hit_hist = ray_hit_hist.to(self.device).bool()

        normalized_rays = rays_hist / self.normalization.ray_max
        normalized_ego = self.normalization.normalize_ego(motion_ego_hist)
        with torch.inference_mode():
            prediction = self.model(
                normalized_rays,
                normalized_ego,
                ray_hit_hist,
                current_fused_rays,
            )
        return (
            prediction['drift'].reshape(-1, 1).to(dtype=torch.float32)
            * self.normalization.drift_scale
            - self.calibration_delta
        )

    @torch.no_grad()
    def update(self, env):
        """Update and return ``(safety_drift, current_shield_rays)``.

        ``env`` must expose ``rays_hist``, ``motion_ego_hist``,
        ``exteroception_updated_mask`` and ``exteroception_history_count``.
        The method supports per-environment asynchronous update phases.
        """
        num_envs = int(env.num_envs)
        env_device = env.rays_hist.device
        dtype = env.rays_hist.dtype
        self._ensure_cache(num_envs, env_device, dtype)

        updated = env.exteroception_updated_mask.bool()
        history_count = env.exteroception_history_count
        ready = history_count >= self.history_length
        infer_mask = updated & (ready if self.use_warmup_gate else torch.ones_like(ready))

        if self.use_warmup_gate:
            self.cached_drift[~ready] = 0.0

        self.last_inference_mask = infer_mask.detach().clone()
        if infer_mask.any():
            ids = infer_mask.nonzero(as_tuple=False).flatten()
            drift = self._predict_physical_drift(
                env.rays_hist[ids],
                env.motion_ego_hist[ids],
                env.rays_hist[ids, -1, :],
            ).to(device=env_device, dtype=dtype)
            self.cached_drift[ids] = drift
            self.inference_count += int(ids.numel())

        # This is deliberately the latest synchronized frame, not the delayed
        # actor observation.  Clone prevents subsequent env history updates
        # from mutating the context already handed to the actor/storage.
        shield_rays = env.rays_hist[:, -1, :].clone()
        return self.cached_drift.clone(), shield_rays

    def reset(self, env_ids=None):
        """Clear cached drift for all or selected environments."""
        if self.cached_drift is None:
            return
        if env_ids is None:
            self.cached_drift.zero_()
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.cached_drift.device)
        env_ids = env_ids.to(self.cached_drift.device)
        if env_ids.dtype == torch.bool:
            env_ids = env_ids.nonzero(as_tuple=False).flatten()
        else:
            env_ids = env_ids.reshape(-1).long()
        if env_ids.numel() > 0:
            self.cached_drift[env_ids] = 0.0
