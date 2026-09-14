import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# These constants are the single source of truth for the live CBF barrier.
# Dynamic-obstacle GT queries import the same values instead of maintaining a
# second, potentially divergent safety geometry.
DEFAULT_SAFE_RADIUS = 0.15
DEFAULT_SAFETY_MARGIN = 0.05
DEFAULT_D_SAFE = DEFAULT_SAFE_RADIUS + DEFAULT_SAFETY_MARGIN
DEFAULT_KAPPA = 10.0
DEFAULT_DAMPING_FACTOR = 1.0

class ExactLSECBFLayer(nn.Module):
    def __init__(self,
                 num_rays=41,
                 fov_deg=240.0,
                 safe_radius=DEFAULT_SAFE_RADIUS,
                 safety_margin=DEFAULT_SAFETY_MARGIN,
                 kappa=DEFAULT_KAPPA,
                 damping_factor=DEFAULT_DAMPING_FACTOR):
        super().__init__()
        
        self.d_safe = safe_radius + safety_margin
        self.kappa = kappa
        self.damping_factor = damping_factor
        
        # Pre-calculate unit direction vectors n_i
        start_angle = -np.deg2rad(fov_deg) / 2
        end_angle = np.deg2rad(fov_deg) / 2
        angles = torch.linspace(start_angle, end_angle, num_rays)
        self.register_buffer('ray_unit_vectors',
            torch.stack([torch.cos(angles), torch.sin(angles)], dim=1))

    def project(
        self, u_bar, lidar_dists, alpha, safety_drift=None,
    ):
        """Stateless closed-form projection with diagnostic tensors.

        This is deliberately free of module-state mutation.  The live
        ``forward`` path records its diagnostics from this result, while
        evaluation-only shadow/oracle projections can call the same formula
        without changing the live actor's diagnostic state.
        """

        u_2d = u_bar[:, :2]
        yaw_rate = u_bar[:, 2:]

        h_i = lidar_dists - self.d_safe
        min_h, _ = torch.min(h_i, dim=1, keepdim=True)
        h_comp = min_h - (1.0 / self.kappa) * torch.log(
            torch.sum(torch.exp(-self.kappa * (h_i - min_h)), dim=1, keepdim=True)
        )
        lambda_i = torch.exp(-self.kappa * (h_i - h_comp)).unsqueeze(-1)
        n_vecs = self.ray_unit_vectors.unsqueeze(0)
        Lg_h = -torch.sum(lambda_i * n_vecs, dim=1)
        Lgh_u = torch.sum(Lg_h * u_2d, dim=1, keepdim=True)
        Lgh_norm_sq = torch.sum(Lg_h ** 2, dim=1, keepdim=True)

        if safety_drift is None:
            safety_drift = torch.zeros_like(Lgh_u)
        elif not isinstance(safety_drift, torch.Tensor):
            safety_drift = torch.as_tensor(
                safety_drift, device=Lgh_u.device, dtype=Lgh_u.dtype
            )
        else:
            safety_drift = safety_drift.to(
                device=Lgh_u.device, dtype=Lgh_u.dtype
            )
        if safety_drift.ndim == 0:
            safety_drift = safety_drift.expand_as(Lgh_u)
        elif safety_drift.ndim == 1:
            safety_drift = safety_drift.reshape(-1, 1)
        if safety_drift.shape != Lgh_u.shape:
            raise ValueError(
                'safety_drift must have shape [B, 1], got {}'.format(
                    tuple(safety_drift.shape)
                )
            )

        alpha_h = alpha * h_comp
        nominal_barrier_residual = safety_drift + Lgh_u + alpha_h
        eta = -nominal_barrier_residual / (
            Lgh_norm_sq + self.damping_factor
        )
        u_s_2d = u_2d + F.relu(eta) * Lg_h
        u_s = torch.cat((u_s_2d, yaw_rate), dim=-1)
        return {
            'safe_action': u_s,
            'h_comp': h_comp,
            'Lgh': Lg_h,
            'Lgh_u': Lgh_u,
            'Lgh_norm_sq': Lgh_norm_sq,
            'safety_drift': safety_drift,
            'alpha_h': alpha_h,
            'nominal_barrier_residual': nominal_barrier_residual,
            'eta': eta,
            'intervention_norm': torch.linalg.vector_norm(
                u_s - u_bar, dim=-1, keepdim=True
            ),
        }

    def forward(self, u_bar, lidar_dists, alpha, safety_drift=None):
        """
        u_bar: [B, 3] Nominal policy (vx, vy, yaw)
        lidar_dists: [B, num_rays] Lidar distances (processed externally to 0.1~5.0)
        alpha: [B, 1] Class-K function parameter (adaptively learned)
        safety_drift: [B, 1] physical m/s drift term. ``None`` is zero.
        """
        diagnostics = self.project(
            u_bar, lidar_dists, alpha, safety_drift=safety_drift
        )
        u_s = diagnostics['safe_action']

        # Detached snapshots are intentionally kept outside the autograd graph
        # so evaluation diagnostics cannot affect the PPO objective.
        self.last_h_comp = diagnostics['h_comp'].detach()
        self.last_Lgh = diagnostics['Lgh'].detach()
        self.last_Lgh_u = diagnostics['Lgh_u'].detach()
        self.last_safety_drift = diagnostics['safety_drift'].detach()
        self.last_alpha_h = diagnostics['alpha_h'].detach()
        self.last_eta = diagnostics['eta'].detach()
        self.last_nominal_barrier_residual = diagnostics[
            'nominal_barrier_residual'
        ].detach()
        self.last_intervention_norm = diagnostics['intervention_norm'].detach()

        return u_s
