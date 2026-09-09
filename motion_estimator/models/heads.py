"""Per-ray and scalar drift heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClosingHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, features):
        return self.net(features).squeeze(1)


class DriftHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.angular_score = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features, current_rays, d_safe, kappa):
        h = current_rays - float(d_safe)
        weights = torch.softmax(-float(kappa) * h, dim=-1)
        z_cbf = (features * weights.unsqueeze(1)).sum(dim=-1)
        angular_logits = self.angular_score(features).squeeze(1)
        angular_weights = torch.softmax(angular_logits, dim=-1)
        z_global = (features * angular_weights.unsqueeze(1)).sum(dim=-1)
        return self.mlp(torch.cat((z_cbf, z_global), dim=-1)).squeeze(-1)
