"""Small spatio-temporal CNN encoder with temporal attention pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = (1, dilation * 2)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=(3, 5),
            padding=padding, dilation=(1, dilation),
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=(3, 5),
            padding=padding, dilation=(1, dilation),
        )
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, input_channels=7, hidden_dim=64, temporal_pool='attention', dilations=(1, 2, 4)):
        super().__init__()
        if temporal_pool not in ('attention', 'mean', 'last'):
            raise ValueError('temporal_pool must be attention, mean, or last')
        self.hidden_dim = int(hidden_dim)
        self.temporal_pool = temporal_pool
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, int(dilation))
            for dilation in dilations
        ])
        self.temporal_query = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.temporal_query, std=0.02)

    def forward(self, features):
        # features: [B, C_in, H, R]
        encoded = self.stem(features)
        for block in self.blocks:
            encoded = block(encoded)
        if self.temporal_pool == 'mean':
            return encoded.mean(dim=2)
        if self.temporal_pool == 'last':
            return encoded[:, :, -1, :]
        # Attention is shared over rays, with a learned temporal query.
        scores = (encoded * self.temporal_query.view(1, -1, 1, 1)).sum(dim=1)
        weights = torch.softmax(scores, dim=1)
        return (encoded * weights.unsqueeze(1)).sum(dim=2)
