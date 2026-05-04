"""Residual MLP regressor for tabular multi-output tasks."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(1, int(n_blocks))):
            blocks.append(ResidualBlock(hidden_dim, dropout=dropout))
        blocks.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
