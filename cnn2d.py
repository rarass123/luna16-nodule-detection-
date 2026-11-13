"""A small 2.5D CNN for candidate false-positive reduction.

The last layer sees the whole crop, not a single averaged number, so a
bright blob can stay in one place instead of being smeared across the patch.
"""

from __future__ import annotations

import torch
from torch import nn


class TinyNoduleCNN(nn.Module):
    def __init__(self, in_channels: int = 9, size_xy: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        spatial = size_xy // 8
        self.head = nn.Linear(64 * spatial * spatial, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x).squeeze(1)
