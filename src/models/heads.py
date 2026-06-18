"""Multi-task classification heads — shared across all backbones."""
from __future__ import annotations

import torch
from torch import nn

from src.datasets.bdd_attr import NUM_CLASSES


class MultiTaskHead(nn.Module):
    """Three independent linear heads on top of a shared feature vector."""

    def __init__(self, in_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weather = nn.Linear(in_features, NUM_CLASSES["weather"])
        self.scene = nn.Linear(in_features, NUM_CLASSES["scene"])
        self.timeofday = nn.Linear(in_features, NUM_CLASSES["timeofday"])

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.dropout(feat)
        return {
            "weather": self.weather(feat),
            "scene": self.scene(feat),
            "timeofday": self.timeofday(feat),
        }
