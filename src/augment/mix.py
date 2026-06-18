"""Mixup / CutMix for the multi-task setting.

The standard formulation mixes a single label. Here, we apply the same
mixing coefficient ``lam`` to all three attribute targets — a common
choice that keeps the augmentation simple. Students may experiment with
per-attribute lambdas in Level 3.
"""
from __future__ import annotations

import numpy as np
import torch

from src.datasets.bdd_attr import ATTRIBUTES


def mixup_data(
    images: torch.Tensor,
    targets: dict[str, torch.Tensor],
    alpha: float = 0.2,
) -> tuple[torch.Tensor, dict, dict, float]:
    """Returns ``(mixed_images, targets_a, targets_b, lam)``."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(images.size(0), device=images.device)

    mixed = lam * images + (1 - lam) * images[perm]
    targets_b = {a: targets[a][perm] for a in ATTRIBUTES}
    return mixed, targets, targets_b, lam


def cutmix_data(
    images: torch.Tensor,
    targets: dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict, dict, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(images.size(0), device=images.device)

    B, _, H, W = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cw, ch = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)

    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

    # Adjust lam to reflect the actual area swapped.
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    targets_b = {a: targets[a][perm] for a in ATTRIBUTES}
    return images, targets, targets_b, lam


def mixed_loss(
    loss_fns: dict,
    logits: dict[str, torch.Tensor],
    targets_a: dict[str, torch.Tensor],
    targets_b: dict[str, torch.Tensor],
    lam: float,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Lam-weighted sum of CE on (a) and CE on (b) for each attribute."""
    weights = weights or {a: 1.0 for a in ATTRIBUTES}
    total = 0.0
    for a in ATTRIBUTES:
        la = loss_fns[a](logits[a], targets_a[a])
        lb = loss_fns[a](logits[a], targets_b[a])
        total = total + weights[a] * (lam * la + (1 - lam) * lb)
    return total
