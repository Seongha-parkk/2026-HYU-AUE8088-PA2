"""Loss functions for class imbalance — for use in Level 3.

All losses operate on a single attribute. To apply different losses per
attribute (e.g. Focal for ``weather``, plain CE for ``scene``), pass a
dict like ``{"weather": FocalLoss(...), "scene": nn.CrossEntropyLoss(), ...}``
to ``MultiTaskTrainer``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., ICCV 2017).

    L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss (Cui et al., CVPR 2019).

    Effective number of samples re-weighting:
        w_c = (1 - beta) / (1 - beta ^ n_c)
    """

    def __init__(self, samples_per_class: torch.Tensor, beta: float = 0.9999) -> None:
        super().__init__()
        eff_num = 1.0 - torch.pow(beta, samples_per_class.float())
        weights = (1.0 - beta) / eff_num.clamp(min=1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer("weights", weights, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights)


class LDAMLoss(nn.Module):
    """LDAM Loss (Cao et al., NeurIPS 2019).

    Margin per class is proportional to n_c^(-1/4); rare classes get
    a wider margin.
    """

    def __init__(self, samples_per_class: torch.Tensor, max_m: float = 0.5, s: float = 30.0) -> None:
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(samples_per_class.float()))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list, persistent=False)
        self.s = s

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        batch_m = self.m_list[targets].view(-1, 1)
        logits_m = logits - index * batch_m
        return F.cross_entropy(self.s * logits_m, targets)


def weighted_cross_entropy(samples_per_class: torch.Tensor) -> nn.CrossEntropyLoss:
    """Plain inverse-frequency weighted CE — the simplest baseline."""
    weights = 1.0 / samples_per_class.float().clamp(min=1)
    weights = weights / weights.sum() * len(samples_per_class)
    return nn.CrossEntropyLoss(weight=weights)
