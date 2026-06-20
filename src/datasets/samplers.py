"""Class-balanced samplers for the multi-task setting.

Multi-task Imbalance is *not* a solved problem — students must decide
which attribute to balance against (or design a hybrid). The helper
below balances against a single attribute. Extending it to a joint
balancing scheme is part of Level 3.
"""
from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from .bdd_attr import ATTRIBUTES, BDDAttrDataset


def class_balanced_sampler(
    dataset: BDDAttrDataset,
    attribute: str = "weather",
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Inverse-frequency sampling over a single attribute."""
    counts = dataset.class_counts(attribute).float()
    # Avoid division by zero for absent classes.
    inv_freq = 1.0 / counts.clamp(min=1)

    weights = torch.zeros(len(dataset))
    for i, s in enumerate(dataset.samples):
        label = getattr(s, attribute)
        if label >= 0:
            weights[i] = inv_freq[label]

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=num_samples or len(dataset),
        replacement=True,
    )


def multi_attr_balanced_sampler(
    dataset: BDDAttrDataset,
    ir_weights: dict[str, float],
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """IR-weighted multi-attribute balanced sampler.

    각 샘플 가중치 = Σ_a (IR_a / ΣIR) × inv_freq_a[label_a]
    → 3속성 모두의 소수 클래스를 통합적으로 고려.
    """
    counts   = {a: dataset.class_counts(a).float() for a in ATTRIBUTES}
    inv_freq = {a: 1.0 / counts[a].clamp(min=1)   for a in ATTRIBUTES}
    ir_total = sum(ir_weights.values())

    weights = torch.zeros(len(dataset))
    for i, s in enumerate(dataset.samples):
        w = 0.0
        for a in ATTRIBUTES:
            label = getattr(s, a)
            if label >= 0:
                w += (ir_weights[a] / ir_total) * inv_freq[a][label].item()
        weights[i] = w

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=num_samples or len(dataset),
        replacement=True,
    )
