"""Class-balanced samplers for the multi-task setting.

Multi-task Imbalance is *not* a solved problem — students must decide
which attribute to balance against (or design a hybrid). The helper
below balances against a single attribute. Extending it to a joint
balancing scheme is part of Level 3.
"""
from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from .bdd_attr import BDDAttrDataset


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
