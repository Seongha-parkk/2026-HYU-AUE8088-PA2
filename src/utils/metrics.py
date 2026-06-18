"""Evaluation metrics for multi-task scene classification.

Primary:    Average Macro-F1 across the 3 attributes.
Secondary:  Average mean-Average-Precision across the 3 attributes.
Mandatory:  Per-attribute Confusion Matrices.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.datasets.bdd_attr import (
    ATTRIBUTES,
    NUM_CLASSES,
    SCENE_CLASSES,
    TIMEOFDAY_CLASSES,
    WEATHER_CLASSES,
)


CLASS_NAMES = {
    "weather": WEATHER_CLASSES,
    "scene": SCENE_CLASSES,
    "timeofday": TIMEOFDAY_CLASSES,
}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=np.arange(num_classes), zero_division=0))


def per_attribute_macro_f1(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, float]:
    return {
        a: macro_f1(targets[a], preds[a], NUM_CLASSES[a])
        for a in ATTRIBUTES
    }


def average_macro_f1(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> float:
    """Primary metric — Avg over the 3 attributes."""
    per = per_attribute_macro_f1(preds, targets)
    return float(np.mean(list(per.values())))


def per_attribute_mAP(
    probs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """One-vs-rest mAP per attribute. ``probs[a]`` has shape (N, C_a)."""
    out = {}
    for a in ATTRIBUTES:
        n_classes = NUM_CLASSES[a]
        y_true_oh = np.eye(n_classes)[targets[a]]
        # average='macro' gives mean of per-class AP — i.e. mAP for this attr.
        out[a] = float(
            average_precision_score(y_true_oh, probs[a], average="macro")
        )
    return out


def average_mAP(
    probs: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> float:
    return float(np.mean(list(per_attribute_mAP(probs, targets).values())))


def confusion_matrices(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    normalize: str = "true",
) -> dict[str, np.ndarray]:
    """Returns a dict ``{attribute: cm}`` of normalized confusion matrices."""
    out = {}
    for a in ATTRIBUTES:
        out[a] = confusion_matrix(
            targets[a],
            preds[a],
            labels=np.arange(NUM_CLASSES[a]),
            normalize=normalize,
        )
    return out


def per_class_prf(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    out = {}
    for a in ATTRIBUTES:
        p, r, f, sup = precision_recall_fscore_support(
            targets[a],
            preds[a],
            labels=np.arange(NUM_CLASSES[a]),
            zero_division=0,
        )
        out[a] = {
            "class": CLASS_NAMES[a],
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "support": sup.tolist(),
        }
    return out


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[dict, dict, dict, list[str]]:
    """Run inference and collect per-attribute argmax preds + softmax probs.

    Returns:
        preds:    {attr: (N,) np.int64}
        probs:    {attr: (N, C_attr) np.float32}
        targets:  {attr: (N,) np.int64}
        ids:      list of image_ids in order
    """
    model.eval()
    out_logits = {a: [] for a in ATTRIBUTES}
    out_target = {a: [] for a in ATTRIBUTES}
    ids: list[str] = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)  # dict of {attr: (B, C_attr)}
        for a in ATTRIBUTES:
            out_logits[a].append(logits[a].cpu())
            out_target[a].append(batch[a])
        ids.extend(batch["image_id"])

    preds, probs, targets = {}, {}, {}
    for a in ATTRIBUTES:
        logit = torch.cat(out_logits[a], dim=0)
        probs[a] = torch.softmax(logit, dim=-1).numpy().astype(np.float32)
        preds[a] = logit.argmax(dim=-1).numpy().astype(np.int64)
        targets[a] = torch.cat(out_target[a], dim=0).numpy().astype(np.int64)
    return preds, probs, targets, ids
