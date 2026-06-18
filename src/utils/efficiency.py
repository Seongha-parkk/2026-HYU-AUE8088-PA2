"""Inference FPS measurement on a single GPU — used in Level 4."""
from __future__ import annotations

import time

import torch
from torch import nn


@torch.no_grad()
def measure_fps(
    model: nn.Module,
    device: torch.device,
    input_size: tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    n_warmup: int = 20,
    n_iter: int = 200,
) -> float:
    """Measure FPS = (batch_size * n_iter) / total_time.

    Defaults follow the README spec: batch=1, 224x224, after warm-up.
    """
    model.eval().to(device)
    x = torch.randn(batch_size, *input_size, device=device)

    for _ in range(n_warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (batch_size * n_iter) / elapsed
