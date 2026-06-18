"""Swin Transformer (Liu et al., ICCV 2021) — student implementation.

Swin-Tiny: dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=7.

Reference (read, do NOT import):
    https://arxiv.org/abs/2103.14030
    https://github.com/microsoft/Swin-Transformer

This is a heavier implementation than ViT — Swin is recommended only if
you've finished ViT and want a stronger backbone for Level 4 / 5.
You may instead skip Swin and go deeper on ViT analysis.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.heads import MultiTaskHead


# TODO (Level 2 — Swin-Tiny):
#   - PatchEmbed (4x4 patches)
#   - PatchMerging (downsample by 2x between stages)
#   - Window MSA + Shifted Window MSA
#   - Stage = sequence of (W-MSA, SW-MSA) pairs
#   - 4 stages with depths=(2, 2, 6, 2), dims=(96, 192, 384, 768)
#   - Final LayerNorm + GAP -> MultiTaskHead(in_features=768)
#
# A reasonable implementation strategy:
#   1) Build PatchEmbed and a single SwinBlock first; verify shapes on a
#      dummy (B=1, 3, 224, 224) input.
#   2) Add cyclic shift + masking for SW-MSA.
#   3) Add PatchMerging and stack 4 stages.


class SwinTiny(nn.Module):
    def __init__(self, head_dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: build the full architecture as described above.
        # The forward pass should end at a (B, 768) feature vector handed
        # to ``MultiTaskHead(in_features=768)``.
        raise NotImplementedError("Level 2: (optional) implement SwinTiny")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError
