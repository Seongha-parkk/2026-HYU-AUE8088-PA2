"""VGG-16 (Simonyan & Zisserman, ICLR 2015) — student implementation.

You may NOT use ``torchvision.models.vgg16`` or ``timm.create_model``.
You MAY read those reference implementations and re-type the architecture.

Skeleton scaffolding (forward pass, head wiring) is provided. Fill in the
``# TODO`` blocks.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.heads import MultiTaskHead


# Standard VGG-16 layer configuration (numbers = output channels, "M" = maxpool).
VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


def make_vgg_layers(cfg: list, batch_norm: bool = True) -> nn.Sequential:
    """Build the convolutional feature extractor from ``cfg``.

    TODO: For each entry in cfg:
      - "M": append nn.MaxPool2d(kernel_size=2, stride=2)
      - int v: append Conv2d(in -> v, k=3, p=1) → (BN) → ReLU(inplace=True)
    Return as nn.Sequential.
    """
    layers: list[nn.Module] = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v

    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG-16-BN with a multi-task classification head."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = make_vgg_layers(VGG16_CFG, batch_norm=True)

        # After 5 maxpool stages, 224x224 -> 7x7. Channels = 512.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # TODO: classifier MLP. The original VGG uses:
        #   Linear(512*7*7, 4096) -> ReLU -> Dropout
        #   Linear(4096, 4096)    -> ReLU -> Dropout
        # Here we end at a 4096-dim feature vector and hand it to the
        # multi-task head.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.head = MultiTaskHead(in_features=4096, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.head(x)
