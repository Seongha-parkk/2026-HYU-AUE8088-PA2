"""ResNet-18 / ResNet-50 (He et al., CVPR 2016) — student implementation.

Use of ``torchvision.models.resnet*`` and ``timm`` is forbidden.
Reference: https://arxiv.org/abs/1512.03385

Two block types:
  - ``BasicBlock``  (used by ResNet-18 / -34) — 2 x (3x3 conv)
  - ``Bottleneck``  (used by ResNet-50 / -101 / -152) — 1x1 → 3x3 → 1x1 with 4x expansion
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.heads import MultiTaskHead


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c: int, mid_c: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        out_c = mid_c * self.expansion
        self.conv1 = conv1x1(in_c, mid_c)
        self.bn1   = nn.BatchNorm2d(mid_c)
        self.conv2 = conv3x3(mid_c, mid_c, stride)
        self.bn2   = nn.BatchNorm2d(mid_c)
        self.conv3 = conv1x1(mid_c, out_c)
        self.bn3   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block: type[nn.Module], layers: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        self.in_c = 64

        # Stem.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages.
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 512 * block.expansion
        self.head = MultiTaskHead(in_features=feat_dim, dropout=dropout)

        self._init_weights()

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        out_c = planes * block.expansion
        downsample = None
        if stride != 1 or self.in_c != out_c:
            downsample = nn.Sequential(
                conv1x1(self.in_c, out_c, stride),
                nn.BatchNorm2d(out_c),
            )

        layers = [block(self.in_c, planes, stride, downsample)]
        self.in_c = out_c
        for _ in range(1, blocks):
            layers.append(block(self.in_c, planes))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def resnet18() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet50() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3])
