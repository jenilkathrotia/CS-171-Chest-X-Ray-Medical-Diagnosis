"""Lightweight custom CNN with depthwise-separable convolutions.

Architecture
------------
    Input (B, 3, 224, 224)
      Stem:    Conv2d(3 -> 32, k=3, s=2) + BN + ReLU
      Block 1: SeparableConv2d(32  -> 64)  + MaxPool(2)
      Block 2: SeparableConv2d(64  -> 128) + MaxPool(2)
      Block 3: SeparableConv2d(128 -> 256) + MaxPool(2)
      Block 4: SeparableConv2d(256 -> 512)
      GlobalAvgPool -> Dropout(0.3) -> Linear(512 -> num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import DROPOUT, NUM_CLASSES


class SeparableConv2d(nn.Module):
    """Depthwise-separable conv: depthwise (groups=in) + pointwise 1x1, with BN+ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class CustomCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(SeparableConv2d(32, 64), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(SeparableConv2d(64, 128), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(SeparableConv2d(128, 256), nn.MaxPool2d(2))
        self.block4 = SeparableConv2d(256, 512)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_custom_cnn(num_classes: int = NUM_CLASSES) -> CustomCNN:
    return CustomCNN(num_classes=num_classes)
