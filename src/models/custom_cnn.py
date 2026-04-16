"""Lightweight custom CNN with depthwise-separable convolutions.

Designed for the 5,863-image Chest X-Ray dataset where a standard deep CNN
would overfit quickly.  Separable convolutions reduce learnable parameters
by ~8-9x per layer compared to standard convolutions.

Public API:
    SeparableConv2d -- reusable depthwise-separable conv block
    CustomCNN       -- full classification network
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import DROPOUT, NUM_CLASSES


class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution block.

    Depthwise conv (one filter per input channel) followed by a pointwise
    1x1 conv that mixes channels, with BatchNorm and ReLU.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Spatial kernel size for the depthwise conv.
        padding:      Padding for the depthwise conv.
    """

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
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class CustomCNN(nn.Module):
    """Lightweight pneumonia classifier using separable convolutions.

    Architecture::

        Input (B, 3, 224, 224)
          -> Conv2d(3, 32) + BN + ReLU + MaxPool
          -> SeparableConv2d(32, 64)   + MaxPool
          -> SeparableConv2d(64, 128)  + MaxPool
          -> SeparableConv2d(128, 256) + MaxPool
          -> AdaptiveAvgPool2d(1)
          -> Dropout -> Linear(256, num_classes)

    Args:
        in_channels: Number of input image channels (3 for RGB).
        num_classes: Number of output classes.
        dropout:     Dropout probability before the FC head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.sep_block1 = nn.Sequential(SeparableConv2d(32, 64), nn.MaxPool2d(2))
        self.sep_block2 = nn.Sequential(SeparableConv2d(64, 128), nn.MaxPool2d(2))
        self.sep_block3 = nn.Sequential(SeparableConv2d(128, 256), nn.MaxPool2d(2))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_block(x)
        x = self.sep_block1(x)
        x = self.sep_block2(x)
        x = self.sep_block3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
