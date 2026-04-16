"""DenseNet121 transfer-learning wrapper.

Loads Torchvision's DenseNet121 with ImageNet-pretrained weights and replaces
the classification head for binary pneumonia detection.

Public API:
    build_densenet121 -- factory that returns a ready-to-train model
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.config import NUM_CLASSES


def build_densenet121(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build a DenseNet121 model with a fresh classification head.

    Args:
        num_classes: Number of output classes (default 2).
        pretrained:  If ``True``, load ImageNet weights; otherwise random init.

    Returns:
        A ``densenet121`` model with the classifier replaced by
        ``nn.Linear(1024, num_classes)``.
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
