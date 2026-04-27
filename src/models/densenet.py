"""DenseNet121 transfer-learning wrapper for binary chest X-ray classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from src.config import NUM_CLASSES


def build_densenet121(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
