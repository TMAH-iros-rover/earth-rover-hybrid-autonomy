from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_action_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
