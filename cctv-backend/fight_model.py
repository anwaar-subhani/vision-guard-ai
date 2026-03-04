"""
Fight model helpers used by fight detector.

Required exports:
- NUM_FRAMES
- FRAME_SIZE
- DEVICE
- get_transforms(split)
- get_improved_model(num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r2plus1d_18


# ===========================================================================
# CONFIGURATION — you can tune these
# ===========================================================================

NUM_FRAMES = 16
FRAME_SIZE = 112

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Kinetics-400 normalization (commonly used for torchvision video models)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]


def get_transforms(split: str = "val"):
    """Return per-frame image transforms used before clip stacking."""
    base = [
        transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
        ])

    return transforms.Compose(base)


def get_improved_model(num_classes: int = 2) -> nn.Module:
    """Build a fight classifier compatible with the uploaded checkpoint."""
    model = r2plus1d_18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model
