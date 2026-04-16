"""Data loading, augmentation, and class-weight computation.

Public API:
    get_transforms   -- build train / eval transform pipelines
    get_dataloaders  -- load train / val / test splits via ImageFolder
    compute_class_weights -- inverse-frequency weights for CrossEntropyLoss
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    NUM_WORKERS,
)


def get_transforms(mode: str = "train") -> transforms.Compose:
    """Return an image transform pipeline.

    Args:
        mode: ``"train"`` for augmented pipeline, ``"eval"`` for deterministic
              inference pipeline.

    Returns:
        A ``transforms.Compose`` instance.
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict[str, DataLoader]:
    """Create DataLoaders for train / val / test splits.

    Expects ``data_dir`` to contain ``train/``, ``val/``, and ``test/``
    subdirectories, each with ``NORMAL/`` and ``PNEUMONIA/`` class folders
    (standard Kaggle layout).

    Args:
        data_dir: Root directory that contains the three split folders.
        batch_size: Mini-batch size.
        num_workers: Parallel data-loading workers.

    Returns:
        Dict mapping split name to its ``DataLoader``.
    """
    data_dir = Path(data_dir)

    split_config = {
        "train": {"transform": get_transforms("train"), "shuffle": True, "drop_last": True},
        "val":   {"transform": get_transforms("eval"),  "shuffle": False, "drop_last": False},
        "test":  {"transform": get_transforms("eval"),  "shuffle": False, "drop_last": False},
    }

    loaders: dict[str, DataLoader] = {}
    for split, cfg in split_config.items():
        split_path = data_dir / split
        if not split_path.exists():
            continue

        ds = datasets.ImageFolder(root=str(split_path), transform=cfg["transform"])
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=cfg["shuffle"],
            num_workers=num_workers,
            drop_last=cfg["drop_last"],
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Compute inverse-frequency class weights for ``CrossEntropyLoss``.

    Formula per class *c*::

        w_c = total_samples / (num_classes * count_c)

    Args:
        dataset: An ``ImageFolder`` dataset (must expose ``.targets``).

    Returns:
        Float tensor of shape ``(NUM_CLASSES,)`` ordered by class index.
    """
    counts = Counter(dataset.targets)
    total = len(dataset.targets)
    weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for cls_idx in range(NUM_CLASSES):
        weights[cls_idx] = total / (NUM_CLASSES * counts[cls_idx])
    return weights
