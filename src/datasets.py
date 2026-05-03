import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from collections import Counter

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=get_transforms(train=True)
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=get_transforms(train=False)
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(train_dataset) -> torch.Tensor:
    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)

    total = sum(class_counts.values())
    num_classes = len(class_counts)

    weights = [
        total / (num_classes * class_counts[i])
        for i in range(num_classes)
    ]

    return torch.tensor(weights, dtype=torch.float)


def get_weighted_sampler(train_dataset) -> WeightedRandomSampler:
    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)

    class_weights = {
        cls: 1.0 / count for cls, count in class_counts.items()
    }

    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler