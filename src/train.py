"""Training loop for pneumonia classification models.

Public API:
    train_one_epoch -- run one training pass over a DataLoader
    validate        -- run one evaluation pass (no gradient)
    run_training    -- full train/val loop with checkpointing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, DEVICE, LEARNING_RATE, NUM_EPOCHS


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """Train *model* for a single epoch.

    Args:
        model:     The network to train.
        loader:    Training ``DataLoader``.
        criterion: Loss function (e.g. ``CrossEntropyLoss``).
        optimizer: Parameter optimizer.
        device:    Computation device.

    Returns:
        Dict with ``"loss"`` (average batch loss) and ``"accuracy"``.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> dict[str, float]:
    """Evaluate *model* on a validation or test set.

    Args:
        model:     The network to evaluate.
        loader:    Eval ``DataLoader``.
        criterion: Loss function.
        device:    Computation device.

    Returns:
        Dict with ``"loss"`` and ``"accuracy"``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def run_training(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    num_epochs: int = NUM_EPOCHS,
    device: torch.device = DEVICE,
    model_name: str = "model",
) -> dict[str, Any]:
    """Full training loop with validation and best-model checkpointing.

    Args:
        model:       Network to train.
        dataloaders: Dict with at least ``"train"`` and optionally ``"val"`` keys.
        criterion:   Loss function.
        optimizer:   Parameter optimizer.
        scheduler:   Optional learning-rate scheduler (stepped per epoch).
        num_epochs:  Total training epochs.
        device:      Computation device.
        model_name:  Base name for checkpoint files.

    Returns:
        Dict containing ``"train_history"`` and ``"val_history"`` (lists of
        per-epoch metric dicts) and ``"best_val_accuracy"``.
    """
    model.to(device)

    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        train_history.append(train_metrics)

        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"Train Loss: {train_metrics['loss']:.4f}  "
            f"Train Acc: {train_metrics['accuracy']:.4f}",
            end="",
        )

        if "val" in dataloaders:
            val_metrics = validate(model, dataloaders["val"], criterion, device)
            val_history.append(val_metrics)
            print(
                f"  |  Val Loss: {val_metrics['loss']:.4f}  "
                f"Val Acc: {val_metrics['accuracy']:.4f}",
                end="",
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
                torch.save(model.state_dict(), ckpt_path)

        print()

        if scheduler is not None:
            scheduler.step()

    return {
        "train_history": train_history,
        "val_history": val_history,
        "best_val_accuracy": best_val_acc,
    }
