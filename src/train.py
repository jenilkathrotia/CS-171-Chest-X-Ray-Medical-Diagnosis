"""Training loop for pneumonia classification models.

Run from the repo root:

    python -m src.train --model custom_cnn  --data-dir data/chest_xray --epochs 15
    python -m src.train --model densenet121 --data-dir data/chest_xray --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    LOG_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    get_device,
)
from src.datasets import compute_class_weights, get_dataloaders
from src.models import build_custom_cnn, build_densenet121

MODEL_BUILDERS = {
    "custom_cnn": build_custom_cnn,
    "densenet121": build_densenet121,
}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": running_loss / total, "accuracy": correct / total}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": running_loss / total, "accuracy": correct / total}


def _build_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[model_name]()


def train(
    model_name: str,
    data_dir: str,
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    log_dir: str | Path = LOG_DIR,
) -> dict[str, Any]:
    """Full training run for a single model.

    Saves the best-val-loss checkpoint to {checkpoint_dir}/{model_name}_best.pt
    and per-epoch metrics to {log_dir}/{model_name}.csv.
    """
    device = get_device()
    print(f"[train] device = {device}")

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    class_weights = compute_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[train] class weights = {class_weights.tolist()}")

    model = _build_model(model_name).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    ckpt_path = checkpoint_dir / f"{model_name}_best.pt"
    log_path = log_dir / f"{model_name}.csv"

    best_val_loss = math.inf
    history: list[dict[str, float]] = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va = validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch {epoch:02d}/{epochs}  "
                f"train_loss={tr['loss']:.4f} train_acc={tr['accuracy']:.4f}  |  "
                f"val_loss={va['loss']:.4f} val_acc={va['accuracy']:.4f}  "
                f"lr={current_lr:.2e}"
            )

            writer.writerow(
                [epoch, tr["loss"], tr["accuracy"], va["loss"], va["accuracy"], current_lr]
            )
            f.flush()

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr["loss"],
                    "train_acc": tr["accuracy"],
                    "val_loss": va["loss"],
                    "val_acc": va["accuracy"],
                    "lr": current_lr,
                }
            )

            scheduler.step(va["loss"])

            if va["loss"] < best_val_loss:
                best_val_loss = va["loss"]
                torch.save(
                    {
                        "model_name": model_name,
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_loss": va["loss"],
                        "val_acc": va["accuracy"],
                    },
                    ckpt_path,
                )
                print(f"  -> saved checkpoint: {ckpt_path}")

    final = history[-1] if history else {}
    return {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(ckpt_path),
        "log_path": str(log_path),
        "final_epoch": final,
        "history": history,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pneumonia classification models.")
    p.add_argument("--model", required=True, choices=sorted(MODEL_BUILDERS))
    p.add_argument("--data-dir", required=True, help="Path to ImageFolder root with train/val/test")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    p.add_argument("--log-dir", default=str(LOG_DIR))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
