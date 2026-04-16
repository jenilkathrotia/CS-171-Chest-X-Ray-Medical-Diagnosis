"""Model evaluation: metrics computation and confusion-matrix plotting.

Public API:
    collect_predictions   -- gather y_true / y_pred from a model + DataLoader
    compute_metrics       -- classification report + confusion matrix
    plot_confusion_matrix -- seaborn heatmap saved to disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, DEVICE, RESULTS_DIR


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> tuple[list[int], list[int]]:
    """Run *model* over a DataLoader and collect ground-truth / predicted labels.

    Args:
        model:  The trained network.
        loader: Eval ``DataLoader`` (test or val split).
        device: Computation device.

    Returns:
        Tuple of ``(y_true, y_pred)`` as plain Python int lists.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_true.extend(labels.tolist())
        y_pred.extend(preds)

    return y_true, y_pred


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Compute classification metrics and confusion matrix.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: Human-readable class names.

    Returns:
        Dict with ``"report"`` (sklearn classification_report as dict)
        and ``"confusion_matrix"`` (numpy array).
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    return {"report": report, "confusion_matrix": cm}


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str] = CLASS_NAMES,
    save_path: str | Path | None = None,
) -> None:
    """Generate and save a confusion-matrix heatmap.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: Human-readable class names.
        save_path:   Where to save the figure. Defaults to
                     ``results/confusion_matrix.png``.
    """
    if save_path is None:
        save_path = RESULTS_DIR / "confusion_matrix.png"

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    labels = np.array([
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm, cm_pct)
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
