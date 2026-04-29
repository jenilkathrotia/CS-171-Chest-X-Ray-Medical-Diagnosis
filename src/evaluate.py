"""Evaluation utilities for chest X-ray pneumonia classifiers.

This module provides reusable functions and a CLI to:
- load a trained checkpoint for a supported model architecture,
- run inference on the test split from ``src.datasets.get_dataloaders``,
- compute sklearn metrics (accuracy, precision, recall, F1),
- explicitly report pneumonia recall (clinical sensitivity),
- persist metrics as JSON for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, NUM_WORKERS, RESULTS_DIR, get_device
from src.datasets import get_dataloaders
from src.models import build_custom_cnn, build_densenet121

MODEL_BUILDERS: dict[str, Any] = {
    "custom_cnn": build_custom_cnn,
    # For evaluation we load fully trained checkpoint weights, so downloading
    # ImageNet defaults is unnecessary and can fail in offline environments.
    "densenet121": lambda: build_densenet121(pretrained=False),
}


def build_model(model_name: str) -> nn.Module:
    """Build a model instance from its registered name."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[model_name]()


def resolve_checkpoint_path(
    model_name: str,
    checkpoint_path: str | None = None,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
) -> Path:
    """Resolve checkpoint path and validate it exists."""
    if checkpoint_path:
        resolved = Path(checkpoint_path).expanduser().resolve()
    else:
        resolved = Path(checkpoint_dir).expanduser().resolve() / f"{model_name}_best.pt"

    if not resolved.exists():
        raise FileNotFoundError(
            "Checkpoint not found at "
            f"'{resolved}'. Provide --checkpoint explicitly or confirm --checkpoint-dir."
        )

    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint path is not a file: '{resolved}'")

    return resolved


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load model weights from a checkpoint file.

    Supports both:
    - training-style checkpoints with ``state_dict`` and metadata, and
    - raw ``state_dict`` objects.
    """
    raw_checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata: dict[str, Any] = {}

    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        state_dict = raw_checkpoint["state_dict"]
        metadata = {k: v for k, v in raw_checkpoint.items() if k != "state_dict"}
    elif isinstance(raw_checkpoint, dict):
        state_dict = raw_checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint format in '{checkpoint_path}'. Expected a dict."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not match model architecture. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )

    return metadata


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run model inference on a loader and collect true/predicted labels."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


def _resolve_pneumonia_label(class_names: Sequence[str]) -> str | None:
    """Find the class name corresponding to pneumonia if present."""
    for name in class_names:
        if "pneumonia" in name.lower():
            return name
    return None


def evaluate_model(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute a full metric bundle with pneumonia recall emphasized."""
    if len(y_true) == 0:
        raise ValueError("Evaluation received 0 samples.")

    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        target_names=list(class_names),
        zero_division=0,
        digits=4,
    )
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    pneumonia_label = _resolve_pneumonia_label(class_names)
    pneumonia_recall = None
    if pneumonia_label is not None and pneumonia_label in report_dict:
        pneumonia_recall = report_dict[pneumonia_label]["recall"]

    metrics: dict[str, Any] = {
        "num_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "pneumonia_label": pneumonia_label,
        "pneumonia_recall_sensitivity": pneumonia_recall,
    }
    return metrics


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Save metrics dictionary to JSON with stable formatting."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return destination


def run_evaluation(
    model_name: str,
    data_dir: str | Path = DATA_DIR,
    checkpoint_path: str | None = None,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    output_dir: str | Path = RESULTS_DIR / "metrics",
) -> tuple[dict[str, Any], Path]:
    """Execute end-to-end evaluation and persist JSON results."""
    device = get_device()
    model = build_model(model_name).to(device)

    resolved_checkpoint = resolve_checkpoint_path(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
    )
    checkpoint_meta = load_checkpoint(model, resolved_checkpoint, device)

    _train_loader, _val_loader, test_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    class_names = list(test_loader.dataset.classes)
    y_true, y_pred = collect_predictions(model, test_loader, device)
    metrics = evaluate_model(y_true=y_true, y_pred=y_pred, class_names=class_names)
    metrics["model_name"] = model_name
    metrics["device"] = str(device)
    metrics["checkpoint_path"] = str(resolved_checkpoint)
    metrics["checkpoint_metadata"] = checkpoint_meta

    output_path = Path(output_dir).expanduser().resolve() / f"{model_name}_metrics.json"
    saved_path = save_metrics_json(metrics, output_path)
    return metrics, saved_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained chest X-ray classifiers.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BUILDERS))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint .pt/.pth path")
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "metrics"))
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for test-set evaluation."""
    args = _parse_args()
    metrics, metrics_path = run_evaluation(
        model_name=args.model,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    print(f"[evaluate] model = {metrics['model_name']}")
    print(f"[evaluate] checkpoint = {metrics['checkpoint_path']}")
    print(f"[evaluate] samples = {metrics['num_samples']}")
    print(f"[evaluate] accuracy = {metrics['accuracy']:.4f}")

    pneumonia_recall = metrics.get("pneumonia_recall_sensitivity")
    if pneumonia_recall is not None:
        label = metrics.get("pneumonia_label", "PNEUMONIA")
        print(f"[evaluate] {label} recall (sensitivity) = {pneumonia_recall:.4f}")
    else:
        print("[evaluate] Pneumonia class not detected in class_names; recall not reported.")

    print("\n[evaluate] classification report")
    print(metrics["classification_report_text"])
    print(f"[evaluate] saved metrics JSON -> {metrics_path}")


if __name__ == "__main__":
    main()
