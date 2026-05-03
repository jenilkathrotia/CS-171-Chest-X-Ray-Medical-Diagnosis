"""Visualization and interpretability utilities for Part 3.

Features:
- Plot training/validation loss and accuracy from CSV logs.
- Build labeled confusion matrix visualizations.
- Generate Grad-CAM overlays for chest X-ray predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LOG_DIR,
    NUM_WORKERS,
    RESULTS_DIR,
    get_device,
)
from src.datasets import get_dataloaders
from src.evaluate import build_model, load_checkpoint, resolve_checkpoint_path


def _ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def plot_training_curves(
    log_paths: Sequence[str | Path],
    output_dir: str | Path = RESULTS_DIR / "plots",
) -> list[Path]:
    """Plot train/val loss and accuracy curves from one or more CSV logs."""
    if not log_paths:
        raise ValueError("No log paths were provided for plotting.")

    output_root = _ensure_dir(output_dir)
    saved: list[Path] = []
    frames: list[tuple[str, pd.DataFrame]] = []

    for log_path in log_paths:
        path = Path(log_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: '{path}'")

        frame = pd.read_csv(path)
        required_cols = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
        missing_cols = required_cols - set(frame.columns)
        if missing_cols:
            raise ValueError(f"Log file '{path}' is missing columns: {sorted(missing_cols)}")

        model_name = path.stem
        frames.append((model_name, frame))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_loss, ax_acc = axes

        ax_loss.plot(frame["epoch"], frame["train_loss"], label="Train Loss", linewidth=2)
        ax_loss.plot(frame["epoch"], frame["val_loss"], label="Val Loss", linewidth=2)
        ax_loss.set_title(f"{model_name} Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend()

        ax_acc.plot(frame["epoch"], frame["train_acc"], label="Train Accuracy", linewidth=2)
        ax_acc.plot(frame["epoch"], frame["val_acc"], label="Val Accuracy", linewidth=2)
        ax_acc.set_title(f"{model_name} Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.grid(alpha=0.3)
        ax_acc.legend()

        fig.tight_layout()
        out_path = output_root / f"{model_name}_training_curves.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    if len(frames) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_loss, ax_acc = axes
        for model_name, frame in frames:
            ax_loss.plot(frame["epoch"], frame["val_loss"], linewidth=2, label=model_name)
            ax_acc.plot(frame["epoch"], frame["val_acc"], linewidth=2, label=model_name)

        ax_loss.set_title("Validation Loss Comparison")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Val Loss")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend()

        ax_acc.set_title("Validation Accuracy Comparison")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Val Accuracy")
        ax_acc.grid(alpha=0.3)
        ax_acc.legend()

        fig.tight_layout()
        compare_path = output_root / "model_comparison_validation_curves.png"
        fig.savefig(compare_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(compare_path)

    return saved


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: str | Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> Path:
    """Generate and save a labeled confusion matrix heatmap."""
    if len(y_true) == 0:
        raise ValueError("Cannot plot confusion matrix for empty predictions.")

    labels = list(range(len(class_names)))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    cm_display = cm.astype(float)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_display, row_sums, where=row_sums != 0)

    annotations = np.empty_like(cm_display, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            if normalize:
                value = f"{cm_display[i, j]:.3f}\n({count})"
            else:
                value = str(count)

            if cm.shape == (2, 2):
                cell_name = [["TN", "FP"], ["FN", "TP"]][i][j]
                annotations[i, j] = f"{cell_name}\n{value}"
            else:
                annotations[i, j] = value

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close()
    return destination


def plot_confusion_matrix_from_metrics_json(
    metrics_json_path: str | Path,
    output_path: str | Path | None = None,
    normalize: bool = False,
) -> Path:
    """Plot confusion matrix from a metrics JSON saved by ``src.evaluate``."""
    path = Path(metrics_json_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: '{path}'")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "confusion_matrix" not in payload or "class_names" not in payload:
        raise ValueError("Metrics JSON must contain 'confusion_matrix' and 'class_names'.")

    cm = np.array(payload["confusion_matrix"], dtype=float)
    class_names = payload["class_names"]
    model_name = payload.get("model_name", path.stem)
    if output_path is None:
        output_path = RESULTS_DIR / "confusion_matrices" / f"{model_name}_confusion_matrix.png"

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    display = cm.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, where=row_sums != 0)

    annotations = np.empty_like(display, dtype=object)
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            count = int(cm[i, j])
            if normalize:
                text = f"{display[i, j]:.3f}\n({count})"
            else:
                text = str(count)
            if display.shape == (2, 2):
                cell_name = [["TN", "FP"], ["FN", "TP"]][i][j]
                annotations[i, j] = f"{cell_name}\n{text}"
            else:
                annotations[i, j] = text

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        display,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close()
    return destination


class GradCAM:
    """Gradient-weighted Class Activation Mapping helper."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Grad-CAM heatmaps and return (cams, logits)."""
        self.model.eval()
        logits = self.model(input_tensor)

        if target_class is None:
            target_indices = logits.argmax(dim=1)
        else:
            target_indices = torch.full(
                (input_tensor.size(0),),
                fill_value=target_class,
                device=input_tensor.device,
                dtype=torch.long,
            )

        selected = logits.gather(1, target_indices.unsqueeze(1)).sum()
        self.model.zero_grad(set_to_none=True)
        selected.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = (weights * self.activations).sum(dim=1, keepdim=True)
        cams = torch.relu(cams)

        flat = cams.view(cams.size(0), -1)
        mins = flat.min(dim=1)[0].view(-1, 1, 1, 1)
        maxs = flat.max(dim=1)[0].view(-1, 1, 1, 1)
        cams = (cams - mins) / (maxs - mins + 1e-8)
        return cams, logits.detach()

    def remove_hooks(self) -> None:
        """Remove registered hooks to avoid leaking references."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def _resolve_gradcam_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    """Resolve a suitable convolutional layer for Grad-CAM."""
    if model_name == "custom_cnn":
        return model.block4.pointwise  # type: ignore[attr-defined]

    if model_name == "densenet121":
        try:
            return model.features.denseblock4.denselayer16.conv2  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise ValueError("Could not resolve DenseNet121 Grad-CAM target layer.") from exc

    raise ValueError(f"Unsupported model for Grad-CAM: '{model_name}'")


def _denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to [0,1] HWC numpy image."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    denorm = image * std + mean
    denorm = denorm.clamp(0.0, 1.0)
    return denorm.permute(1, 2, 0).cpu().numpy()


def _overlay_heatmap(image_hwc: np.ndarray, cam_hw: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a CAM heatmap on top of an image."""
    cmap = plt.get_cmap("jet")
    colored = cmap(cam_hw)[..., :3]
    overlay = (1 - alpha) * image_hwc + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


@torch.no_grad()
def _predict_labels(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    return preds, confs


def generate_gradcam_examples(
    model_name: str,
    data_dir: str | Path = DATA_DIR,
    checkpoint_path: str | None = None,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    num_examples: int = 6,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    output_dir: str | Path = RESULTS_DIR / "gradcam",
    target_class: int | None = None,
) -> list[Path]:
    """Generate Grad-CAM overlays from test samples and save them to disk."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")

    device = get_device()
    model = build_model(model_name).to(device)
    ckpt = resolve_checkpoint_path(model_name, checkpoint_path, checkpoint_dir)
    load_checkpoint(model, ckpt, device)

    _train_loader, _val_loader, test_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    class_names = list(test_loader.dataset.classes)

    target_layer = _resolve_gradcam_target_layer(model_name, model)
    gradcam = GradCAM(model, target_layer)

    output_root = _ensure_dir(output_dir)
    saved_paths: list[Path] = []
    saved_count = 0

    try:
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cams, logits = gradcam.generate(images, target_class=target_class)
            preds, confs = _predict_labels(logits)

            cams_resized = torch.nn.functional.interpolate(
                cams,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            for idx in range(images.size(0)):
                original = _denormalize_image(images[idx])
                heatmap = cams_resized[idx, 0].detach().cpu().numpy()
                overlay = _overlay_heatmap(original, heatmap, alpha=0.4)

                true_label = class_names[int(labels[idx].item())]
                pred_label = class_names[int(preds[idx].item())]
                confidence = float(confs[idx].item())

                fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
                axes[0].imshow(original)
                axes[0].set_title("Original X-ray")
                axes[0].axis("off")

                axes[1].imshow(heatmap, cmap="jet")
                axes[1].set_title("Grad-CAM Heatmap")
                axes[1].axis("off")

                axes[2].imshow(overlay)
                axes[2].set_title(
                    f"Overlay\nTrue: {true_label} | Pred: {pred_label} ({confidence:.2f})"
                )
                axes[2].axis("off")

                fig.tight_layout()
                file_path = output_root / f"{model_name}_gradcam_{saved_count:03d}.png"
                fig.savefig(file_path, dpi=220, bbox_inches="tight")
                plt.close(fig)

                saved_paths.append(file_path)
                saved_count += 1
                if saved_count >= num_examples:
                    return saved_paths
    finally:
        gradcam.remove_hooks()

    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot logs, confusion matrices, and Grad-CAM visualizations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_parser = subparsers.add_parser("plot-logs", help="Plot training CSV logs")
    plot_parser.add_argument(
        "--logs",
        nargs="+",
        default=[str(LOG_DIR / "custom_cnn.csv"), str(LOG_DIR / "densenet121.csv")],
    )
    plot_parser.add_argument("--output-dir", default=str(RESULTS_DIR / "plots"))

    cm_parser = subparsers.add_parser(
        "confusion-matrix", help="Plot confusion matrix from metrics JSON"
    )
    cm_parser.add_argument("--metrics-json", required=True)
    cm_parser.add_argument("--output-path", default=None)
    cm_parser.add_argument("--normalize", action="store_true")

    cam_parser = subparsers.add_parser("gradcam", help="Generate Grad-CAM overlays")
    cam_parser.add_argument("--model", required=True, choices=["custom_cnn", "densenet121"])
    cam_parser.add_argument("--data-dir", default=str(DATA_DIR))
    cam_parser.add_argument("--checkpoint", default=None)
    cam_parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    cam_parser.add_argument("--num-examples", type=int, default=6)
    cam_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    cam_parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    cam_parser.add_argument("--output-dir", default=str(RESULTS_DIR / "gradcam"))
    cam_parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional class index to explain (default: predicted class per image).",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    sns.set_theme(style="whitegrid")
    args = _parse_args()

    if args.command == "plot-logs":
        outputs = plot_training_curves(log_paths=args.logs, output_dir=args.output_dir)
        print("[interpret] generated plot files:")
        for path in outputs:
            print(f"  - {path}")
        return

    if args.command == "confusion-matrix":
        out = plot_confusion_matrix_from_metrics_json(
            metrics_json_path=args.metrics_json,
            output_path=args.output_path,
            normalize=args.normalize,
        )
        print(f"[interpret] saved confusion matrix -> {out}")
        return

    if args.command == "gradcam":
        outputs = generate_gradcam_examples(
            model_name=args.model,
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            target_class=args.target_class,
        )
        if not outputs:
            print("[interpret] No Grad-CAM images were generated.")
        else:
            print("[interpret] generated Grad-CAM files:")
            for path in outputs:
                print(f"  - {path}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
