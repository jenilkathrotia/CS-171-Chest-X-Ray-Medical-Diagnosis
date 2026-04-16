"""Grad-CAM interpretability for pneumonia classification models.

Registers forward/backward hooks on a target convolutional layer to compute
gradient-weighted class activation maps, then overlays them on the original
X-ray image.

Public API:
    GradCAM         -- hook-based Grad-CAM extractor
    overlay_gradcam -- blend heatmap onto an image and save
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GRADCAM_DIR


class GradCAM:
    """Grad-CAM extractor for a given model and target layer.

    Usage::

        cam = GradCAM(model, model.sep_block3[0])
        heatmap = cam(input_tensor, target_class=1)

    Args:
        model:        The neural network.
        target_layer: A convolutional ``nn.Module`` inside *model* whose
                      activations and gradients will be captured.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self, _module: nn.Module, _input: tuple, output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self, _module: nn.Module, _grad_input: tuple, grad_output: tuple
    ) -> None:
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(
        self, x: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single image.

        Args:
            x:            Input tensor of shape ``(1, C, H, W)``.
            target_class: Class index to explain. If ``None``, uses the
                          predicted class.

        Returns:
            Numpy array of shape ``(H, W)`` with values in ``[0, 1]``.
        """
        self.model.eval()
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam

    def remove_hooks(self) -> None:
        """Detach hooks from the model."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    save_path: str | Path | None = None,
) -> None:
    """Overlay a Grad-CAM heatmap on an image and save to disk.

    Args:
        image:     Original image as a numpy array ``(H, W)`` or ``(H, W, 3)``
                   with values in ``[0, 1]``.
        heatmap:   Grad-CAM heatmap ``(H, W)`` with values in ``[0, 1]``.
        alpha:     Heatmap opacity.
        save_path: Destination file. Defaults to ``results/gradcam/overlay.png``.
    """
    if save_path is None:
        save_path = GRADCAM_DIR / "overlay.png"

    fig, ax = plt.subplots(figsize=(6, 6))

    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
