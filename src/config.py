"""Centralized configuration for the Chest X-Ray Pneumonia Diagnosis project.

All file paths, hyperparameter defaults, and environment flags live here.
No other module should contain hardcoded paths.
"""

import os
import platform
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

IN_COLAB: bool = "COLAB_GPU" in os.environ or Path("/content").exists()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if IN_COLAB:
    DATA_DIR = Path(
        os.environ.get(
            "DATA_DIR",
            "/root/.cache/kagglehub/datasets/paultimothymooney/"
            "chest-xray-pneumonia/versions/2/chest_xray",
        )
    )
else:
    DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

GRADCAM_DIR = RESULTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 2
CLASS_NAMES: list[str] = ["NORMAL", "PNEUMONIA"]

# ---------------------------------------------------------------------------
# Image / DataLoader defaults
# ---------------------------------------------------------------------------

IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32

# Windows spawns workers via 'spawn' which is slow; default to 0 there.
NUM_WORKERS: int = 0 if platform.system() == "Windows" else 2

# ImageNet normalization (required for DenseNet121 pretrained weights)
IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 20
DROPOUT: float = 0.5

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
