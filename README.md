# Deep Learning for Pediatric Pneumonia Detection from Chest X-Ray Images

> CS 171 Machine Learning course project at San Jose State University -- comparing a lightweight custom CNN against a fine-tuned DenseNet121 for binary pneumonia classification from pediatric chest radiographs.

**Team:** Aaron Dang, Jenil Kathrotia, Aidan Marra

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running on Google Colab](#running-on-google-colab)
- [Timeline](#timeline)
- [Related Work](#related-work)
- [References](#references)

---

## Problem Statement

Pneumonia is a leading cause of hospitalization and death, especially in young children. Chest X-rays are the standard screening tool, but manual interpretation is slow and error-prone under high patient volume. This project builds a deep learning system that classifies pediatric chest X-ray images as **Normal** or **Pneumonia**.

> In this application, **false negatives** (missed pneumonia) are far more dangerous than false positives. A sick child sent home untreated can face life-threatening complications. **Recall is therefore the primary evaluation metric.**

---

## Dataset

[Chest X-Ray Images (Pneumonia) -- Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **5,863 JPEG** anterior-posterior chest radiographs from pediatric patients aged 1-5 years
- Collected at Guangzhou Women and Children's Medical Center
- Two classes: **NORMAL** and **PNEUMONIA**
- Pre-split into `train/`, `val/`, and `test/` folders

| Split | NORMAL | PNEUMONIA | Total |
|-------|-------:|----------:|------:|
| Train |  1,341 |     3,875 | 5,216 |
| Val   |      8 |         8 |    16 |
| Test  |    234 |       390 |   624 |

The training set has a **~74/26 class imbalance** (Pneumonia-heavy), which is addressed via inverse-frequency class weighting in the loss function.

### Preprocessing Pipeline

All preprocessing is implemented in `src/datasets.py`:

- **Resize** to 256px, then **RandomResizedCrop** to 224px (train) or **CenterCrop** to 224px (eval)
- **RandomHorizontalFlip** and **RandomRotation(10)** for training augmentation
- **ImageNet normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for compatibility with pretrained DenseNet121 weights
- **Class weighting** via `CrossEntropyLoss(weight=...)` using the formula `w_c = total / (num_classes * count_c)`

---

## Methodology

### Model 1: Custom Lightweight CNN

A parameter-efficient network built with **depthwise-separable convolutions** (same core idea as MobileNet). This reduces parameters by ~8-9x per layer compared to standard convolutions, which is critical given the small dataset size.

```
Input (B, 3, 224, 224)
  -> Conv2d(3, 32) + BatchNorm + ReLU + MaxPool
  -> SeparableConv2d(32, 64)   + MaxPool
  -> SeparableConv2d(64, 128)  + MaxPool
  -> SeparableConv2d(128, 256) + MaxPool
  -> AdaptiveAvgPool2d(1)
  -> Dropout(0.5) -> Linear(256, 2)
```

Each `SeparableConv2d` block consists of a depthwise convolution (one filter per channel), a pointwise 1x1 convolution, BatchNorm, and ReLU.

### Model 2: DenseNet121 (Transfer Learning)

DenseNet121 with ImageNet-pretrained weights, widely used in chest X-ray research (it is the backbone behind CheXNet). The original 1000-class head is replaced with a 2-class linear layer. Training uses a two-phase strategy:

1. **Phase 1 (frozen):** Only the classifier head is trained for 5 epochs at LR=1e-3
2. **Phase 2 (unfrozen):** The full network is fine-tuned for 15 epochs at LR=1e-4

### Implementation Stack

- **PyTorch + Torchvision** for models, data loading, transforms, and training
- **scikit-learn** for classification metrics
- **seaborn / matplotlib** for confusion matrices and training curves
- **Grad-CAM** for model interpretability (heatmap visualization)

---

## Evaluation Metrics

| Priority | Metric | Why |
|----------|--------|-----|
| 1 | **Recall** | Missed pneumonia = untreated child. Minimize false negatives. |
| 2 | **F1-Score** | Balances recall and precision for fair model comparison. |
| 3 | **Precision** | Keeps false alarms manageable for clinical trust. |
| 4 | **Accuracy** | Reported for completeness; misleading alone under class imbalance. |

Both models are also evaluated with a **confusion matrix** and **Grad-CAM heatmaps** for interpretability.

---

## Project Structure

```
CS-171-Chest-X-Ray-Medical-Diagnosis/
├── data/                          # Dataset (not tracked; see Getting Started)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   └── 02_train.ipynb             # Training + evaluation + comparison
├── src/
│   ├── __init__.py
│   ├── config.py                  # Centralized paths, constants, device
│   ├── datasets.py                # Transforms, DataLoaders, class weights
│   ├── train.py                   # Training loop + checkpointing
│   ├── evaluate.py                # Metrics, confusion matrix, predictions
│   ├── interpret.py               # Grad-CAM heatmap visualization
│   └── models/
│       ├── __init__.py            # Re-exports CustomCNN, build_densenet121
│       ├── custom_cnn.py          # SeparableConv2d + CustomCNN
│       └── densenet.py            # DenseNet121 wrapper
├── results/                       # Generated plots, metrics, checkpoints
│   ├── checkpoints/               # Saved .pt model weights
│   └── gradcam/                   # Grad-CAM overlay images
├── requirements.txt
└── README.md
```

### Design Principles

- **All logic lives in `src/`** -- notebooks only call functions from `src/` modules
- **Centralized paths** in `src/config.py` -- no hardcoded paths anywhere else; auto-detects local vs Colab environment
- **Pure PyTorch** -- no high-level wrappers (FastAI, Lightning) per course requirements

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- A Kaggle account (to download the dataset)

### 1. Clone the repository

```bash
git clone https://github.com/jenilkathrotia/CS-171-Chest-X-Ray-Medical-Diagnosis.git
cd CS-171-Chest-X-Ray-Medical-Diagnosis
```

### 2. Create a virtual environment and install dependencies

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract so the folder structure matches:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The `data/` folder is gitignored and will not be committed.

### 4. Run the EDA notebook

Open `notebooks/01_eda.ipynb` in Jupyter, VS Code, or Cursor and run all cells. This will:

- Verify the dataset directory layout and file counts
- Display class distribution charts
- Show sample X-ray images
- Validate the DataLoader pipeline end-to-end
- Compute and print class weights

If all cells run without errors, the data pipeline is ready.

### 5. Train and evaluate models

Open `notebooks/02_train.ipynb` and run all cells. This will:

- Run a smoke test (1 batch through both models)
- Train the Custom CNN with class-weighted loss
- Plot training curves (loss + accuracy)
- Evaluate on the test set (recall, F1, precision, confusion matrix)
- Train DenseNet121 with two-phase fine-tuning
- Compare both models side-by-side

Best model checkpoints are saved to `results/checkpoints/`. Confusion matrix plots are saved to `results/`.

---

## Running on Google Colab

Both notebooks auto-detect the Colab environment and handle dataset download automatically. No manual data setup is needed.

1. Upload the repository to Google Drive or clone it in a Colab cell
2. Open a notebook (`.ipynb`) in Colab
3. Select a **GPU runtime** (Runtime > Change runtime type > T4 GPU)
4. Run all cells -- the first cell downloads the dataset via `kagglehub`

The `src/config.py` module switches `DATA_DIR` to the kagglehub cache path when it detects a Colab environment.

---

## Timeline

| Date | Milestone |
|------|-----------|
| Apr 15 | Project proposal submitted |
| Apr 15-20 | Data preprocessing, EDA, pipeline validation |
| Apr 20-27 | Custom CNN implementation and training |
| Apr 22 | Team progress discussion |
| Apr 27-May 4 | DenseNet121 fine-tuning and model comparison |
| May 4-11 | Full evaluation, Grad-CAM, final report, presentation |

---

## Related Work

- **Kermany et al. (2018)** demonstrated that transfer learning can accurately analyze pediatric chest X-rays and distinguish bacterial vs. viral pneumonia.
- **CheXNet (Rajpurkar et al., 2017)** used DenseNet121 for chest X-ray diagnosis and achieved radiologist-level pneumonia detection on the ChestX-ray14 benchmark.

This project addresses gaps in prior work by comparing a lightweight custom model against transfer learning, emphasizing recall over accuracy, and adding Grad-CAM interpretability.

---

## References

1. **Kaggle.** *Chest X-Ray Images (Pneumonia).* https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. **Kermany, D. S., et al.** *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018.
3. **Rajpurkar, P., et al.** *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225.
4. **PyTorch Torchvision Documentation.** DenseNet121 model builder and pretrained weights.
