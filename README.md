# Deep Learning for Pediatric Pneumonia Detection from Chest X-Ray Images

> CS-171 course project — comparing a lightweight custom CNN against a fine-tuned DenseNet121 for pediatric pneumonia classification from chest radiographs.

---

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Innovation and Objectives](#innovation-and-objectives)
- [Evaluation Metrics](#evaluation-metrics)
- [Related Work](#related-work)
- [Timeline](#timeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

Pneumonia is a serious lung infection that can become life-threatening if it is not detected early. In medical settings, chest X-ray analysis is one of the most common ways to support pneumonia diagnosis, but manual interpretation can be time-consuming and difficult, especially when many patients must be screened quickly. This project proposes a neural-network-based system that classifies pediatric chest X-ray images as either **Normal** or **Pneumonia** using a publicly available Kaggle dataset containing about 5,863 images.

**Dataset:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## Problem Statement

The goal of this project is to build a neural network that can automatically detect pneumonia from chest X-ray images. This problem is important because delayed or incorrect diagnosis can affect treatment decisions and patient outcomes. A reliable model could support doctors by serving as a screening tool, helping identify suspicious cases faster and reducing the chance that a pneumonia case is missed.

> ⚠️ In this application, **false negatives** are especially dangerous because they may leave sick patients untreated. Recall is therefore prioritized during evaluation.

---

## Dataset

The selected dataset contains **5,863 JPEG chest X-ray images** organized into `train`, `test`, and `val` folders, labeled into two categories: **Normal** and **Pneumonia**. The chest radiographs are anterior-posterior images collected from pediatric patients aged 1–5 years at Guangzhou Women and Children's Medical Center, and were screened for quality control before use.

**Input feature:** chest X-ray image
**Target label:** diagnosis category (Normal / Pneumonia)

### Preprocessing

- Image resizing to a fixed input size
- Normalization (ImageNet statistics for pretrained models)
- Data augmentation: horizontal flipping and small affine transformations to improve generalization
- Class imbalance handling via `CrossEntropyLoss` class weights, compared against simple oversampling

Torchvision supports common augmentation pipelines, and PyTorch's `CrossEntropyLoss` supports class weights for imbalanced classification.

---

## Methodology

This project compares **two model approaches**:

### 1. Custom Lightweight CNN

A deep CNN built with **spatially separable convolutions** to reduce the number of parameters while still learning useful visual patterns. Proposed architecture:

```
Input image
  → Conv block
  → Depthwise-separable conv blocks (BatchNorm + ReLU)
  → Max-pooling
  → Global average pooling
  → Dropout
  → Fully connected classification head
```

This design is lightweight and serves as the efficiency baseline.

### 2. DenseNet121 (Transfer Learning)

**DenseNet121** with pretrained ImageNet weights from Torchvision. DenseNet121 is widely used in chest X-ray research and is the backbone behind **CheXNet**, a well-known model for pneumonia detection from chest X-rays.

**Implementation stack:** PyTorch + Torchvision for data loading, preprocessing, transfer learning, and evaluation.

---

## Innovation and Objectives

The innovative part of this project is the **comparison between a lightweight custom CNN and a domain-relevant pre-trained DenseNet121** on the same medical imaging task. Instead of only reporting accuracy, the project focuses on clinically meaningful evaluation — especially reducing false negatives — and adds simple **heatmap-based interpretability** so model decisions can be better understood.

**Objectives:**

1. Build a baseline custom CNN for pneumonia classification.
2. Fine-tune DenseNet121 using transfer learning.
3. Compare both models using performance and error analysis.
4. Identify which approach is more suitable for medical-image screening tasks.

---

## Evaluation Metrics

| Metric | Why it matters |
| --- | --- |
| **Accuracy** | Overall measure of correct predictions (not sufficient alone for medical tasks). |
| **Recall** | Fraction of pneumonia cases correctly identified — the most important metric here. |
| **Precision** | Fraction of predicted pneumonia cases that are truly positive. |
| **F1-score** | Balances precision and recall under class imbalance. |
| **Confusion matrix** | Full picture of true/false positives and negatives. |

Precision-recall tradeoffs will also be examined, because missing a pneumonia case is more serious than generating an extra warning.

---

## Related Work

- **Kermany et al. (2018)** showed that deep learning with transfer learning can accurately analyze pediatric chest X-ray images and even distinguish bacterial vs. viral pneumonia.
- **CheXNet (Rajpurkar et al., 2017)** used a 121-layer CNN based on DenseNet121 for chest X-ray diagnosis and reported very strong pneumonia detection performance on a large benchmark.

These studies show deep learning is highly promising for medical imaging, but many approaches still face challenges such as dataset imbalance, limited interpretability, and generalization across data sources. This project addresses those issues by comparing a lightweight custom model against a transfer-learning model and by emphasizing recall and interpretability.

---

## Timeline

| Week | Goals |
| --- | --- |
| **1** | Download dataset, inspect class distribution, preprocess images, prepare train/val/test loaders. |
| **2** | Build and train the custom CNN, tune basic hyperparameters, record baseline results. |
| **3** | Fine-tune DenseNet121 with transfer learning and compare against the custom model. |
| **4** | Evaluate with recall / precision / F1 / confusion matrix, generate visualizations, write final report, prepare presentation. |

---

## Project Structure

```
CS-171-Chest-X-Ray-Medical-Diagnosis/
├── data/                  # Dataset (not tracked; see Getting Started)
│   ├── train/
│   ├── val/
│   └── test/
├── notebooks/             # Exploratory analysis and experiments
├── src/
│   ├── datasets.py        # Data loaders and augmentation
│   ├── models/
│   │   ├── custom_cnn.py  # Lightweight separable-conv CNN
│   │   └── densenet.py    # DenseNet121 transfer-learning model
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Metrics + confusion matrix
│   └── interpret.py       # Grad-CAM / heatmap visualization
├── results/               # Saved metrics, plots, checkpoints
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jenilkathrotia/CS-171-Chest-X-Ray-Medical-Diagnosis.git
cd CS-171-Chest-X-Ray-Medical-Diagnosis
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract into `data/` so the structure matches `data/train`, `data/val`, `data/test`.

### 4. Train a model

```bash
python src/train.py --model custom_cnn
python src/train.py --model densenet121 --pretrained
```

### 5. Evaluate

```bash
python src/evaluate.py --checkpoint results/densenet121_best.pt
```

---

## Conclusion

This project proposes a deep learning system for detecting pneumonia from pediatric chest X-ray images. By comparing a custom lightweight CNN with a pre-trained DenseNet121 model, the project studies both efficiency and predictive performance in a medically important classification task. The expected outcome is a model that can support pneumonia screening with strong recall and practical value for computer-aided diagnosis. Because pneumonia detection is a high-stakes application, this project has meaningful real-world relevance in healthcare and medical AI.

---

## References

1. **Kaggle.** *Chest X-Ray Images (Pneumonia).* https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. **Kermany, D. S., et al.** *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018.
3. **Rajpurkar, P., et al.** *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225.
4. **PyTorch Torchvision Documentation.** DenseNet121 model builder and pretrained weights.
