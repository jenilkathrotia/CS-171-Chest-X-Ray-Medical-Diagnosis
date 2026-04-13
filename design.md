# Design Document — Deep Learning for Pediatric Pneumonia Detection from Chest X-Ray Images

**Course:** CS-171
**Author:** Jenil Kathrotia
**Status:** Proposal

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Data Description](#3-data-description)
4. [Methodology](#4-methodology)
5. [Innovation and Objectives](#5-innovation-and-objectives)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Related Work](#7-related-work)
8. [Timeline](#8-timeline)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction

Pneumonia is a serious lung infection that can become life-threatening if it is not detected early. In medical settings, chest X-ray analysis is one of the most common ways to support pneumonia diagnosis, but manual interpretation can be time-consuming and difficult, especially when many patients must be screened quickly. This project proposes a neural-network-based system that classifies pediatric chest X-ray images as either **Normal** or **Pneumonia** using a publicly available Kaggle dataset containing about 5,863 images. The dataset used is *Chest X-Ray Images (Pneumonia)* on Kaggle.

**Dataset URL:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 2. Problem Statement

The goal of this project is to build a neural network that can automatically detect pneumonia from chest X-ray images. This problem is important because delayed or incorrect diagnosis can affect treatment decisions and patient outcomes. A reliable model could support doctors by serving as a screening tool, helping identify suspicious cases faster and reducing the chance that a pneumonia case is missed.

> ⚠️ In this application, **false negatives** are especially dangerous because they may leave sick patients untreated.

---

## 3. Data Description

The selected dataset contains **5,863 JPEG chest X-ray images** organized into `train`, `test`, and `validation` folders, with images labeled into two categories: **Normal** and **Pneumonia**. The chest radiographs are anterior-posterior images collected from pediatric patients aged 1 to 5 years at Guangzhou Women and Children's Medical Center. The dataset description states that the images were screened for quality control before use.

The main input feature is the chest X-ray image itself, while the target label is the diagnosis category. Since image sizes are not fully uniform, preprocessing will include:

- **Image resizing** to a uniform input size
- **Normalization** (ImageNet statistics for pretrained models)
- **Data augmentation** such as horizontal flipping and small affine transformations to improve generalization

Because the dataset may be imbalanced, the project will use **class weighting** during training and compare it against simple **oversampling**. Torchvision supports common augmentation pipelines, and PyTorch's `CrossEntropyLoss` supports class weights for imbalanced classification.

---

## 4. Methodology

This project will use **two model approaches**.

### 4.1 Custom Lightweight CNN

The first will be a custom deep CNN built with **spatially separable convolutions** to reduce the number of parameters while still learning useful visual patterns from X-ray images. A possible architecture is:

```
Input image
  → Convolution block
  → Depthwise-separable conv blocks (BatchNorm + ReLU)
  → Max-pooling layers
  → Global average pooling
  → Dropout
  → Fully connected classification layer
```

This design is lightweight and suitable for comparing efficiency and accuracy.

### 4.2 DenseNet121 (Transfer Learning)

The second approach will use **DenseNet121** as the pre-trained model. DenseNet121 is available in Torchvision with pretrained weights and is widely used in chest X-ray research. It is also the backbone architecture behind **CheXNet**, a well-known model for pneumonia detection from chest X-rays.

**Implementation stack:** PyTorch and Torchvision for data loading, preprocessing, transfer learning, and model evaluation.

---

## 5. Innovation and Objectives

The innovative part of this project is the **comparison between a lightweight custom CNN and a domain-relevant pre-trained DenseNet121** on the same medical imaging task. Instead of only reporting accuracy, the project will focus on **clinically meaningful evaluation**, especially reducing false negatives. Another useful addition will be simple model interpretability using **heatmap-based visualization**, so the model's decisions can be better understood.

### Main Objectives

1. Build a baseline custom CNN for pneumonia classification.
2. Fine-tune DenseNet121 using transfer learning.
3. Compare both models using performance and error analysis.
4. Identify which approach is more suitable for medical-image screening tasks.

---

## 6. Evaluation Metrics

The main evaluation metrics will be:

| Metric | Role |
| --- | --- |
| **Accuracy** | Overall measure of correct predictions — not sufficient alone for medical tasks. |
| **Recall** | Especially important; measures how many pneumonia cases are correctly identified. |
| **Precision** | Shows how many predicted pneumonia cases are actually correct. |
| **F1-score** | Balances precision and recall — strong metric when class imbalance exists. |
| **Confusion matrix** | Full view of true/false positives and negatives. |

Precision-recall tradeoffs will also be examined because missing a pneumonia case is more serious than generating an extra warning.

---

## 7. Related Work

Previous work by **Kermany et al.** showed that deep learning with transfer learning can accurately analyze pediatric chest X-ray images and even distinguish bacterial and viral pneumonia in the original research dataset. Another important related study is **CheXNet**, which used a 121-layer convolutional neural network based on DenseNet121 for chest X-ray diagnosis and reported very strong pneumonia detection performance on a large chest X-ray benchmark.

These studies show that deep learning is highly promising for medical imaging, but many approaches still face challenges such as dataset imbalance, limited interpretability, and the need for strong generalization across data sources. This project addresses these issues by comparing a lightweight custom model with a transfer-learning model and by emphasizing recall and interpretability.

---

## 8. Timeline

| Week | Goals |
| --- | --- |
| **Week 1** | Download dataset, inspect class distribution, preprocess images, and prepare train/validation/test loaders. |
| **Week 2** | Build and train the custom CNN, tune basic hyperparameters, and record baseline results. |
| **Week 3** | Fine-tune DenseNet121 with transfer learning and compare results against the custom model. |
| **Week 4** | Evaluate models using recall, precision, F1-score, and confusion matrix; generate visualizations; write final report and prepare presentation. |

---

## 9. Conclusion

This project proposes a deep learning system for detecting pneumonia from pediatric chest X-ray images. By comparing a custom lightweight CNN with a pre-trained DenseNet121 model, the project aims to study both efficiency and predictive performance in a medically important classification task. The expected outcome is a model that can support pneumonia screening with strong recall and practical value for computer-aided diagnosis. Because pneumonia detection is a high-stakes application, this project has meaningful real-world relevance in healthcare and medical AI.

---

## 10. References

1. **Kaggle.** *Chest X-Ray Images (Pneumonia).* https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. **Kermany, D. S., et al.** *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018.
3. **Rajpurkar, P., et al.** *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225.
4. **PyTorch Torchvision Documentation.** DenseNet121 model builder and pretrained weights.
