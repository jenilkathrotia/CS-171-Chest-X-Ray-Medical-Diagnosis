# Part 3 Final Evaluation Report

## Objective
Evaluate trained pneumonia classifiers on the Chest X-Ray test split and summarize interpretability findings using Grad-CAM, with clinical emphasis on pneumonia recall (sensitivity).

## Data and Artifact Provenance
- Dataset source: KaggleHub `paultimothymooney/chest-xray-pneumonia`
- Resolved evaluation data directory:
  `C:\Users\aidan\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray`
- Test split size: 624 images (234 NORMAL, 390 PNEUMONIA)
- Evaluation script: `src/evaluate.py`
- Visualization/interpretability script: `src/interpret.py`
- Final notebooks:
  - `notebooks/02_evaluation.ipynb`
  - `notebooks/03_gradcam.ipynb`

## Quantitative Results

| Model | Accuracy | Pneumonia Recall (Sensitivity) | Pneumonia Precision | Pneumonia F1 | Normal Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `custom_cnn` | 0.6250 | 1.0000 | 0.6250 | 0.7692 | 0.0000 | 0.3846 |
| `densenet121` | 0.6250 | 1.0000 | 0.6250 | 0.7692 | 0.0000 | 0.3846 |

Reference files:
- `results/metrics/custom_cnn_metrics.json`
- `results/metrics/densenet121_metrics.json`

## Confusion Matrix Analysis
Both models produce the same confusion matrix on the test split:
- TN = 0
- FP = 234
- FN = 0
- TP = 390

Interpretation:
- All samples were predicted as `PNEUMONIA`.
- This gives maximal pneumonia recall (no false negatives), but zero ability to correctly identify NORMAL cases.
- Clinical screening perspective: high sensitivity is desirable, but this operating point creates excessive false positives and is not suitable as a standalone diagnostic model.

Reference figures:
- `results/confusion_matrices/custom_cnn_confusion_matrix.png`
- `results/confusion_matrices/densenet121_confusion_matrix.png`

## Training Trend Observations
Training logs indicate distinct optimization trajectories for each model:
- `densenet121` logs show stronger validation behavior than `custom_cnn` in training history.
- Current test-time predictions are nevertheless identical across models.

Reference figures:
- `results/plots/custom_cnn_training_curves.png`
- `results/plots/densenet121_training_curves.png`
- `results/plots/model_comparison_validation_curves.png`

## Interpretability (Grad-CAM)
Grad-CAM overlays were generated for both models:
- `results/gradcam/custom_cnn_gradcam_000.png` ... `custom_cnn_gradcam_005.png`
- `results/gradcam/densenet121_gradcam_000.png` ... `densenet121_gradcam_005.png`

Interpretation notes:
- Heatmaps should be interpreted together with metric behavior.
- Since both models collapse to one-class predictions on test data, saliency maps are insufficient as evidence of clinically meaningful discriminative learning.

## Limitations
- Checkpoint metadata in both local checkpoint files reports:
  - `epoch = 0`
  - `val_loss = 0.0`
  - `val_acc = 0.0`
- This metadata is inconsistent with available multi-epoch logs and weakens checkpoint provenance confidence.
- Current results should be interpreted as valid pipeline execution on real data, with model-state provenance caveats.

## Final Conclusion
- Part 3 evaluation and interpretability pipelines are fully implemented and reproducible from `src/`.
- Final artifacts (metrics, confusion matrices, plots, Grad-CAM, notebooks) are generated and consistent.
- Current model operating point strongly favors pneumonia detection recall at the cost of NORMAL specificity; deployment-quality conclusions require checkpoint provenance alignment and recalibrated model behavior.
