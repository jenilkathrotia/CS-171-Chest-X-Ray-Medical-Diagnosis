# Project Brain — CS 171 Chest X-Ray Pneumonia Diagnosis

> Internal context document. **Never commit this file** (it is gitignored).
> Last updated: 2026-04-30

---

## A. Current System Snapshot

### Part Ownership and Status
- Part 1 (Aaron): complete in `src/datasets.py` with train/eval transforms, dataloaders,
  and class-weight helpers.
- Part 2 (Jenil): complete in `src/models/custom_cnn.py`, `src/models/densenet.py`,
  and `src/train.py` with checkpoint + CSV logging outputs.
- Part 3 (Aidan): evaluation + visualization + interpretability execution completed,
  including full test-set evaluation artifacts, notebooks, and report output.

### Verified Runtime Conventions
- Input pipeline is 3-channel RGB tensors for both models.
- Normalization is ImageNet-compatible (`IMAGENET_MEAN`, `IMAGENET_STD` from config).
- Training saves:
  - checkpoints at `results/checkpoints/{model_name}_best.pt`
  - logs at `results/logs/{model_name}.csv`
- Existing logs present:
  - `results/logs/custom_cnn.csv`
  - `results/logs/densenet121.csv`

---

## B. Implemented Architecture

### `src/models/custom_cnn.py`
- `SeparableConv2d`: depthwise conv + pointwise conv + BN + ReLU.
- `CustomCNN` stack:
  - Stem: `Conv2d(3 -> 32, stride=2)` + BN + ReLU
  - Blocks: `(32->64)`, `(64->128)`, `(128->256)` each with max-pool
  - Final block: `(256->512)`
  - Head: adaptive avg pool -> dropout (`DROPOUT`) -> `Linear(512, NUM_CLASSES)`
- Factory: `build_custom_cnn(num_classes=NUM_CLASSES)`.

### `src/models/densenet.py`
- `build_densenet121(num_classes=NUM_CLASSES, pretrained=True)` uses torchvision
  DenseNet121 with classifier replaced by `Linear(in_features, num_classes)`.

---

## C. Implemented Data Pipeline (`src/datasets.py`)

### Transforms
- Train transform:
  `Resize(256) -> RandomCrop(IMAGE_SIZE) -> RandomHorizontalFlip() -> RandomRotation(10) -> ToTensor() -> Normalize(ImageNet)`
- Eval transform:
  `Resize(256) -> CenterCrop(IMAGE_SIZE) -> ToTensor() -> Normalize(ImageNet)`

### Dataloaders
- `get_dataloaders(data_dir, batch_size=32, num_workers=2)` returns a tuple:
  `(train_loader, val_loader, test_loader)`.
- Uses `ImageFolder` with split directories:
  `train/`, `val/`, `test/`.
- `train_loader` is shuffled; `val_loader` and `test_loader` are deterministic.

### Imbalance Handling
- `compute_class_weights(train_dataset)` returns inverse-frequency class weights.
- `get_weighted_sampler(train_dataset)` is available for optional sampling strategy.

---

## D. Training System (`src/train.py`)

### Complete Features
- Model registry: `custom_cnn`, `densenet121`.
- Epoch loops:
  - `train_one_epoch(...)`
  - `validate(...)`
- Loss: class-weighted `CrossEntropyLoss`.
- Optimizer + scheduler: Adam + `ReduceLROnPlateau`.
- Structured logging to CSV with columns:
  `epoch, train_loss, train_acc, val_loss, val_acc, lr`.
- Best checkpoint persistence with metadata and `state_dict`.
- CLI entrypoint (`python -m src.train ...`) and config-driven defaults.

---

## E. Part 3 Implementation Status (Evaluation + Visualization + Interpretability)

### `src/evaluate.py` (implemented)
- Reusable CLI + importable functions are implemented.
- Dynamic checkpoint path resolution and robust file-format/path validation are implemented.
- Evaluation metrics implemented via sklearn:
  - accuracy, precision, recall, F1
  - full classification report
  - confusion matrix output
  - explicit pneumonia recall (sensitivity) extraction
- JSON output implemented at `results/metrics/{model_name}_metrics.json`.

### `src/interpret.py` (implemented)
- Training-log visualization implemented:
  - CSV parsing via pandas
  - train/val loss and train/val accuracy plots
  - multi-model validation comparison plot
- Confusion matrix rendering implemented:
  - labeled TN/FP/FN/TP cells for binary classification
  - output to `results/confusion_matrices/`
- Grad-CAM implemented:
  - model-specific target-layer resolution for `custom_cnn` and `densenet121`
  - heatmap generation and overlay on de-normalized tensors
  - output to `results/gradcam/`

### Reproducibility and module design
- Defaults are sourced from `src/config.py` with CLI overrides.
- No hardcoded absolute paths in module logic.
- Modules are usable from CLI and importable from notebooks.
- Evaluation path is deterministic (`model.eval()` + no-grad inference).

### Validated notebook evidence and artifact semantics
- `notebooks/train_colab.ipynb` contains executed outputs for:
  - Colab GPU training runs for both models,
  - checkpoint/log export actions to Drive,
  - epoch-by-epoch logs matching `results/logs/*.csv`.
- `notebooks/01_eda.ipynb` contains executed outputs for:
  - dataset split/class counts,
  - image sampling and distribution checks,
  - dataloader/preprocessing sanity checks.

### Results folder map (current local workspace semantics)
- `results/logs/`: training history traces from model training runs.
- `results/plots/`: training curve visualizations derived from log files.
- `results/checkpoints/`: trained checkpoints sourced from `origin/part-2`.
  Metadata is non-degenerate (`custom_cnn`: epoch=1, val_loss~0.882, val_acc=0.625;
  `densenet121`: epoch=6, val_loss~0.086, val_acc=0.9375).
- `results/smoke_data/`: synthetic tiny dataset for pipeline smoke tests.
- `results/metrics/`, `results/confusion_matrices/`, `results/gradcam/`:
  outputs regenerated end-to-end from the current `part-2` checkpoints.
- `results/metrics.md`: final written Part 3 report summary tied to generated artifacts.

### Latest full-data execution summary (2026-04-30)
- Checkpoint source: `origin/part-2` branch (pulled via `git checkout origin/part-2 -- results/checkpoints/*_best.pt results/logs/*.csv`).
- Real evaluation data source:
  `C:/Users/aidan/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray`
- Evaluated test samples: `624` (234 NORMAL, 390 PNEUMONIA).
- `custom_cnn` results:
  - accuracy `0.8333`
  - pneumonia recall (sensitivity) `0.9128`
  - normal recall `0.7009`
  - macro F1 `0.8159`
  - confusion matrix `TN=164, FP=70, FN=34, TP=356`
- `densenet121` results:
  - accuracy `0.8878`
  - pneumonia recall (sensitivity) `0.9897`
  - normal recall `0.7179`
  - macro F1 `0.8722`
  - confusion matrix `TN=168, FP=66, FN=4, TP=386`
- Regenerated outputs (all derived from rerun on these checkpoints):
  - `results/metrics/*.json`
  - `results/confusion_matrices/*.png`
  - `results/plots/*.png`
  - `results/gradcam/*_000..005.png`
- Completed Part 3 notebooks:
  - `notebooks/02_evaluation.ipynb`
  - `notebooks/03_gradcam.ipynb`

### How to interpret current outputs
- Metrics, confusion matrices, training curves, and Grad-CAM heatmaps are all
  derived from the same checkpoint+dataset run and are internally consistent.
- DenseNet121 is the recommended model (higher accuracy, much higher pneumonia
  sensitivity, higher NORMAL precision). Custom CNN is a reasonable lightweight
  baseline.

---

## F. Current Milestone Checklist

### Completed
- [x] Part 1: preprocessing/data pipeline
- [x] Part 2: model training + baseline logs
- [x] Part 3: `src/evaluate.py` implementation
- [x] Part 3: `src/interpret.py` implementation
- [x] Part 3: artifact-generation pipeline implementation
- [x] Part 3: full test-set evaluation and interpretability execution
- [x] Part 3: evaluation/interpretability notebooks completed
- [x] Part 3: written report artifact completed (`results/metrics.md`)

### Resolved Caveats
- [x] Checkpoint provenance: previously the local checkpoints showed
      `epoch=0, val_loss=0.0, val_acc=0.0`. Resolved on 2026-04-30 by pulling
      verified `*_best.pt` files from `origin/part-2`. New metadata is
      non-degenerate and consistent with `results/logs/*.csv`.

### Ongoing Limitations
- Single test split; no calibration / threshold sweep performed.
- Class imbalance partially mitigated via class-weighted loss.

---

## Quick Reference — File Map

| File                       | Status                        | Owner         | Purpose |
|----------------------------|-------------------------------|---------------|---------|
| `src/config.py`            | Implemented                   | Team          | Centralized paths and constants |
| `src/datasets.py`          | Implemented (Part 1 complete) | Aaron         | Data loading, transforms, weighting |
| `src/models/custom_cnn.py` | Implemented (Part 2 complete) | Jenil / Team  | Custom separable CNN |
| `src/models/densenet.py`   | Implemented (Part 2 complete) | Jenil / Team  | DenseNet121 transfer model |
| `src/train.py`             | Implemented (Part 2 complete) | Jenil / Team  | Training loop + checkpoints + CSV |
| `src/evaluate.py`          | Implemented (Part 3)          | Aidan         | Test metrics + JSON report output |
| `src/interpret.py`         | Implemented (Part 3)          | Aidan         | Curves, confusion matrices, Grad-CAM |
| `results/logs/*.csv`       | Available                     | Generated     | Training/validation history traces |
| `results/checkpoints/*.pt` | Verified (sourced from `part-2`) | Generated  | Trained model checkpoints with valid metadata |
| `results/smoke_data/`      | Present locally               | Generated     | Synthetic pipeline smoke-test dataset |
| `results/metrics/*.json`   | Generated                     | Generated     | Full test-set evaluation summaries for current checkpoint set |
| `results/confusion_matrices/*.png` | Generated            | Generated     | Confusion matrix visualizations |
| `results/gradcam/*.png`    | Generated                     | Generated     | Grad-CAM visualization outputs |
| `notebooks/02_evaluation.ipynb` | Implemented             | Aidan         | Final model comparison and metric narrative notebook |
| `notebooks/03_gradcam.ipynb` | Implemented               | Aidan         | Final interpretability review notebook |
| `results/metrics.md`       | Implemented                   | Aidan         | Final Part 3 written report |
