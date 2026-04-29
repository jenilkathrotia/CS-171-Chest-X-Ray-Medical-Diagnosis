# Development History

Chronological log of all changes made during the CS 171 Chest X-Ray Pneumonia
Diagnosis project. Each session appends a new date-headed section.

---

## [2026-04-15] Project Initialization

- Created `README.md` with project overview, methodology, timeline, and structure
- Created `.gitignore` (data/, checkpoints, .env, venv, pycache)
- Created `INTERNAL_PROJECT_BRAIN.md` — internal architecture/planning doc (gitignored)
- Updated `.gitignore` to exclude `INTERNAL_PROJECT_BRAIN.md`

## [2026-04-15] Project Scaffolding

### Directory structure
- Created `src/` package with `__init__.py`
- Created `src/models/` package with `__init__.py` (re-exports `CustomCNN`, `build_densenet121`)
- Created `notebooks/` with `.gitkeep`
- Created `results/` with `.gitkeep`

### Dependencies
- Created `requirements.txt` — torch, torchvision, numpy, matplotlib, seaborn, scikit-learn, kagglehub, Pillow

### Core modules (fully implemented)
- `src/config.py` — environment detection (local vs Colab), centralized paths (`DATA_DIR`, `RESULTS_DIR`, `CHECKPOINT_DIR`, `GRADCAM_DIR`), dataset constants, ImageNet normalization stats, device selection, training hyperparameter defaults
- `src/datasets.py` — `get_transforms()` for train/eval augmentation pipelines, `get_dataloaders()` wrapping `ImageFolder` for train/val/test splits, `compute_class_weights()` with inverse-frequency formula
- `src/models/custom_cnn.py` — `SeparableConv2d` (depthwise + pointwise + BN + ReLU), `CustomCNN` (4-stage separable-conv network with GAP, dropout, FC head)
- `src/models/densenet.py` — `build_densenet121()` factory loading pretrained ImageNet weights with replaced classifier head

### Skeleton modules (interfaces defined, logic ready for Weeks 2-4)
- `src/train.py` — `train_one_epoch()`, `validate()`, `run_training()` with checkpoint saving and epoch logging
- `src/evaluate.py` — `compute_metrics()` via sklearn, `plot_confusion_matrix()` via seaborn heatmap
- `src/interpret.py` — `GradCAM` class with forward/backward hooks, `overlay_gradcam()` for heatmap visualization

### Documentation
- Created `DEVELOPMENT_HISTORY.md` (this file)

## [2026-04-15] EDA Notebook

- Created `notebooks/01_eda.ipynb` — 7-section exploratory analysis notebook
  - Section 0: Environment setup, Colab detection, kagglehub download guard
  - Section 1: Directory layout audit with per-class file counts
  - Section 2: Class distribution bar chart saved to `results/class_distribution.png`
  - Section 3: Sample image grid (raw, 4 per class) from the train split
  - Section 4: Image size + channel mode audit with width/height histograms
  - Section 5: DataLoader end-to-end validation (shapes, label values, pixel range, augmented batch grid)
  - Section 6: Class weights computation and usage printout for `CrossEntropyLoss`
  - Section 7: EDA summary table and pipeline readiness checklist

## [2026-04-15] Training Readiness

### Bug fixes
- `src/models/custom_cnn.py` — changed `CustomCNN` default `in_channels` from 1 to 3; ImageFolder outputs 3-channel RGB tensors even for grayscale source images, so the model must match
- `notebooks/01_eda.ipynb` — replaced fragile `DATA_DIR.parent.parent / "results"` save path with `RESULTS_DIR` from config (works on both local and Colab)

### New functionality
- `src/evaluate.py` — added `collect_predictions(model, loader, device)` function that runs a model over a DataLoader and returns `(y_true, y_pred)` lists, bridging the gap between `run_training()` and `compute_metrics()`

### Training notebook
- Created `notebooks/02_train.ipynb` — full training and evaluation pipeline
  - Section 0-1: Environment setup, data loading, class weight computation
  - Section 2: Smoke test (1 batch forward pass through both models, shape assertion)
  - Section 3: Custom CNN training with class-weighted CrossEntropyLoss, Adam, StepLR
  - Section 4: Training curves plot (loss + accuracy) for Custom CNN
  - Section 5: Custom CNN test-set evaluation (recall, F1, precision, confusion matrix)
  - Section 6: DenseNet121 Phase 1 (frozen backbone, 5 epochs) + Phase 2 (full fine-tune, 15 epochs)
  - Section 7: DenseNet121 training curves (combined phases)
  - Section 8: DenseNet121 test-set evaluation
  - Section 9: Side-by-side metric comparison table with best-model indicators

## [2026-04-15] README Rewrite

- Rewrote `README.md` to match actual codebase (replaced placeholder CLI commands with notebook-based workflow)
- Added accurate project structure tree with all current files
- Added complete local setup guide (venv creation, Windows + macOS/Linux, dataset download, notebook execution order)
- Added Google Colab section explaining auto-detection and kagglehub integration
- Updated methodology to reflect actual 3-channel CustomCNN architecture and DenseNet121 two-phase fine-tuning
- Added dataset statistics table with per-split class counts
- Added design principles section (logic in src/, centralized paths, pure PyTorch)

## [2026-04-27] Part 3 Kickoff - State Synchronization

### Documentation synchronization
- Updated `INTERNAL_PROJECT_BRAIN.md` to reflect current implemented architecture and training flow.
- Corrected stale assumptions (e.g., grayscale/1-channel pipeline references) to match actual 3-channel ImageNet-normalized preprocessing in `src/datasets.py`.
- Added a dedicated Part 3 implementation section covering `src/evaluate.py` and `src/interpret.py` deliverables.

### Milestone status confirmation
- **Part 1 complete (Aaron):**
  - `src/datasets.py` implements transforms, train/val/test dataloaders, class weight utilities.
- **Part 2 complete (Jenil):**
  - `src/models/custom_cnn.py` and `src/models/densenet.py` are implemented.
  - `src/train.py` provides full training loop, checkpointing, and metric logging.
  - Existing logs confirmed in `results/logs/custom_cnn.csv` and `results/logs/densenet121.csv`.

### Part 3 planned implementation scope
- Build `src/evaluate.py` for reproducible test-set inference and sklearn reporting (accuracy, precision, recall, F1), including explicit pneumonia recall/sensitivity.
- Build `src/interpret.py` for:
  - training curve visualizations from CSV logs,
  - labeled confusion matrix generation,
  - Grad-CAM overlays for model interpretability on chest X-rays.
- Ensure all paths/configuration are dynamic via `src/config.py` defaults and CLI overrides.

### Expected artifacts from Part 3
- `results/metrics/*.json` for evaluation summaries.
- `results/plots/*` for training/validation curves.
- `results/confusion_matrices/*` for model confusion matrices.
- `results/gradcam/*` for Grad-CAM visual explanations.

## [2026-04-28] Documentation Consolidation - Evidence Audit

### Notebook evidence verification
- Reviewed `notebooks/train_colab.ipynb` and confirmed:
  - Colab GPU workflow was executed (`nvidia-smi` output captured).
  - Dataset path resolved from kagglehub (`.../chest_xray` with `NORMAL` and `PNEUMONIA` classes).
  - Both model training runs were executed in notebook output:
    - `custom_cnn` for 15 epochs
    - `densenet121` for 10 epochs
  - Notebook output shows checkpoint/log copy operations to Drive path:
    - `/content/drive/MyDrive/cs171_chest_xray/checkpoints`
    - `/content/drive/MyDrive/cs171_chest_xray/logs`
- Reviewed `notebooks/01_eda.ipynb` and confirmed:
  - Dataset class counts reported in outputs (`train`, `val`, `test` split counts).
  - EDA and preprocessing sanity checks were run (sampling, distributions, loader checks).

### Current `results/` artifact inventory (local workspace)
- `results/logs/custom_cnn.csv` and `results/logs/densenet121.csv`:
  - Per-epoch training/validation metrics from training pipeline.
- `results/plots/*.png`:
  - Visualization artifacts generated from CSV logs (loss/accuracy curves and validation comparison).
- `results/metrics/*.json`, `results/confusion_matrices/*.png`, `results/gradcam/*.png`:
  - Evaluation/interpretability outputs generated by current Part 3 scripts.
- `results/smoke_data/`:
  - Small synthetic image dataset (2 images/class/split) used for smoke validation.

### Checkpoint metadata audit and interpretation note
- Local files `results/checkpoints/custom_cnn_best.pt` and `results/checkpoints/densenet121_best.pt` were inspected via `torch.load`.
- Both currently report metadata:
  - `epoch = 0`
  - `val_loss = 0.0`
  - `val_acc = 0.0`
- Therefore, current locally generated metrics/confusion/Grad-CAM outputs should be interpreted as **pipeline smoke-validation artifacts**, not final report-grade model performance results.

## [2026-04-29] Part 3 Final Execution Pass (Real Test Data)

### Input validation and environment setup
- Installed project dependencies from `requirements.txt` in the active Python environment.
- Downloaded Kaggle dataset via `kagglehub` and resolved real evaluation root:
  - `C:/Users/aidan/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray`
- Confirmed local hardware availability for this run:
  - `cuda=False`, `mps=False` (CPU evaluation run).

### Final evaluation artifacts regenerated
- Ran `src/evaluate.py` for both models against the real test split (624 images):
  - `results/metrics/custom_cnn_metrics.json`
  - `results/metrics/densenet121_metrics.json`
- Current test metrics for both models are identical in this checkpoint set:
  - Accuracy: `0.6250`
  - Pneumonia recall (sensitivity): `1.0000`
  - Normal recall: `0.0000`

### Visualization and interpretability artifacts regenerated
- Regenerated curve plots from training logs:
  - `results/plots/custom_cnn_training_curves.png`
  - `results/plots/densenet121_training_curves.png`
  - `results/plots/model_comparison_validation_curves.png`
- Regenerated confusion matrices from metrics JSON:
  - `results/confusion_matrices/custom_cnn_confusion_matrix.png`
  - `results/confusion_matrices/densenet121_confusion_matrix.png`
- Generated six Grad-CAM examples per model:
  - `results/gradcam/custom_cnn_gradcam_000.png` ... `_005.png`
  - `results/gradcam/densenet121_gradcam_000.png` ... `_005.png`

### Notebook and report deliverables completed
- Created final evaluation notebook:
  - `notebooks/02_evaluation.ipynb`
  - executed validation copy: `notebooks/02_evaluation.executed.ipynb`
- Created final interpretability notebook:
  - `notebooks/03_gradcam.ipynb`
  - executed validation copy: `notebooks/03_gradcam.executed.ipynb`
- Added written report deliverable:
  - `results/metrics.md`

### Consistency note
- This execution pass is full-data at evaluation time, but checkpoint metadata still reports:
  - `epoch=0`, `val_loss=0.0`, `val_acc=0.0`
- This provenance caveat is explicitly documented in project context files and report language.
