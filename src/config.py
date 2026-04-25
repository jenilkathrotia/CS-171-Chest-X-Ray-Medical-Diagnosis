from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# training config
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

# normalization (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]