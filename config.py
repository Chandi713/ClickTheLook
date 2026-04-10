import os
import torch
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.abspath(os.getenv("DATA_ROOT"))

LOCAL_PATHS = {
    "train_csv":    os.path.join(DATA_ROOT, "img_info_dataframes", "train.csv"),
    "val_csv":      os.path.join(DATA_ROOT, "img_info_dataframes", "validation.csv"),
    "train_images": os.path.join(DATA_ROOT, "deepfashion2_original_images", "train", "image"),
    "val_images":   os.path.join(DATA_ROOT, "deepfashion2_original_images", "validation", "image"),
}

YOLO_DATASET_DIR    = os.path.abspath(os.getenv("YOLO_DATASET_DIR",    "./data/yolo"))
TRAINING_OUTPUT_DIR = os.path.abspath(os.getenv("TRAINING_OUTPUT_DIR", "./artifacts/runs"))
WEIGHTS_DIR         = os.path.abspath(os.getenv("WEIGHTS_DIR",         "./artifacts/weights"))
EXPORTS_DIR         = os.path.abspath(os.getenv("EXPORTS_DIR",         "./artifacts/exports"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT",   "ClickTheLook")

INFERENCE_MODEL_PATH = os.path.abspath(os.getenv("INFERENCE_MODEL_PATH", os.path.join(WEIGHTS_DIR, "best.pt")))

# ---------------------------------------------------------------------------
# Data-loading parameters
# ---------------------------------------------------------------------------
CSV_BATCH_SIZE       = 20000
IMAGE_DOWNLOAD_BATCH = 200       # kept for chunk progress; symlinks make it instant
MIN_DISK_FREE_GB     = 5.0
TEST_SPLIT_RATIO     = 0.10      # fraction of training images held out as final test set

# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
CATEGORIES = {
    1: "short_sleeve_top", 2: "long_sleeve_top",
    3: "short_sleeve_outwear", 4: "long_sleeve_outwear",
    5: "vest", 6: "sling", 7: "shorts", 8: "trousers",
    9: "skirt", 10: "short_sleeve_dress", 11: "long_sleeve_dress",
    12: "vest_dress", 13: "sling_dress"
}
NUM_CLASSES = len(CATEGORIES)
class_names = [CATEGORIES[i + 1] for i in range(NUM_CLASSES)]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    "model":            "yolo11n.pt",
    "epochs":           1,             # keep original, can reduce for quick tests
    "batch":            64,             # increase to better utilize GPU memory
    "imgsz":            512,            # smaller images → faster training
    "patience":         10,
    # "freeze":           10,             # keep backbone frozen
    "optimizer":        "AdamW",
    "lr0":              0.01,
    "lrf":              0.1,
    "momentum":         0.937,
    "weight_decay":     0.0005,
    "warmup_epochs":    1.0,
    "warmup_momentum":  0.8,
    "warmup_bias_lr":   0.1,
    "cos_lr":           True,
    "label_smoothing":  0.1,

    # -- Data Augmentation --
    "hsv_h":            0.015,
    "hsv_s":            0.2,            # reduce saturation shift
    "hsv_v":            0.2,            # reduce brightness shift
    "degrees":          5.0,            # smaller rotation
    "translate":        0.05,           # smaller translation
    "scale":            0.2,            # smaller scale changes
    "shear":            0.0,
    "perspective":      0.0,
    "flipud":           0.0,
    "fliplr":           0.5,
    "mosaic":           0.5,            # partial mosaic
    "mixup":            0.0,
    "amp":              True,           # keep Automatic Mixed Precision

    # -- Performance / Hardware --
    # "rect":             True,
    "device":           DEVICE,
    "workers":          8,              # more dataloader workers
    # "cache":            "disk",         # cache dataset in RAM for speed
    "cache":            False,

    # -- Output & logging --
    "project":          TRAINING_OUTPUT_DIR,
    "name":             "yolo11_deepfashion2",
    "exist_ok":         True,
    "pretrained":       True,
    "verbose":          True,
    "val":              False,          # skip validation for faster epoch iteration
}

if torch.cuda.device_count() > 1:
    TRAINING_CONFIG["device"] = list(range(torch.cuda.device_count()))
    TRAINING_CONFIG["batch"] *= torch.cuda.device_count()

# ---------------------------------------------------------------------------
# Directory initialisation
# ---------------------------------------------------------------------------
for _d in [YOLO_DATASET_DIR, TRAINING_OUTPUT_DIR, WEIGHTS_DIR, EXPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)
