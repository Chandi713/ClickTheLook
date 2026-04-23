import os
import torch
from dotenv import load_dotenv

load_dotenv()

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

CSV_BATCH_SIZE       = 20000
IMAGE_DOWNLOAD_BATCH = 200       # chunk size for CSV streaming; symlinks make image resolution instant
MIN_DISK_FREE_GB     = 5.0
TEST_SPLIT_RATIO     = 0.10      # held-out fraction of training image filenames for the final test split

CATEGORIES = {
    1: "short_sleeve_top", 2: "long_sleeve_top",
    3: "short_sleeve_outwear", 4: "long_sleeve_outwear",
    5: "vest", 6: "sling", 7: "shorts", 8: "trousers",
    9: "skirt", 10: "short_sleeve_dress", 11: "long_sleeve_dress",
    12: "vest_dress", 13: "sling_dress"
}
NUM_CLASSES = len(CATEGORIES)
class_names = [CATEGORIES[i + 1] for i in range(NUM_CLASSES)]

if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

TRAINING_CONFIG = {
    "model":            "yolo8s.pt",
    "epochs":           1,
    "batch":            64,
    "imgsz":            512,
    "patience":         10,
    # "freeze":           10,
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

    "hsv_h":            0.015,
    "hsv_s":            0.2,
    "hsv_v":            0.2,
    "degrees":          5.0,
    "translate":        0.05,
    "scale":            0.2,
    "shear":            0.0,
    "perspective":      0.0,
    "flipud":           0.0,
    "fliplr":           0.5,
    "mosaic":           0.5,
    "mixup":            0.0,
    "amp":              True,

    # "rect":             True,
    "device":           DEVICE,
    "workers":          8,
    # "cache":            "disk",
    "cache":            False,

    "project":          TRAINING_OUTPUT_DIR,
    "name":             "yolo8s_deepfashion2",
    "exist_ok":         True,
    "pretrained":       True,
    "verbose":          True,
    "val":              False,          # Ultralytics val loop each epoch (slower)
}

if torch.cuda.device_count() > 1:
    TRAINING_CONFIG["device"] = list(range(torch.cuda.device_count()))
    TRAINING_CONFIG["batch"] *= torch.cuda.device_count()

# Swap tracker implementation: "deepsort" or "sort". DeepSort lives in src/live/deepsort_tracker.py.
TRACKER_BACKEND = os.getenv("TRACKER_BACKEND", "deepsort")

LIVE_CONFIG = {
    "conf":                0.5,
    "iou":                 0.45,
    "device":              DEVICE,
    "line_thickness":      2,

    "tracker_max_age":     5,
    "tracker_min_hits":    2,
    "tracker_iou_thresh":  0.2,

    "deepsort_max_age":              5,
    "deepsort_n_init":               3,
    "deepsort_max_cosine_distance":  0.3,
    "deepsort_embedder":             "mobilenet",

    "use_global_ids":          True,
    "gid_max_gap_s":           43.0,
    "gid_cosine_threshold":    0.15,
    "gid_spatial_gate_pps":    50.0,

    "dedup_blur_threshold":    90.0,   # Laplacian variance; below = discard as blurry
    "dedup_phash_threshold":   13,   # dHash Hamming distance; below = near-duplicate (0–64)

    "output_min_visible_s":    3.0,    # min seconds visible to include identity in output JSON
}

for _d in [YOLO_DATASET_DIR, TRAINING_OUTPUT_DIR, WEIGHTS_DIR, EXPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)
