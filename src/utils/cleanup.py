import gc
import os
import shutil

from config import TRAINING_OUTPUT_DIR, YOLO_DATASET_DIR
from src.utils.utils import get_disk_free_gb


def cleanup(keep_weights: bool = True):
    if os.path.exists(YOLO_DATASET_DIR):
        shutil.rmtree(YOLO_DATASET_DIR)
        print(f"  ✓ Removed {YOLO_DATASET_DIR}")
    if not keep_weights and os.path.exists(TRAINING_OUTPUT_DIR):
        shutil.rmtree(TRAINING_OUTPUT_DIR)
        print(f"  ✓ Removed {TRAINING_OUTPUT_DIR}")
    gc.collect()
    print(f"  Disk free: {get_disk_free_gb(YOLO_DATASET_DIR):.1f} GB")
