import gc
import os
import random
import sys

# Ensure the pipeline root is on the path so all packages resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("All libraries imported successfully!")
print(f"PyTorch: {torch.__version__}")

# -- GPU status --
print("GPU Status:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

import mlflow

from config import (
    CATEGORIES, CSV_BATCH_SIZE, DATA_ROOT, IMAGE_DOWNLOAD_BATCH, LOCAL_PATHS,
    MIN_DISK_FREE_GB, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, NUM_CLASSES,
    TEST_SPLIT_RATIO, TRAINING_CONFIG, TRAINING_OUTPUT_DIR, WEIGHTS_DIR,
    YOLO_DATASET_DIR, class_names,
)

print(f"Using {TRAINING_CONFIG['cache']} for cache.")
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Batch size is {TRAINING_CONFIG['batch']} for {gpu_mem:.0f} GB GPU")
    if torch.cuda.device_count() > 1:
        print(f"Multi-GPU: {torch.cuda.device_count()} GPUs, batch={TRAINING_CONFIG['batch']}")
else:
    print(f"Batch size is {TRAINING_CONFIG['batch']} (CPU mode)")

print(f"\nData root  : {DATA_ROOT}")
print(f"CSV Batch  : {CSV_BATCH_SIZE:,} | Image Batch: {IMAGE_DOWNLOAD_BATCH}")
print(f"Min Free Disk: {MIN_DISK_FREE_GB} GB")
print(f"Classes    : {NUM_CLASSES}")

# ---------------------------------------------------------------------------
# 2. Path Verification
# ---------------------------------------------------------------------------
from src.utils.utils import get_disk_free_gb, verify_local_paths

verify_local_paths()

# ---------------------------------------------------------------------------
# 3. Disk Space Check
# ---------------------------------------------------------------------------
free = get_disk_free_gb(YOLO_DATASET_DIR)
print(f"Disk free space: {free:.1f} GB")
if free < MIN_DISK_FREE_GB:
    print(f"  ⚠ Below minimum ({MIN_DISK_FREE_GB} GB)!")
elif free < 50:
    print(f"  Pipeline will stop gracefully if disk fills up.")
else:
    print(f"  Plenty of space.")

# ---------------------------------------------------------------------------
# 4. Data Exploration
# ---------------------------------------------------------------------------
from src.data.data_loader import loader
from src.data.data_analysis import analyze_metadata

print("Loading training metadata...")
train_df = loader.load_csv(LOCAL_PATHS["train_csv"])
print(f"Shape: {train_df.shape}")
train_df.head()

analyze_metadata(train_df, "train")

print("Loading validation metadata...")
val_df = loader.load_csv(LOCAL_PATHS["val_csv"])
analyze_metadata(val_df, "validation")

del train_df, val_df
gc.collect()
print("Freed exploration DataFrames.")

# ---------------------------------------------------------------------------
# 5. Batch Conversion: DeepFashion2 → YOLO Format
# ---------------------------------------------------------------------------
from src.data.conversion import carve_test_split, convert_split, load_test_filenames
from src.data.dataset import labels_exist, restore_symlinks, symlinks_valid, test_labels_exist
from src.data.data_loader import loader as _loader

# ── Carve test split (once, deterministic) ────────────────────────────────────
test_fnames  = load_test_filenames()
train_fnames = None

if test_fnames is None:
    print(f"Carving test split ({TEST_SPLIT_RATIO:.0%} of training images, seed=42)...")
    train_fnames, test_fnames = carve_test_split(LOCAL_PATHS["train_csv"], TEST_SPLIT_RATIO)
else:
    print(f"Test split already carved: {len(test_fnames):,} images held out.")

# ── Convert all splits ────────────────────────────────────────────────────────
if not labels_exist():
    if train_fnames is None:
        # Compute train fnames as complement of test (only needed here on first run)
        _all = set()
        for _, _chunk in _loader.stream_csv_batches(LOCAL_PATHS["train_csv"]):
            _all.update(os.path.basename(p) for p in _chunk["path"])
            del _chunk
        train_fnames = _all - test_fnames

    print("PHASE: Converting training data to YOLO format\n")
    train_stats = convert_split(
        "train", LOCAL_PATHS["train_csv"], LOCAL_PATHS["train_images"],
        allowed_fnames=train_fnames,
    )
    print("\nPHASE: Converting validation data to YOLO format\n")
    val_stats = convert_split(
        "val", LOCAL_PATHS["val_csv"], LOCAL_PATHS["val_images"]
    )
    print("\nPHASE: Converting test data to YOLO format\n")
    convert_split(
        "test", LOCAL_PATHS["train_csv"], LOCAL_PATHS["train_images"],
        allowed_fnames=test_fnames,
    )
elif not symlinks_valid():
    print("Labels found. Symlinks broken — restoring without re-conversion...")
    restore_symlinks()
    print("Symlinks restored.")
else:
    print("YOLO-format data ready. Skipping conversion.")

# Backfill test split if added to an existing installation
if not test_labels_exist():
    print("\nPHASE: Converting test data to YOLO format (backfill)\n")
    convert_split(
        "test", LOCAL_PATHS["train_csv"], LOCAL_PATHS["train_images"],
        allowed_fnames=test_fnames,
    )

# ---------------------------------------------------------------------------
# 6. Dataset Verification + YAML
# ---------------------------------------------------------------------------
from src.data.dataset import create_dataset_yaml, verify_dataset

verify_dataset()
yaml_path = create_dataset_yaml()

# ---------------------------------------------------------------------------
# 7. Model Training
# ---------------------------------------------------------------------------
import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

from src.training.train import export_model, load_model, train_model, update_model_registry

# ── MLflow: initialise experiment and start run ───────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)
active_run = mlflow.start_run(run_name=TRAINING_CONFIG["name"])
print(f"MLflow run ID : {active_run.info.run_id}")
print(f"Experiment    : {MLFLOW_EXPERIMENT}")
print(f"Tracking URI  : {MLFLOW_TRACKING_URI}")

mlflow.log_params({k: str(v) for k, v in TRAINING_CONFIG.items()})

model = load_model()
results = train_model(model, yaml_path)

# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------
from src.evaluation.evaluate import load_best_model, run_validation, visualize_results

model = load_best_model(model)
metrics = run_validation(model, yaml_path)
visualize_results()

# ── MLflow: log validation metrics ────────────────────────────────────────────
mlflow.log_metrics({
    "mAP50":     metrics.box.map50,
    "mAP50_95":  metrics.box.map,
    "precision": metrics.box.mp,
    "recall":    metrics.box.mr,
})
for cls_name, ap in zip(class_names, metrics.box.ap50):
    mlflow.log_metric(f"AP50_{cls_name}", ap)
print("Metrics logged to MLflow.")

# ---------------------------------------------------------------------------
# 9. Model registry comparison + conditional export
# ---------------------------------------------------------------------------
should_export, registry_status = update_model_registry(model, metrics)

# ── MLflow: tag the run outcome ────────────────────────────────────────────────
mlflow.set_tag("registry_result", registry_status)
mlflow.set_tag("base_model",      TRAINING_CONFIG["model"])
mlflow.set_tag("run_name",        TRAINING_CONFIG["name"])
print(f"Registry result: {registry_status}")

# ---------------------------------------------------------------------------
# 10. Inference
# ---------------------------------------------------------------------------
from src.inference.inference import run_sample_inference

run_sample_inference(model)

# ---------------------------------------------------------------------------
# 11. Export (only if new model earned best slot)
# ---------------------------------------------------------------------------
if should_export:
    export_model(model)
else:
    print("Model did not improve over stored best. Skipping export.")

# ── MLflow: log artifacts and close run ───────────────────────────────────────
run_dir = os.path.join(TRAINING_OUTPUT_DIR, TRAINING_CONFIG["name"])
for fname in ["results.png", "results.csv", "confusion_matrix.png"]:
    p = os.path.join(run_dir, fname)
    if os.path.exists(p):
        mlflow.log_artifact(p, artifact_path="training_artifacts")

best_pt = os.path.join(WEIGHTS_DIR, "best.pt")
if registry_status == "best" and os.path.exists(best_pt):
    mlflow.log_artifact(best_pt, artifact_path="weights")

scores_json = os.path.join(WEIGHTS_DIR, "scores.json")
if os.path.exists(scores_json):
    mlflow.log_artifact(scores_json, artifact_path="registry")

mlflow.end_run()
print(f"\nMLflow run closed.")
print(f"Dashboard : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
print(f"Then open : http://localhost:5000")

# ---------------------------------------------------------------------------
# 11. Final Test Evaluation (run only once on your final model)
# ---------------------------------------------------------------------------
# Uncomment when you have finalized the model — do NOT run repeatedly.
# from src.evaluation.evaluate import run_test_evaluation
# test_metrics = run_test_evaluation(model, yaml_path)

# ---------------------------------------------------------------------------
# 12. Cleanup (disabled — remove comments to run)
# ---------------------------------------------------------------------------
# from src.utils.cleanup import cleanup
# cleanup(keep_weights=True)

# ---------------------------------------------------------------------------
# 12. Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"\nData root  : {DATA_ROOT}")
print(f"YOLO dir   : {YOLO_DATASET_DIR}")
print(f"Weights    : {WEIGHTS_DIR}/best.pt  (best), last.pt (second-best)")
print(f"Scores     : {WEIGHTS_DIR}/scores.json")
print(f"\nModel      : {TRAINING_CONFIG['model']} | {TRAINING_CONFIG['epochs']} epochs")
print(f"Batch      : {TRAINING_CONFIG['batch']} | Device: {TRAINING_CONFIG['device']}")
print(f"\nClasses ({NUM_CLASSES}):")
for i in range(NUM_CLASSES):
    print(f"  {i:2d}: {CATEGORIES[i + 1]}")
print(f"\nMLflow")
print(f"  Experiment : {MLFLOW_EXPERIMENT}")
print(f"  Tracking   : {MLFLOW_TRACKING_URI}")
print(f"  Dashboard  : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
print(f"  Then open  : http://localhost:5000")
