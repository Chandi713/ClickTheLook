# ClickTheLook

An end-to-end video fashion intelligence pipeline. Starting from raw DeepFashion2 annotations, it fine-tunes a YOLO8s model to detect clothing items across 13 categories, then runs that model on video — tracking each garment across frames, assigning it a stable identity, selecting its best-quality crop, and producing a time-indexed JSON manifest with precise timestamps. The output is designed to power downstream applications such as "shop the look" overlays and visual fashion search.

---

## What It Does

ClickTheLook is split into two phases:

**Phase 1 — Training:** Converts DeepFashion2 annotations into YOLO format, trains a YOLO8s detection model, evaluates it on a held-out validation set, and saves the best-performing weights to a managed model registry. All metrics and artifacts are tracked in MLflow.

**Phase 2 — Inference:** Takes a video file and the trained model, runs the full detection and tracking pipeline frame by frame, and produces a structured JSON file listing every clothing item detected — its category, a cropped image, and the exact timestamps for every window in which it was visible.

---

## Quick Demo

Want to test the system without setting up the full project? The `Demo/` folder is a self-contained package that runs the entire inference pipeline end to end — on a local machine, a cloud instance, or an HPC cluster.

**What's included:**

```
Demo/
├── Demo.ipynb          # Full inference pipeline — run all cells top to bottom
├── requirements.txt    # All dependencies, install once
└── best.pt             # Fine-tuned YOLO8s weights, ready to use
```

**Steps to run:**

1. Place your input video inside `Demo/` or any sub-folder within it.
2. Open `Demo.ipynb` and edit the paths in **Step 2 (Paths and Settings)** — this is the only cell you need to touch:

```python
VIDEO_PATH      = "./your_video.mp4"          # path to input video
MODEL_PATH      = "./your_model.pt"           # path to model weights
SAVE_VIDEO      = "./output/annotated.mp4"    # where to save annotated video
OUTPUT_BASE_DIR = "./output"                  # all output folders created automatically
```

3. Run all cells from top to bottom.

All output folders (`output/`, `detections/`, `logs/`) are created automatically by the notebook — nothing needs to be set up in advance. When the run completes, the time-indexed JSON manifest and crop images are waiting in `OUTPUT_BASE_DIR`.

---

## Pipeline

### Phase 1 — Training Pipeline

```
DeepFashion2 Dataset (CSV + images)
         │
         ▼
┌────────────────────────┐
│  Data Loading          │  Streams train/val CSVs, inspects class distribution
│  & Analysis            │  and annotation statistics
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Dataset Conversion    │  Converts DeepFashion2 bounding box annotations to
│                        │  YOLO label format; carves a 10% held-out test split;
│                        │  creates symlinks for zero-copy image resolution
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  YOLO8s Fine-tuning    │  AdamW + cosine LR decay + AMP; input size 512×512;
│                        │  batch 64; label smoothing; mosaic & HSV augmentation
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Validation &          │  Computes mAP50, mAP50-95, precision, recall, and
│  Evaluation            │  per-class AP50 on the validation set
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Model Registry        │  Compares new run against stored best by mAP50;
│                        │  promotes, demotes, or discards automatically;
│                        │  exports winning weights to ONNX
└────────┬───────────────┘
         │
         ▼
  artifacts/weights/best.pt
```

### Phase 2 — Inference Pipeline

```
Input Video + best.pt
         │
         ▼
┌────────────────────────┐
│  Video Ingestion       │  Reads frames sequentially via OpenCV; extracts
│                        │  resolution, FPS, and frame count for timestamps
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Clothing Detection    │  YOLO8s detects garments per frame; filtered by
│  (YOLO8s)              │  confidence ≥ 0.50 and NMS IoU 0.45
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  DeepSORT Tracking     │  Assigns and maintains tracker IDs across frames
│                        │  using motion prediction and appearance embeddings
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Global Re-ID          │  Re-links garment identities that leave and
│  (GlobalIdentityManager│  re-enter the frame; gates: class → time gap →
│                        │  spatial drift → appearance similarity
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Best-Crop Selection   │  Accumulates all crops in memory; keeps the
│                        │  largest bounding box seen across the full video
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Crop Quality Filtering│  Removes blurry crops (Laplacian variance < 90)
│                        │  and near-duplicates (perceptual hash distance < 13)
└────────┬───────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  Output Generation                         │
│  · Annotated video (.mp4)                  │
│  · Crop images per identity (.jpg)         │
│  · Time-indexed identity JSON              │
│  · Full diagnostic run log JSON            │
│  · MLflow experiment record               │
└────────────────────────────────────────────┘
```

---

## Features

- **DeepFashion2 data pipeline** — streams CSVs, converts annotations to YOLO format, and carves a reproducible test split
- **YOLO8s fine-tuning** — 13-class clothing detection with AMP, cosine LR, and mosaic augmentation
- **Validation & per-class evaluation** — mAP50, mAP50-95, precision, recall, and AP50 per category
- **Model registry** — automatically tracks best and second-best weights by mAP50 across training runs
- **DeepSORT tracking** — maintains garment identity across frames using motion and appearance
- **Global Re-Identification** — recovers garment identity across long disappearance gaps (up to 43 s)
- **Best-frame crop selection** — globally best crop per identity selected across the entire video
- **Crop quality filtering** — blur detection and near-duplicate removal before saving
- **Time-indexed JSON output** — `HH:MM:SS.mmm` timestamps for every visibility window per garment
- **MLflow experiment tracking** — parameters, metrics, and artifacts logged automatically per run
- **HPC-ready notebook** — fully self-contained, no project imports required
- **GPU acceleration** — CUDA (HPC/server), Apple MPS (Mac), CPU fallback

---

## Tech Stack

| Component               | Technology                         |
|-------------------------|------------------------------------|
| Object Detection        | Ultralytics YOLO8s                 |
| Multi-Object Tracking   | DeepSORT (`deep-sort-realtime`)    |
| Video Processing        | OpenCV                             |
| Dataset                 | DeepFashion2 (13 clothing classes) |
| Numerical Computing     | NumPy                              |
| GPU Acceleration        | NVIDIA CUDA / Apple MPS            |
| Experiment Tracking     | MLflow                             |
| Configuration           | Python-dotenv + `config.py`        |
| HPC Execution           | Jupyter Notebook (self-contained)  |

---

## Dataset & Classes

Trained on **DeepFashion2**, a large-scale benchmark for clothing detection and retrieval containing richly annotated images across consumer, street, and shop photography. Accessories, footwear, and non-clothing objects are out of scope.

| ID | Category              | ID | Category            |
|----|-----------------------|----|---------------------|
| 1  | Short Sleeve Top      | 8  | Trousers            |
| 2  | Long Sleeve Top       | 9  | Skirt               |
| 3  | Short Sleeve Outwear  | 10 | Short Sleeve Dress  |
| 4  | Long Sleeve Outwear   | 11 | Long Sleeve Dress   |
| 5  | Vest                  | 12 | Vest Dress          |
| 6  | Sling                 | 13 | Sling Dress         |
| 7  | Shorts                |    |                     |

---

## Installation

**Prerequisites:** Python 3.8+, pip

```bash
git clone https://github.com/your-username/ClickTheLook.git
cd ClickTheLook
pip install -r requirements.txt
```

**Environment setup** — create a `.env` file in the project root:

```env
DATA_ROOT=/path/to/deepfashion2
YOLO_DATASET_DIR=./data/yolo
TRAINING_OUTPUT_DIR=./artifacts/runs
WEIGHTS_DIR=./artifacts/weights
EXPORTS_DIR=./artifacts/exports
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT=ClickTheLook
INFERENCE_MODEL_PATH=./artifacts/weights/best.pt
```

---

## Usage

### 1. Train the Detection Model

```bash
python scripts/main.py
```

Runs the full training pipeline: loads DeepFashion2 metadata, converts annotations to YOLO format (skipped if already done), trains YOLO8s, evaluates on the validation set, updates the model registry, and exports to ONNX if the new run is the best so far. All metrics and artifacts are logged to MLflow.

### 2. Run the Inference Pipeline on Video

```bash
python scripts/live.py --source input.mp4 --save output.mp4
```

| Argument | Description |
|---|---|
| `--source` | Path to input video file |
| `--save` | Path to save annotated output video |
| `--no-show` | Disable real-time preview (recommended for servers) |

### 3. Run on HPC Cluster

Open `Demo.ipynb` on your cluster and edit **Step 2 (Paths and Settings)** only:

```python
VIDEO_PATH      = "/path/to/input.mp4"
MODEL_PATH      = "/path/to/best.pt"
SAVE_VIDEO      = "/path/to/output.mp4"
OUTPUT_BASE_DIR = "/path/to/outputs"
MLFLOW_TRACKING_URI = "./mlruns"
```

Then run all cells top to bottom. No other project files are needed — all logic is self-contained in the notebook.

### 4. View MLflow Results

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://localhost:5000` to browse training metrics, inference run stats, and logged artifacts.

---

## Configuration

All parameters are centralised in `config.py` (scripts) or **Step 2** of `Demo.ipynb` (HPC).

### Training

| Parameter | Default | Description |
|---|---|---|
| `model` | `yolo8s.pt` | Base model (ImageNet pretrained) |
| `epochs` | `1` | Training epochs |
| `batch` | `64` | Batch size (scales with GPU count) |
| `imgsz` | `512` | Input image size |
| `optimizer` | `AdamW` | Optimiser |
| `cos_lr` | `True` | Cosine learning rate decay |
| `amp` | `True` | Automatic Mixed Precision |
| `label_smoothing` | `0.1` | Label smoothing factor |
| `patience` | `10` | Early stopping patience |

### Detection

| Parameter | Default | Description |
|---|---|---|
| `conf` | `0.50` | Minimum detection confidence |
| `iou` | `0.45` | NMS IoU threshold |

### DeepSORT Tracker

| Parameter | Default | Description |
|---|---|---|
| `deepsort_max_age` | `5` | Frames before a lost track is deleted |
| `deepsort_n_init` | `3` | Detections needed to confirm a track |
| `deepsort_max_cosine_distance` | `0.30` | Appearance similarity gate |
| `deepsort_embedder` | `mobilenet` | Appearance feature extractor |

### Global Re-Identification

| Parameter | Default | Description |
|---|---|---|
| `use_global_ids` | `True` | Enable/disable re-ID layer |
| `gid_max_gap_s` | `43.0` | Max seconds a lost identity stays in the re-ID pool |
| `gid_cosine_threshold` | `0.15` | Max appearance distance to accept a re-ID match |
| `gid_spatial_gate_pps` | `50.0` | Max pixels/second drift allowed for a match |

### Crop Quality Filtering

| Parameter | Default | Description |
|---|---|---|
| `dedup_blur_threshold` | `90.0` | Laplacian variance below this = too blurry |
| `dedup_phash_threshold` | `13` | Perceptual hash distance below this = near-duplicate |

### Output Filtering

| Parameter | Default | Description |
|---|---|---|
| `output_min_visible_s` | `3.0` | Exclude identities visible for less than this |

---

## Output Structure

```
output/
├── logs/
│   └── <run_id>.json          # Full diagnostic log (latencies, anomalies, timeline)
├── detections/
│   └── <run_id>/
│       ├── 3_trousers.jpg
│       ├── 7_long_sleeve_top.jpg
│       └── ...                # Best crop image per identity
└── output/
    └── <run_id>.json          # Filtered identity manifest with timestamps
mlruns/                        # MLflow experiment store
```

### Identity JSON Format (`output/<run_id>.json`)

```json
[
  {
    "id": 12,
    "class": "long_sleeve_top",
    "crop_path": "./detections/<run_id>/12_long_sleeve_top.jpg",
    "num_intervals": 2,
    "intervals": [
      {
        "start_frame": 143,
        "end_frame": 289,
        "start_ts": "00:00:04.767",
        "end_ts": "00:00:09.633",
        "duration_seconds": 4.867,
        "duration_text": "4s"
      },
      {
        "start_frame": 540,
        "end_frame": 721,
        "start_ts": "00:00:18.000",
        "end_ts": "00:00:24.033",
        "duration_seconds": 6.033,
        "duration_text": "6s"
      }
    ],
    "total_visible_seconds": 10.9,
    "total_visible_text": "10s"
  }
]
```

Only identities that pass all filters are included:
- Crop image exists on disk (survived blur + dedup filtering)
- `total_visible_seconds >= 3.0`

---

## MLflow Tracking

Every training and inference run automatically logs to MLflow:

| Category | Details |
|---|---|
| **Training params** | Model, epochs, batch, imgsz, optimizer, augmentation flags |
| **Training metrics** | mAP50, mAP50-95, precision, recall, per-class AP50 |
| **Inference params** | Model path, device, conf/IoU, all tracker and re-ID thresholds, video resolution & FPS |
| **Inference metrics** | Avg FPS, per-stage latency avg/p95, total detections, unique IDs, crops saved, blur/dedup removal rates, re-ID effectiveness |
| **Artifacts** | Run log JSON, output JSON, crop grid PNG, training curves, confusion matrix, model weights |

---

## Project Structure

```
ClickTheLook/
├── config.py                        # All parameters in one place
├── requirements.txt
├── .env                             # Environment variables (not committed)
│
├── scripts/
│   ├── main.py                      # Full training pipeline entry point
│   └── live.py                      # Video inference pipeline entry point
│
├── configs/
│   └── run_config.yaml              # Inference run configuration file
│
├── src/
│   ├── data/
│   │   ├── data_loader.py           # Streams DeepFashion2 CSVs in batches
│   │   ├── data_analysis.py         # Analyses class distribution and annotation stats
│   │   ├── conversion.py            # Converts annotations to YOLO format; carves test split
│   │   └── dataset.py               # YAML generation, dataset verification, symlink management
│   │
│   ├── training/
│   │   └── train.py                 # Model training, model registry (best/last tracking), ONNX export
│   │
│   ├── evaluation/
│   │   └── evaluate.py              # Validation metrics, per-class AP50, test set evaluation
│   │
│   ├── inference/
│   │   └── inference.py             # Single-image inference and post-training sanity checks
│   │
│   ├── live/
│   │   ├── live_detect.py           # Main frame loop and pipeline orchestration
│   │   ├── deepsort_tracker.py      # DeepSORT wrapper and appearance embedding extraction
│   │   ├── tracker.py               # SORT tracker (lightweight fallback)
│   │   ├── sort.py                  # SORT algorithm implementation
│   │   ├── global_id.py             # Global Re-Identification manager
│   │   ├── run_logger.py            # Per-run statistics collection and JSON log writer
│   │   └── model_compare.py         # Side-by-side model output comparison utility
│   │
│   └── utils/
│       ├── utils.py                 # Shared helpers (disk checks, path verification)
│       └── cleanup.py               # Removes intermediate data while preserving weights
│
├── artifacts/
│   ├── weights/                     # best.pt, last.pt, scores.json (model registry)
│   ├── runs/                        # YOLO training run outputs
│   └── exports/                     # ONNX exported models
│
├── Demo.ipynb                       # Self-contained HPC inference notebook
└── mlruns/                          # MLflow tracking store
```

---

## Key Design Decisions

**Two-phase pipeline:** Training and inference are fully independent. Once `best.pt` is produced, the inference pipeline has no dependency on the training code — it can run anywhere, including HPC clusters via the self-contained notebook.

**Model registry:** The training script automatically compares each new run against the stored best by mAP50. The best weights are promoted and the previous best is preserved as `last.pt`. This prevents accidental overwriting of a good model and tracks which run produced it.

**Two-layer identity tracking:** DeepSORT maintains identity within short gaps; `GlobalIdentityManager` handles longer disappearances. Each layer can be toggled independently with a single config flag, making ablation straightforward.

**In-memory crop selection:** Crops are accumulated throughout the entire video and written to disk only after the final frame. This ensures the globally best crop is always saved, not just the first acceptable one.

**Separation of output and logs:** `output/` holds clean, filtered, actionable identity data. `logs/` holds full verbose diagnostics. Downstream consumers only ever need `output/`.

**Single configuration source:** Every threshold — detection, tracking, re-ID, quality, output — is declared once in `config.py` or the notebook's settings cell. There are no defaults scattered across source files.
