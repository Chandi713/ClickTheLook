import json
import os
import shutil
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO

from config import EXPORTS_DIR, TRAINING_CONFIG, TRAINING_OUTPUT_DIR, WEIGHTS_DIR


def load_model() -> YOLO:
    """
    Load from the globally managed best checkpoint if one exists,
    otherwise fall back to the pretrained base model (first run).
    """
    best_path = os.path.join(WEIGHTS_DIR, "best.pt")
    if os.path.exists(best_path):
        print(f"Fine-tuning from existing best: {best_path}")
        return YOLO(best_path)
    print(f"No checkpoint found — training from scratch: {TRAINING_CONFIG['model']}")
    return YOLO(TRAINING_CONFIG["model"])


def train_model(model: YOLO, yaml_path: str):
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Dataset : {yaml_path}")
    print(f"Device  : {TRAINING_CONFIG['device']}")
    print(f"Batch   : {TRAINING_CONFIG['batch']}")
    print()

    params = {k: v for k, v in TRAINING_CONFIG.items() if k != "model"}
    results = model.train(data=yaml_path, **params)
    print("\nTraining complete!")
    return results


def export_model(model: YOLO):
    export_path = model.export(format="onnx", dynamic=True)
    dest = os.path.join(EXPORTS_DIR, os.path.basename(export_path))
    shutil.move(str(export_path), dest)
    print(f"ONNX export: {dest}")


def update_model_registry(model: YOLO, metrics):
    """
    Compare the newly trained model against the stored best and last.

    Rules
    -----
    new > best          → new becomes best, old best becomes last.  Returns (True,  "best").
    last < new <= best  → new becomes last.                         Returns (False, "last").
    new <= last         → discarded, nothing written.               Returns (False, "discarded").

    On the very first run (no scores yet) the new model always becomes best.

    Returns (should_export: bool, status: str).
    """
    scores_path = os.path.join(WEIGHTS_DIR, "scores.json")
    new_map50   = float(metrics.box.map50)

    scores    = _load_scores(scores_path)
    best_map50 = scores.get("best")
    last_map50 = scores.get("last")

    # Path YOLO wrote the best checkpoint for this run
    run_best = os.path.join(
        TRAINING_OUTPUT_DIR,
        TRAINING_CONFIG.get("name", "yolo11_deepfashion2"),
        "weights", "best.pt",
    )

    print("\n" + "=" * 60)
    print("MODEL REGISTRY COMPARISON")
    print("=" * 60)
    print(f"  New        mAP50 : {new_map50:.4f}")
    print(f"  Best       mAP50 : {best_map50:.4f}" if best_map50 is not None
          else "  Best       mAP50 : (none — first run)")
    print(f"  Last       mAP50 : {last_map50:.4f}" if last_map50 is not None
          else "  Last       mAP50 : (none)")

    if not os.path.exists(run_best):
        print(f"  WARNING: run checkpoint not found at {run_best}. Skipping update.")
        return False, "discarded"

    best_dest = os.path.join(WEIGHTS_DIR, "best.pt")
    last_dest = os.path.join(WEIGHTS_DIR, "last.pt")

    # ── Case 1: first run OR new is better than stored best ───────────────────
    if best_map50 is None or new_map50 > best_map50:
        if os.path.exists(best_dest):          # promote old best → last
            shutil.copy2(best_dest, last_dest)
            scores["last"] = best_map50
        shutil.copy2(run_best, best_dest)
        scores["best"] = new_map50
        _save_scores(scores_path, scores)
        print(f"  -> NEW BEST  ({new_map50:.4f}). Exporting.")
        return True, "best"

    # ── Case 2: worse than best, but better than last (or no last yet) ────────
    if last_map50 is None or new_map50 > last_map50:
        shutil.copy2(run_best, last_dest)
        scores["last"] = new_map50
        _save_scores(scores_path, scores)
        print(f"  -> New LAST  ({new_map50:.4f}). Saved, not exported.")
        return False, "last"

    # ── Case 3: worse than both → discard ─────────────────────────────────────
    print(f"  -> DISCARDED ({new_map50:.4f} <= last {last_map50:.4f}). Nothing saved.")
    return False, "discarded"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_scores(scores_path: str) -> dict:
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)
    return {}


def _save_scores(scores_path: str, scores: dict):
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or export the YOLOv11 model.")
    parser.add_argument("--action", required=True, choices=["train", "export"],
                        help="'train' — run full training. 'export' — export weights to ONNX.")
    parser.add_argument("--yaml-path", default=None,
                        help="Path to dataset YAML. Required for --action train.")
    args = parser.parse_args()

    if args.action == "train":
        if not args.yaml_path:
            parser.error("--yaml-path is required for --action train.")
        model = load_model()
        train_model(model, args.yaml_path)
    elif args.action == "export":
        model = load_model()
        export_model(model)
