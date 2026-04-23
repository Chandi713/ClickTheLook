import os
import sys

import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO

from config import TRAINING_CONFIG, TRAINING_OUTPUT_DIR, class_names


def load_best_model(current_model: YOLO) -> YOLO:
    best_path = os.path.join(TRAINING_OUTPUT_DIR,
                             TRAINING_CONFIG.get("name", "yolo8s_deepfashion2"),
                             "weights", "best.pt")
    if os.path.exists(best_path):
        print(f"Loading best checkpoint from current run: {best_path}")
        return YOLO(best_path)
    print("Best checkpoint not found — using model from memory.")
    return current_model


def run_validation(model: YOLO, yaml_path: str):
    metrics = model.val(data=yaml_path)
    print(f"\n{'='*60}\nVALIDATION RESULTS\n{'='*60}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")
    print("\nPer-class AP50:")
    for name, ap in zip(class_names, metrics.box.ap50):
        print(f"  {name:25s}: {ap:.4f}")
    return metrics


def run_test_evaluation(model: YOLO, yaml_path: str):
    """Final honest evaluation on the held-out test set. Run only once on the final model."""
    print(f"\n{'='*60}\nFINAL TEST SET EVALUATION\n{'='*60}")
    print("WARNING: run only once on your final model — do not use to tune hyperparameters.")
    metrics = model.val(data=yaml_path, split="test")
    print(f"\nmAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")
    print("\nPer-class AP50 (test):")
    for name, ap in zip(class_names, metrics.box.ap50):
        print(f"  {name:25s}: {ap:.4f}")
    return metrics


def visualize_results():
    rdir = os.path.join(TRAINING_OUTPUT_DIR, TRAINING_CONFIG.get("name", "yolo8s_deepfashion2"))
    for fname, title in [("results.png", "Training Results"), ("confusion_matrix.png", "Confusion Matrix")]:
        p = os.path.join(rdir, fname)
        if os.path.exists(p):
            plt.figure(figsize=(15, 10) if "results" in fname else (12, 10))
            plt.imshow(Image.open(p))
            plt.axis("off")
            plt.title(title)
            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv11 model.")
    parser.add_argument("--yaml-path", required=True,
                        help="Path to dataset YAML.")
    parser.add_argument("--model-path", default=None,
                        help="Path to model weights (.pt). Defaults to best.pt in runs dir.")
    args = parser.parse_args()

    if args.model_path:
        model = YOLO(args.model_path)
    else:
        model = load_best_model(None)
        if model is None:
            parser.error(f"No best.pt found in {TRAINING_OUTPUT_DIR}. Provide --model-path.")

    run_validation(model, args.yaml_path)
    visualize_results()
