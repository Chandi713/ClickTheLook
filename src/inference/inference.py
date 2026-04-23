import os
import random
import sys

import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO

from config import CATEGORIES, INFERENCE_MODEL_PATH, YOLO_DATASET_DIR


def run_inference(model: YOLO, image_path: str, conf: float = 0.5) -> dict:
    results = model(image_path, conf=conf)
    dets = []
    for r in results:
        for box in r.boxes:
            dets.append({
                "class_name": CATEGORIES[int(box.cls[0]) + 1],
                "confidence": float(box.conf[0]),
                "bbox":       box.xyxy[0].tolist(),
            })
    return {"image_path": image_path, "detections": dets, "results": results}


def visualize_inference(result: dict, show_labels: bool = True):
    ann = result["results"][0].plot(labels=show_labels, conf=show_labels)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Detected {len(result['detections'])} items")
    plt.show()
    for d in result["detections"]:
        print(f"  {d['class_name']}: {d['confidence']:.2f}")


def run_sample_inference(model: YOLO):
    vdir = os.path.join(YOLO_DATASET_DIR, "images", "val")
    if os.path.exists(vdir):
        imgs = [f for f in os.listdir(vdir) if f.endswith((".jpg", ".jpeg", ".png"))]
        for p in random.sample([os.path.join(vdir, f) for f in imgs], min(4, len(imgs))):
            print(f"\n{'='*60}\n{os.path.basename(p)}")
            visualize_inference(run_inference(model, p))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with a trained YOLOv11 model.")
    parser.add_argument("--model-path", default=INFERENCE_MODEL_PATH,
                        help=f"Path to model weights (.pt). Defaults to INFERENCE_MODEL_PATH from .env ({INFERENCE_MODEL_PATH}).")
    parser.add_argument("--image-path", default=None,
                        help="Path to a single image. Omit to run on 4 random val samples.")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5).")
    parser.add_argument("--no-labels", action="store_true",
                        help="Show bounding boxes only, without class labels or confidence scores.")
    args = parser.parse_args()

    model = YOLO(args.model_path)
    if args.image_path:
        visualize_inference(
            run_inference(model, args.image_path, conf=args.conf),
            show_labels=not args.no_labels,
        )
    else:
        run_sample_inference(model)
