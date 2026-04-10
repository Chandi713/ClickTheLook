"""
Side-by-side comparison of two YOLO models on a video source.

Ported from yolov8-object-tracking-main/yolo/v8/detect/evaluate_models.py,
adapted to use ClickTheLook config and class names.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO

from config import CATEGORIES, WEIGHTS_DIR, LIVE_CONFIG


def run_model(model_path: str, source: str, device=None, conf: float = 0.25, iou: float = 0.45):
    """Run inference on a video, collect per-frame detection stats."""
    device = device or LIVE_CONFIG.get("device", "cpu")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_stats = []

    print(f"\n{'=' * 60}")
    print(f"Running: {model_path}  ({total_frames} frames)")
    print(f"{'=' * 60}")

    results = model.predict(
        source=source, device=device, conf=conf, iou=iou,
        stream=True, verbose=False,
    )

    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            frame_stats.append({
                "frame": frame_idx + 1,
                "num_detections": 0,
                "mean_conf": 0.0,
                "min_conf": 0.0,
                "max_conf": 0.0,
                "class_counts": {},
            })
            continue

        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        class_counts = defaultdict(int)
        for c in classes:
            # Use ClickTheLook CATEGORIES (1-indexed) for class names
            class_counts[CATEGORIES.get(c + 1, str(c))] += 1

        frame_stats.append({
            "frame": frame_idx + 1,
            "num_detections": len(confs),
            "mean_conf": float(np.mean(confs)),
            "min_conf": float(np.min(confs)),
            "max_conf": float(np.max(confs)),
            "class_counts": dict(class_counts),
        })

        if (frame_idx + 1) % 100 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames} | "
                  f"detections={len(confs)} | "
                  f"mean_conf={np.mean(confs):.3f}")

    return frame_stats


def _summarize(stats):
    total_dets = sum(s["num_detections"] for s in stats)
    frames_with_dets = sum(1 for s in stats if s["num_detections"] > 0)
    all_confs = [s["mean_conf"] for s in stats if s["num_detections"] > 0]
    all_max = [s["max_conf"] for s in stats if s["num_detections"] > 0]
    all_min = [s["min_conf"] for s in stats if s["num_detections"] > 0]

    class_totals = defaultdict(int)
    for s in stats:
        for cls, cnt in s["class_counts"].items():
            class_totals[cls] += cnt

    return {
        "total_frames": len(stats),
        "frames_with_detections": frames_with_dets,
        "total_detections": total_dets,
        "avg_dets_per_frame": total_dets / len(stats) if stats else 0,
        "mean_conf": float(np.mean(all_confs)) if all_confs else 0.0,
        "max_conf": float(np.max(all_max)) if all_max else 0.0,
        "min_conf": float(np.min(all_min)) if all_min else 0.0,
        "class_totals": dict(class_totals),
    }


def print_comparison(stats_a, model_a, stats_b, model_b):
    """Print side-by-side frame-level and summary comparison."""
    name_a = Path(model_a).stem
    name_b = Path(model_b).stem
    col = 12
    max_frames = max(len(stats_a), len(stats_b))

    print(f"\n{'=' * 90}")
    print("FRAME-LEVEL COMPARISON (every 50 frames)")
    print(f"{'=' * 90}")
    print(f"{'':6}   {'[ ' + name_a + ' ]':^{col * 2 + 3}}   {'[ ' + name_b + ' ]':^{col * 2 + 3}}")
    header = (f"{'Frame':>6} | "
              f"{'Dets':>{col}} {'MeanConf':>{col}} | "
              f"{'Dets':>{col}} {'MeanConf':>{col}}")
    print(header)
    print("-" * len(header))

    for i in range(0, max_frames, 50):
        a = stats_a[i] if i < len(stats_a) else None
        b = stats_b[i] if i < len(stats_b) else None
        a_dets = a["num_detections"] if a else "-"
        a_conf = f"{a['mean_conf']:.3f}" if a and a["num_detections"] > 0 else "-"
        b_dets = b["num_detections"] if b else "-"
        b_conf = f"{b['mean_conf']:.3f}" if b and b["num_detections"] > 0 else "-"
        print(f"{i + 1:>6} | {str(a_dets):>{col}} {a_conf:>{col}} | "
              f"{str(b_dets):>{col}} {b_conf:>{col}}")

    # Summary
    sa = _summarize(stats_a)
    sb = _summarize(stats_b)

    print(f"\n{'=' * 70}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {name_a:>18} {name_b:>18}")
    print(f"{'-' * 70}")

    rows = [
        ("Total frames",         sa["total_frames"],                    sb["total_frames"]),
        ("Frames w/ detections", sa["frames_with_detections"],          sb["frames_with_detections"]),
        ("Total detections",     sa["total_detections"],                sb["total_detections"]),
        ("Avg detections/frame", f"{sa['avg_dets_per_frame']:.2f}",    f"{sb['avg_dets_per_frame']:.2f}"),
        ("Mean confidence",      f"{sa['mean_conf']:.4f}",             f"{sb['mean_conf']:.4f}"),
        ("Max confidence",       f"{sa['max_conf']:.4f}",              f"{sb['max_conf']:.4f}"),
        ("Min confidence",       f"{sa['min_conf']:.4f}",              f"{sb['min_conf']:.4f}"),
    ]
    for label, va, vb in rows:
        print(f"{label:<30} {str(va):>18} {str(vb):>18}")

    # Class breakdown
    all_classes = sorted(set(list(sa["class_totals"].keys()) + list(sb["class_totals"].keys())))
    if all_classes:
        print(f"\n{'-' * 70}")
        print(f"{'Class Breakdown':<30} {name_a:>18} {name_b:>18}")
        print(f"{'-' * 70}")
        for cls in all_classes:
            va = sa["class_totals"].get(cls, 0)
            vb = sb["class_totals"].get(cls, 0)
            print(f"  {cls:<28} {str(va):>18} {str(vb):>18}")

    print(f"{'=' * 70}\n")


def run_comparison(model_a: str, model_b: str, source: str,
                   device=None, conf: float = 0.25, iou: float = 0.45):
    """Run both models and print comparison table."""
    stats_a = run_model(model_a, source, device, conf, iou)
    stats_b = run_model(model_b, source, device, conf, iou)
    print_comparison(stats_a, model_a, stats_b, model_b)
    return stats_a, stats_b


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two ClickTheLook models on a video.")
    parser.add_argument("--model-a", default=os.path.join(WEIGHTS_DIR, "best.pt"),
                        help="Path to first model (default: best.pt from registry)")
    parser.add_argument("--model-b", default=os.path.join(WEIGHTS_DIR, "last.pt"),
                        help="Path to second model (default: last.pt from registry)")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--device", default=None, help="Device: mps / cpu / cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    args = parser.parse_args()

    run_comparison(args.model_a, args.model_b, args.source,
                   args.device, args.conf, args.iou)
