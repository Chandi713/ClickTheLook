#!/usr/bin/env python3
"""
ClickTheLook — Live Detection & Tracking CLI

Usage:
    # Webcam (default)
    python scripts/live.py

    # Video file
    python scripts/live.py --source path/to/video.mp4

    # Save annotated output
    python scripts/live.py --source video.mp4 --save output.mp4

    # Compare two models on a video
    python scripts/live.py --compare --source video.mp4
    python scripts/live.py --compare --source video.mp4 --model-a best.pt --model-b last.pt
"""
import os
import sys

# Ensure pipeline root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ClickTheLook — live detection, tracking, and model comparison.",
    )
    # Mode
    parser.add_argument("--compare", action="store_true",
                        help="Run model comparison mode instead of live detection.")

    # Common
    parser.add_argument("--source", default=0,
                        help="Video source: 0 for webcam, or path to video file.")
    parser.add_argument("--device", default=None,
                        help="Device: cpu, mps, cuda, or index.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU threshold.")

    # Live detection args
    parser.add_argument("--model", default=None,
                        help="Path to .pt weights (live detection mode).")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable display window.")
    parser.add_argument("--save", default=None,
                        help="Path to save annotated output video.")

    # Compare mode args
    parser.add_argument("--model-a", default=None,
                        help="First model for comparison (default: best.pt).")
    parser.add_argument("--model-b", default=None,
                        help="Second model for comparison (default: last.pt).")

    args = parser.parse_args()

    # Coerce source to int if it looks like a webcam index
    src = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    if args.compare:
        from src.live.model_compare import run_comparison
        from config import WEIGHTS_DIR

        model_a = args.model_a or os.path.join(WEIGHTS_DIR, "best.pt")
        model_b = args.model_b or os.path.join(WEIGHTS_DIR, "last.pt")
        run_comparison(
            model_a=model_a,
            model_b=model_b,
            source=str(src),
            device=args.device,
            conf=args.conf or 0.25,
            iou=args.iou or 0.45,
        )
    else:
        from src.live.live_detect import run_live_detection

        run_live_detection(
            source=src,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            show=not args.no_show,
            save_video=args.save,
        )


if __name__ == "__main__":
    main()
