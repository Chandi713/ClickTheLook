"""
Real-time video detection + SORT tracking using a trained ClickTheLook model.

Uses the standard Ultralytics model.predict(stream=True) API — no forked internals.
"""
import os
import sys
import time

import cv2
import numpy as np

# Ensure pipeline root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO

from config import CATEGORIES, INFERENCE_MODEL_PATH, LIVE_CONFIG
from src.live.tracker import init_tracker, update_tracker, draw_tracks


def run_live_detection(
    source,
    model_path: str = None,
    conf: float = None,
    iou: float = None,
    device: str = None,
    show: bool = True,
    save_video: str = None,
):
    """
    Run real-time object detection + tracking on a video source.

    Parameters
    ----------
    source : int (webcam index, e.g. 0) or str (path to video file).
    model_path : path to .pt weights. Defaults to config.INFERENCE_MODEL_PATH.
    conf : confidence threshold. Defaults to LIVE_CONFIG["conf"].
    iou : NMS IoU threshold. Defaults to LIVE_CONFIG["iou"].
    device : 'cpu', 'mps', 'cuda', or device index. Defaults to LIVE_CONFIG["device"].
    show : display annotated frames in a window.
    save_video : if set, path to write the annotated output video.
    """
    # Resolve defaults from config
    model_path = model_path or INFERENCE_MODEL_PATH
    conf = conf if conf is not None else LIVE_CONFIG["conf"]
    iou = iou if iou is not None else LIVE_CONFIG["iou"]
    device = device if device is not None else LIVE_CONFIG["device"]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ── Init tracker ──────────────────────────────────────────────────────────
    tracker = init_tracker(
        max_age=LIVE_CONFIG["tracker_max_age"],
        min_hits=LIVE_CONFIG["tracker_min_hits"],
        iou_thresh=LIVE_CONFIG["tracker_iou_thresh"],
    )

    # ── Open video source ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_webcam = isinstance(source, int)

    print(f"Source: {'webcam' if is_webcam else source}  ({w}x{h} @ {fps:.1f} FPS)")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")

    # ── Video writer (optional) ───────────────────────────────────────────────
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))
        print(f"Saving output to: {save_video}")

    # ── Frame loop ────────────────────────────────────────────────────────────
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ── Detect ────────────────────────────────────────────────────
            results = model.predict(
                frame, conf=conf, iou=iou, device=device, verbose=False,
            )
            result = results[0]
            boxes = result.boxes

            # Build detection array for SORT: [x1, y1, x2, y2, conf, class_id]
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                clss = boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, confs, clss])
            else:
                dets = np.empty((0, 6))

            # ── Track ─────────────────────────────────────────────────────
            tracked = update_tracker(tracker, dets)

            # ── Draw ──────────────────────────────────────────────────────
            annotated = draw_tracks(
                frame, tracker, tracked, CATEGORIES,
                line_thickness=LIVE_CONFIG.get("line_thickness", 2),
            )

            # ── FPS overlay ───────────────────────────────────────────────
            elapsed = time.time() - t_start
            live_fps = frame_idx / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated,
                f"FPS: {live_fps:.1f}  Frame: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            # ── Display ──────────────────────────────────────────────────
            if show:
                cv2.imshow("ClickTheLook — Live Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n'q' pressed — stopping.")
                    break

            # ── Save ──────────────────────────────────────────────────────
            if writer is not None:
                writer.write(annotated)

            # ── Progress (for video files) ────────────────────────────────
            if not is_webcam and total_frames > 0 and frame_idx % 100 == 0:
                pct = frame_idx / total_frames * 100
                print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)  FPS: {live_fps:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\nDone. {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS avg)")


# ---------------------------------------------------------------------------
# CLI entry (can also be called from scripts/live.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClickTheLook — live detection + tracking")
    parser.add_argument("--source", default=0,
                        help="Video source: 0 for webcam, or path to video file.")
    parser.add_argument("--model", default=None, help="Path to .pt weights.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Device: cpu, mps, cuda, or index.")
    parser.add_argument("--no-show", action="store_true", help="Disable display window.")
    parser.add_argument("--save", default=None, help="Path to save output video.")
    args = parser.parse_args()

    # Coerce source to int if it looks like a webcam index
    src = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    run_live_detection(
        source=src,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=not args.no_show,
        save_video=args.save,
    )
