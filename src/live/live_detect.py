"""
Real-time video detection + tracking using a trained ClickTheLook model.

Uses the standard Ultralytics model.predict() API — no forked internals.
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO

from config import CATEGORIES, INFERENCE_MODEL_PATH, LIVE_CONFIG, TRACKER_BACKEND
from src.live.run_logger import RunLogger, extract_model_info

if TRACKER_BACKEND == "deepsort":
    from src.live.deepsort_tracker import init_tracker, update_tracker, draw_tracks
else:
    from src.live.tracker import init_tracker, update_tracker, draw_tracks


def run_live_detection(
    source,
    model_path: str = None,
    conf: float = None,
    iou: float = None,
    device: str = None,
    show: bool = True,
    save_video: str = None,
    verbose: bool = False,
    enable_logging: bool = True,
):
    """
    Run real-time object detection + tracking on a video source.

    Parameters
    ----------
    source        : int (webcam index) or str (path to video file).
    model_path    : path to .pt weights. Defaults to config.INFERENCE_MODEL_PATH.
    conf          : confidence threshold.
    iou           : NMS IoU threshold.
    device        : 'cpu', 'mps', 'cuda', or device index.
    show          : display annotated frames in a window.
    save_video    : path to write annotated output video.
    verbose       : print per-frame track IDs to stdout.
    enable_logging: write JSON run log to logs/ on completion.
    """
    model_path = model_path or INFERENCE_MODEL_PATH
    conf   = conf   if conf   is not None else LIVE_CONFIG["conf"]
    iou    = iou    if iou    is not None else LIVE_CONFIG["iou"]
    device = device if device is not None else LIVE_CONFIG["device"]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ── Init tracker ──────────────────────────────────────────────────────────
    print(f"Tracker backend: {TRACKER_BACKEND}")
    if TRACKER_BACKEND == "deepsort":
        embedder_gpu = not (isinstance(device, str) and device == "cpu")
        tracker_config = {
            "max_age":             LIVE_CONFIG["deepsort_max_age"],
            "min_hits":            LIVE_CONFIG["deepsort_n_init"],
            "max_cosine_distance": LIVE_CONFIG["deepsort_max_cosine_distance"],
            "embedder":            LIVE_CONFIG["deepsort_embedder"],
            "embedder_gpu":        embedder_gpu,
        }
        tracker = init_tracker(**tracker_config)
    else:
        tracker_config = {
            "max_age":   LIVE_CONFIG["tracker_max_age"],
            "min_hits":  LIVE_CONFIG["tracker_min_hits"],
            "iou_thresh": LIVE_CONFIG["tracker_iou_thresh"],
        }
        tracker = init_tracker(**tracker_config)

    # ── Open video source ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_webcam    = isinstance(source, int)

    print(f"Source: {'webcam' if is_webcam else source}  ({w}x{h} @ {src_fps:.1f} FPS)")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")

    # ── Video writer (optional) ───────────────────────────────────────────────
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, src_fps, (w, h))
        print(f"Saving output to: {save_video}")

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = None
    if enable_logging:
        logger = RunLogger(
            model_path=model_path,
            model_info=extract_model_info(model),
            tracker_backend=TRACKER_BACKEND,
            tracker_config=tracker_config,
            source=source,
            source_meta={
                "type":              "webcam" if is_webcam else "video_file",
                "width":             w,
                "height":            h,
                "source_fps":        src_fps,
                "total_frames_src":  total_frames,
                "duration_s":        round(total_frames / src_fps, 2) if src_fps > 0 else None,
            },
            conf=conf,
            iou=iou,
            device=device,
            save_video=save_video,
        )
        print(f"Logging enabled — run ID: {logger.run_id}")

    # ── Frame loop ────────────────────────────────────────────────────────────
    frame_idx = 0
    t_start   = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ── Detect ────────────────────────────────────────────────────
            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf, iou=iou, device=device, verbose=False)
            t_infer = (time.perf_counter() - t0) * 1000.0

            result = results[0]
            boxes  = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                clss  = boxes.cls.cpu().numpy().reshape(-1, 1)
                dets  = np.hstack([xyxy, confs, clss])
            else:
                dets = np.empty((0, 6))

            # ── Track ─────────────────────────────────────────────────────
            t1 = time.perf_counter()
            tracked = update_tracker(tracker, dets, frame)
            t_track = (time.perf_counter() - t1) * 1000.0

            # ── Draw ──────────────────────────────────────────────────────
            t2 = time.perf_counter()
            annotated = draw_tracks(
                frame, tracker, tracked, CATEGORIES,
                line_thickness=LIVE_CONFIG.get("line_thickness", 2),
            )
            t_draw = (time.perf_counter() - t2) * 1000.0

            # ── Log frame ─────────────────────────────────────────────────
            if logger is not None:
                logger.log_frame(
                    frame_idx=frame_idx,
                    inference_ms=t_infer,
                    tracker_ms=t_track,
                    draw_ms=t_draw,
                    detections=dets,
                    tracked=tracked,
                    class_names=CATEGORIES,
                )

            # ── Verbose per-frame print ───────────────────────────────────
            if verbose:
                if len(tracked) > 0:
                    track_ids = tracked[:, 8].astype(int).tolist()
                    class_ids = tracked[:, 4].astype(int).tolist()
                    labels = [
                        f"ID{tid}:{CATEGORIES.get(cid + 1, str(cid))}"
                        for tid, cid in zip(track_ids, class_ids)
                    ]
                    print(f"  Frame {frame_idx:05d} | dets={len(dets):2d} | tracks={len(tracked):2d} | {', '.join(labels)}")
                else:
                    print(f"  Frame {frame_idx:05d} | dets={len(dets):2d} | tracks=  0 |")

            # ── FPS overlay ───────────────────────────────────────────────
            elapsed  = time.time() - t_start
            live_fps = frame_idx / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated,
                f"FPS: {live_fps:.1f}  Frame: {frame_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            # ── Display ───────────────────────────────────────────────────
            if show:
                cv2.imshow("ClickTheLook — Live Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n'q' pressed — stopping.")
                    break

            # ── Save ──────────────────────────────────────────────────────
            if writer is not None:
                writer.write(annotated)

            # ── Progress (video files) ────────────────────────────────────
            if not is_webcam and total_frames > 0 and frame_idx % 100 == 0:
                pct = frame_idx / total_frames * 100
                print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)  FPS: {live_fps:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\nDone. {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS avg)")

        if logger is not None:
            log_path = logger.finalize(frame_idx, elapsed)
            print(f"Run log  : {log_path}")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ClickTheLook — live detection + tracking")
    parser.add_argument("--source",   default=0,    help="Video source: 0 for webcam, or path to video file.")
    parser.add_argument("--model",    default=None, help="Path to .pt weights.")
    parser.add_argument("--conf",     type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou",      type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--device",   default=None, help="Device: cpu, mps, cuda, or index.")
    parser.add_argument("--no-show",  action="store_true", help="Disable display window.")
    parser.add_argument("--save",     default=None, help="Path to save output video.")
    parser.add_argument("--verbose",  action="store_true", help="Print per-frame track IDs to stdout.")
    parser.add_argument("--no-log",   action="store_true", help="Disable JSON run logging.")
    args = parser.parse_args()

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
        verbose=args.verbose,
        enable_logging=not args.no_log,
    )
