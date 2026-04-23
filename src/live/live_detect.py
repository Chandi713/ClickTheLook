"""
Real-time video detection + tracking using a trained ClickTheLook model.

Uses the standard Ultralytics model.predict() API — no forked internals.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from ultralytics import YOLO

from config import CATEGORIES, INFERENCE_MODEL_PATH, LIVE_CONFIG, TRACKER_BACKEND
from src.live.run_logger import RunLogger, extract_model_info

if TRACKER_BACKEND == "deepsort":
    from src.live.deepsort_tracker import init_tracker, update_tracker, draw_tracks
else:
    from src.live.tracker import init_tracker, update_tracker, draw_tracks

_DETECTIONS_DIR = Path(__file__).resolve().parents[2] / "detections"
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


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
    conf = conf if conf is not None else LIVE_CONFIG["conf"]
    iou = iou if iou is not None else LIVE_CONFIG["iou"]
    device = device if device is not None else LIVE_CONFIG["device"]

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Tracker backend: {TRACKER_BACKEND}")
    if TRACKER_BACKEND == "deepsort":
        import torch as _torch

        embedder_gpu = _torch.cuda.is_available()  # DeepSORT embedder is CUDA-only; MPS/CPU → False
        tracker_config = {
            "max_age": LIVE_CONFIG["deepsort_max_age"],
            "min_hits": LIVE_CONFIG["deepsort_n_init"],
            "max_cosine_distance": LIVE_CONFIG["deepsort_max_cosine_distance"],
            "embedder": LIVE_CONFIG["deepsort_embedder"],
            "embedder_gpu": embedder_gpu,
        }
        tracker = init_tracker(**tracker_config)
    else:
        tracker_config = {
            "max_age": LIVE_CONFIG["tracker_max_age"],
            "min_hits": LIVE_CONFIG["tracker_min_hits"],
            "iou_thresh": LIVE_CONFIG["tracker_iou_thresh"],
        }
        tracker = init_tracker(**tracker_config)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_webcam = isinstance(source, int)

    print(f"Source: {'webcam' if is_webcam else source}  ({w}x{h} @ {src_fps:.1f} FPS)")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, src_fps, (w, h))
        print(f"Saving output to: {save_video}")

    logger = None
    if enable_logging:
        logger = RunLogger(
            model_path=model_path,
            model_info=extract_model_info(model),
            tracker_backend=TRACKER_BACKEND,
            tracker_config=tracker_config,
            source=source,
            source_meta={
                "type": "webcam" if is_webcam else "video_file",
                "width": w,
                "height": h,
                "source_fps": src_fps,
                "total_frames_src": total_frames,
                "duration_s": round(total_frames / src_fps, 2) if src_fps > 0 else None,
            },
            conf=conf,
            iou=iou,
            device=device,
            save_video=save_video,
        )
        print(f"Logging enabled — run ID: {logger.run_id}")

    gid_manager = None
    if TRACKER_BACKEND == "deepsort" and LIVE_CONFIG.get("use_global_ids", False):
        from src.live.global_id import GlobalIdentityManager

        gid_manager = GlobalIdentityManager(
            max_gap_s=LIVE_CONFIG.get("gid_max_gap_s", 10.0),
            cosine_threshold=LIVE_CONFIG.get("gid_cosine_threshold", 0.15),
            spatial_gate_pps=LIVE_CONFIG.get("gid_spatial_gate_pps", 200.0),
        )
        print("Global re-ID: enabled")

    frame_idx = 0
    t_start = time.time()
    best_crops: dict = {}
    id_map: dict = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf, iou=iou, device=device, verbose=False)
            t_infer = (time.perf_counter() - t0) * 1000.0

            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1, 1)
                clss = boxes.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, confs, clss])
            else:
                dets = np.empty((0, 6))

            t1 = time.perf_counter()
            tracked = update_tracker(tracker, dets, frame)
            t_track = (time.perf_counter() - t1) * 1000.0

            if gid_manager is not None:
                embeddings = tracker.get_embeddings() if hasattr(tracker, "get_embeddings") else {}
                id_map = gid_manager.update(tracked, embeddings, CATEGORIES, time.time())
            else:
                id_map = {}

            if len(tracked) > 0:
                fh, fw = frame.shape[:2]
                for row in tracked:
                    x1c = max(0, int(row[0]))
                    y1c = max(0, int(row[1]))
                    x2c = min(fw, int(row[2]))
                    y2c = min(fh, int(row[3]))
                    tid = int(row[8])
                    gid = id_map.get(tid, tid)
                    area = (x2c - x1c) * (y2c - y1c)
                    if area > best_crops.get(gid, {}).get("area", -1) and area > 0:
                        best_crops[gid] = {
                            "area": area,
                            "crop": frame[y1c:y2c, x1c:x2c].copy(),
                            "class_name": CATEGORIES.get(int(row[4]) + 1, str(int(row[4]))),
                        }

            t2 = time.perf_counter()
            annotated = draw_tracks(
                frame,
                tracker,
                tracked,
                CATEGORIES,
                line_thickness=LIVE_CONFIG.get("line_thickness", 2),
                id_map=id_map or None,
            )
            t_draw = (time.perf_counter() - t2) * 1000.0

            if logger is not None:
                logger.log_frame(
                    frame_idx=frame_idx,
                    inference_ms=t_infer,
                    tracker_ms=t_track,
                    draw_ms=t_draw,
                    detections=dets,
                    tracked=tracked,
                    class_names=CATEGORIES,
                    id_map=id_map or None,
                )

            if verbose:
                if len(tracked) > 0:
                    track_ids = tracked[:, 8].astype(int).tolist()
                    class_ids = tracked[:, 4].astype(int).tolist()
                    labels = [
                        f"G{id_map.get(tid, tid)}:{CATEGORIES.get(cid + 1, str(cid))}"
                        for tid, cid in zip(track_ids, class_ids)
                    ]
                    print(
                        f"  Frame {frame_idx:05d} | dets={len(dets):2d} | tracks={len(tracked):2d} | "
                        f"{', '.join(labels)}"
                    )
                else:
                    print(f"  Frame {frame_idx:05d} | dets={len(dets):2d} | tracks=  0 |")

            elapsed = time.time() - t_start
            live_fps = frame_idx / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated,
                f"FPS: {live_fps:.1f}  Frame: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if show:
                cv2.imshow("ClickTheLook — Live Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n'q' pressed — stopping.")
                    break

            if writer is not None:
                writer.write(annotated)

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

        final_crops: dict = {}

        if best_crops:
            blur_thresh = LIVE_CONFIG.get("dedup_blur_threshold", 100.0)
            phash_thresh = LIVE_CONFIG.get("dedup_phash_threshold", 8)

            valid = {
                gid: info
                for gid, info in best_crops.items()
                if info["crop"] is not None and info["crop"].size > 0
            }
            blurry_count = sum(1 for info in valid.values() if _blur_score(info["crop"]) < blur_thresh)
            valid = {
                gid: info
                for gid, info in valid.items()
                if _blur_score(info["crop"]) >= blur_thresh
            }

            # Largest-area first: when dHash marks a near-duplicate, the bigger crop is kept.
            candidates = sorted(valid.items(), key=lambda x: x[1]["area"], reverse=True)
            kept_hashes = []
            final_crops = {}
            near_dupes = 0
            for gid, info in candidates:
                h = _dhash(info["crop"])
                if any(bin(h ^ kh).count("1") < phash_thresh for kh in kept_hashes):
                    near_dupes += 1
                else:
                    kept_hashes.append(h)
                    final_crops[gid] = info

            run_id = logger.run_id if logger is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
            det_dir = _DETECTIONS_DIR / run_id
            det_dir.mkdir(parents=True, exist_ok=True)
            for gid, info in final_crops.items():
                cv2.imwrite(str(det_dir / f"{gid}_{info['class_name']}.jpg"), info["crop"])

            parts = [f"{len(final_crops)} saved"]
            if blurry_count:
                parts.append(f"{blurry_count} blurry removed")
            if near_dupes:
                parts.append(f"{near_dupes} near-duplicate(s) removed")
            print(f"Crops    : {', '.join(parts)} → {det_dir}")

        if logger is not None:
            log_path = logger.finalize(frame_idx, elapsed)
            print(f"Run log  : {log_path}")

            run_id = logger.run_id
            det_dir = _DETECTIONS_DIR / run_id
            out_data = []
            for entry in logger.identity_summary:
                gid = entry["id"]
                cls = entry["class"]
                crop_file = det_dir / f"{gid}_{cls}.jpg"
                if not crop_file.exists():
                    continue
                if entry["total_visible_seconds"] < LIVE_CONFIG.get("output_min_visible_s", 3.0):
                    continue
                out_data.append(
                    {
                        "id": gid,
                        "class": cls,
                        "crop_path": str(crop_file),
                        "num_intervals": len(entry["intervals"]),
                        "intervals": entry["intervals"],
                        "total_visible_seconds": entry["total_visible_seconds"],
                        "total_visible_text": entry["total_visible_text"],
                    }
                )
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = _OUTPUT_DIR / f"{run_id}.json"
            with open(out_path, "w") as fh:
                json.dump(out_data, fh, indent=2)
            print(f"Output   : {out_path}")


def _blur_score(img_bgr: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Low value means blurry."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _dhash(img_bgr: np.ndarray, size: int = 8) -> int:
    """Difference hash: 64-bit perceptual fingerprint of an image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum(1 << i for i, v in enumerate(diff.flatten()) if v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClickTheLook — live detection + tracking")
    parser.add_argument("--source", default=0, help="Video source: 0 for webcam, or path to video file.")
    parser.add_argument("--model", default=None, help="Path to .pt weights.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Device: cpu, mps, cuda, or index.")
    parser.add_argument("--no-show", action="store_true", help="Disable display window.")
    parser.add_argument("--save", default=None, help="Path to save output video.")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame track IDs to stdout.")
    parser.add_argument("--no-log", action="store_true", help="Disable JSON run logging.")
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
