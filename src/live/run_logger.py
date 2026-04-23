"""
RunLogger — collects per-frame stats during live detection and writes a
structured JSON summary to logs/<YYYYMMDD_HHMMSS>.json on run completion.

Design: all data is accumulated in-memory lists (zero I/O in the hot loop);
numpy aggregation and disk write happen only once in finalize().
"""
import json
import platform
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import torch

    _TORCH_VERSION = torch.__version__
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _GPU_NAME = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None
except Exception:
    _TORCH_VERSION = "unknown"
    _CUDA_AVAILABLE = False
    _GPU_NAME = None

try:
    import ultralytics

    _ULTRALYTICS_VERSION = ultralytics.__version__
except Exception:
    _ULTRALYTICS_VERSION = "unknown"

try:
    import psutil

    _RAM_GB = round(psutil.virtual_memory().total / 1e9, 1)
    _CPU_NAME = platform.processor() or "unknown"
except Exception:
    _RAM_GB = None
    _CPU_NAME = platform.processor() or "unknown"

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

# Subsample timeline rows in JSON so long runs do not produce huge files.
_TIMELINE_SAMPLE_INTERVAL = 30


def extract_model_info(model) -> dict:
    """Pull metadata from an Ultralytics YOLO model safely."""
    info = {
        "task": getattr(model, "task", "detect"),
        "num_classes": len(getattr(model, "names", {})),
        "class_names": getattr(model, "names", {}),
    }
    try:
        result = model.info(verbose=False)
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            info["num_layers"] = int(result[0])
            info["num_params"] = int(result[1])
    except Exception:
        pass
    return info


def _arr_stats(arr) -> dict:
    """Return avg/min/max/p50/p95 for a numeric list."""
    if not arr:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    a = np.array(arr, dtype=float)
    return {
        "avg": round(float(np.mean(a)), 3),
        "min": round(float(np.min(a)), 3),
        "max": round(float(np.max(a)), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
    }


def _frame_to_ts(frame_idx: int, fps: float) -> str:
    """Convert a frame index to a video timestamp string HH:MM:SS.mmm."""
    total_s = frame_idx / max(fps, 1e-6)
    hours = int(total_s // 3600)
    minutes = int((total_s % 3600) // 60)
    seconds = total_s % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _duration_text(seconds: float) -> str:
    """Format a duration as a human-readable string, e.g. '4m 28s' or '45s'."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s" if m > 0 else f"{s}s"


class RunLogger:
    """
    Instantiate once per run. Call log_frame() every frame, finalize() at end.
    """

    def __init__(
        self,
        *,
        model_path: str,
        model_info: dict,
        tracker_backend: str,
        tracker_config: dict,
        source,
        source_meta: dict,
        conf: float,
        iou: float,
        device,
        save_video: str | None,
    ):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._t_start = time.time()
        self._src_fps = max(float(source_meta.get("source_fps", 30.0)), 1e-6)

        self._inference_ms: list[float] = []
        self._tracker_ms: list[float] = []
        self._draw_ms: list[float] = []
        self._dets_per_frame: list[int] = []
        self._tracks_per_frame: list[int] = []
        self._conf_scores: list[float] = []

        self._class_det_counts: dict[str, int] = {}
        self._class_trk_counts: dict[str, int] = {}
        self._track_registry: dict[int, dict] = {}

        self._gid_current: dict[int, dict] = {}
        self._gid_intervals: dict[int, dict] = {}

        self._timeline: list[dict] = []
        self._frame_anomalies: list[dict] = []

        self._meta: dict = {
            "run_id": self.run_id,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "total_duration_s": None,
            "system": {
                "hostname": socket.gethostname(),
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "torch_version": _TORCH_VERSION,
                "ultralytics_version": _ULTRALYTICS_VERSION,
                "cpu": _CPU_NAME,
                "gpu_available": _CUDA_AVAILABLE,
                "gpu_name": _GPU_NAME,
                "ram_gb": _RAM_GB,
            },
            "model": {
                "path": str(model_path),
                "conf_threshold": conf,
                "iou_threshold": iou,
                "device": str(device),
                **model_info,
            },
            "tracker": {
                "backend": tracker_backend,
                **tracker_config,
            },
            "source": {
                "input": str(source),
                "output_video": str(save_video) if save_video else None,
                **source_meta,
            },
            "performance": {},
            "detections": {},
            "tracking": {},
            "anomalies": {},
            "timeline": {},
        }

    def log_frame(
        self,
        frame_idx: int,
        inference_ms: float,
        tracker_ms: float,
        draw_ms: float,
        detections: np.ndarray,
        tracked: np.ndarray,
        class_names: dict,
        id_map: dict | None = None,
    ) -> None:
        n_dets = len(detections)
        n_tracks = len(tracked)
        total_ms = inference_ms + tracker_ms + draw_ms

        self._inference_ms.append(inference_ms)
        self._tracker_ms.append(tracker_ms)
        self._draw_ms.append(draw_ms)
        self._dets_per_frame.append(n_dets)
        self._tracks_per_frame.append(n_tracks)

        if n_dets > 0:
            self._conf_scores.extend(detections[:, 4].tolist())
            for det in detections:
                cid = int(det[5])
                name = class_names.get(cid + 1, f"cls_{cid}")
                self._class_det_counts[name] = self._class_det_counts.get(name, 0) + 1

        if n_tracks > 0:
            for row in tracked:
                tid = int(row[8])
                cid = int(row[4])
                name = class_names.get(cid + 1, f"cls_{cid}")
                self._class_trk_counts[name] = self._class_trk_counts.get(name, 0) + 1
                if tid not in self._track_registry:
                    self._track_registry[tid] = {
                        "class": name,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "gap_count": 0,
                    }
                else:
                    entry = self._track_registry[tid]
                    # Gap frames: SORT/DeepSort kept the track via Kalman while YOLO missed a box.
                    expected_last = entry["last_frame"] + 1
                    if frame_idx > expected_last:
                        entry["gap_count"] += frame_idx - expected_last
                    entry["last_frame"] = frame_idx

        gids_this_frame: set = set()
        if n_tracks > 0:
            for row in tracked:
                tid = int(row[8])
                gid = id_map.get(tid, tid) if id_map else tid
                cid = int(row[4])
                name = class_names.get(cid + 1, f"cls_{cid}")
                gids_this_frame.add(gid)
                if gid not in self._gid_current:
                    self._gid_current[gid] = {
                        "start_frame": frame_idx,
                        "last_frame": frame_idx,
                        "class_name": name,
                    }
                else:
                    self._gid_current[gid]["last_frame"] = frame_idx

        for gid in list(self._gid_current):
            if gid not in gids_this_frame:
                self._close_interval(gid)

        if n_dets == 0 and frame_idx > 1:
            self._frame_anomalies.append(
                {
                    "frame": frame_idx,
                    "type": "zero_detections",
                    "inference_ms": round(inference_ms, 1),
                }
            )
        if total_ms > 500:
            self._frame_anomalies.append(
                {
                    "frame": frame_idx,
                    "type": "slow_frame",
                    "total_ms": round(total_ms, 1),
                }
            )

        if frame_idx % _TIMELINE_SAMPLE_INTERVAL == 0:
            inst_fps = round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0
            self._timeline.append(
                {
                    "frame": frame_idx,
                    "detections": n_dets,
                    "tracks": n_tracks,
                    "inference_ms": round(inference_ms, 1),
                    "tracker_ms": round(tracker_ms, 1),
                    "draw_ms": round(draw_ms, 1),
                    "inst_fps": inst_fps,
                }
            )

    def finalize(self, frames_processed: int, total_elapsed: float) -> str:
        """Compute aggregate stats and write JSON. Returns path to log file."""
        m = self._meta

        m["timestamp_end"] = datetime.now().isoformat()
        m["total_duration_s"] = round(total_elapsed, 2)

        total_ms_per_frame = [
            i + t + d
            for i, t, d in zip(self._inference_ms, self._tracker_ms, self._draw_ms)
        ]

        m["performance"] = {
            "frames_processed": frames_processed,
            "realtime_avg_fps": round(frames_processed / total_elapsed, 2) if total_elapsed > 0 else 0,
            "inference_ms": _arr_stats(self._inference_ms),
            "tracker_ms": _arr_stats(self._tracker_ms),
            "draw_ms": _arr_stats(self._draw_ms),
            "total_pipeline_ms": _arr_stats(total_ms_per_frame),
            "frames_with_zero_dets": self._dets_per_frame.count(0),
            "frames_with_zero_tracks": self._tracks_per_frame.count(0),
            "zero_detection_rate": round(self._dets_per_frame.count(0) / max(frames_processed, 1), 4),
            "bottleneck": self._identify_bottleneck(),
        }

        m["detections"] = {
            "total": sum(self._dets_per_frame),
            "per_frame": _arr_stats(self._dets_per_frame),
            "confidence": _arr_stats(self._conf_scores),
            "per_class_detections": dict(
                sorted(self._class_det_counts.items(), key=lambda x: x[1], reverse=True)
            ),
            "per_class_tracked_frames": dict(
                sorted(self._class_trk_counts.items(), key=lambda x: x[1], reverse=True)
            ),
        }

        lifespans = [v["last_frame"] - v["first_frame"] + 1 for v in self._track_registry.values()]
        track_summary = sorted(
            [
                {
                    "id": tid,
                    "class": v["class"],
                    "first_frame": v["first_frame"],
                    "last_frame": v["last_frame"],
                    "lifespan_frames": v["last_frame"] - v["first_frame"] + 1,
                    "kalman_gap_frames": v["gap_count"],
                }
                for tid, v in self._track_registry.items()
            ],
            key=lambda x: x["lifespan_frames"],
            reverse=True,
        )

        m["tracking"] = {
            "total_unique_ids": len(self._track_registry),
            "tracks_per_frame": _arr_stats(self._tracks_per_frame),
            "track_lifespan_frames": _arr_stats(lifespans) if lifespans else {},
            "total_kalman_gap_frames": sum(v["gap_count"] for v in self._track_registry.values()),
            "longest_track": track_summary[0] if track_summary else None,
            "shortest_track": track_summary[-1] if track_summary else None,
            "track_summary": track_summary,
        }

        slow_frames = [a for a in self._frame_anomalies if a["type"] == "slow_frame"]
        zero_det_seqs = self._consecutive_zero_det_sequences()
        m["anomalies"] = {
            "slow_frames_count": len(slow_frames),
            "slow_frames_threshold_ms": 500,
            "slow_frames": slow_frames[:50],
            "zero_detection_sequences": zero_det_seqs,
            "id_reassignment_events": len(self._track_registry)
            - (max(self._tracks_per_frame) if self._tracks_per_frame else 0),
        }

        m["timeline"] = {
            "sample_interval_frames": _TIMELINE_SAMPLE_INTERVAL,
            "samples": self._timeline,
        }

        for gid in list(self._gid_current):
            self._close_interval(gid)

        identity_summary = []
        for gid, data in sorted(self._gid_intervals.items()):
            total_s = sum(iv["duration_seconds"] for iv in data["intervals"])
            identity_summary.append(
                {
                    "id": gid,
                    "class": data["class_name"],
                    "intervals": data["intervals"],
                    "total_visible_seconds": round(total_s, 3),
                    "total_visible_text": _duration_text(total_s),
                }
            )
        m["identities"] = identity_summary
        self.identity_summary = identity_summary

        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f"{self.run_id}.json"
        with open(log_path, "w") as fh:
            json.dump(m, fh, indent=2)

        return str(log_path)

    def _close_interval(self, gid: int):
        entry = self._gid_current.pop(gid, None)
        if entry is None:
            return
        start_f = entry["start_frame"]
        end_f = entry["last_frame"]
        dur_s = (end_f - start_f) / self._src_fps
        interval = {
            "start_frame": start_f,
            "end_frame": end_f,
            "start_ts": _frame_to_ts(start_f, self._src_fps),
            "end_ts": _frame_to_ts(end_f, self._src_fps),
            "duration_seconds": round(dur_s, 3),
            "duration_text": _duration_text(dur_s),
        }
        if gid not in self._gid_intervals:
            self._gid_intervals[gid] = {"class_name": entry["class_name"], "intervals": []}
        self._gid_intervals[gid]["intervals"].append(interval)

    def _identify_bottleneck(self) -> str:
        """Return which stage dominates wall-clock time."""
        avgs = {
            "inference": np.mean(self._inference_ms) if self._inference_ms else 0,
            "tracker": np.mean(self._tracker_ms) if self._tracker_ms else 0,
            "draw": np.mean(self._draw_ms) if self._draw_ms else 0,
        }
        return max(avgs, key=avgs.get)

    def _consecutive_zero_det_sequences(self) -> list[dict]:
        """Find runs of consecutive zero-detection frames."""
        seqs, in_seq, start = [], False, 0
        for i, n in enumerate(self._dets_per_frame):
            if n == 0 and not in_seq:
                in_seq, start = True, i + 1
            elif n > 0 and in_seq:
                seqs.append({"start_frame": start, "end_frame": i, "length": i - start + 1})
                in_seq = False
        if in_seq:
            seqs.append(
                {
                    "start_frame": start,
                    "end_frame": len(self._dets_per_frame),
                    "length": len(self._dets_per_frame) - start + 1,
                }
            )
        return sorted(seqs, key=lambda x: x["length"], reverse=True)[:20]
