"""
DeepSort tracker wrapper — drop-in replacement for src/live/tracker.py.

To disable DeepSort: set TRACKER_BACKEND = "sort" in config.py.
To remove DeepSort entirely: delete this file and set TRACKER_BACKEND = "sort".
No other files need changes.
"""
import cv2
import numpy as np
from random import randint

from deep_sort_realtime.deepsort_tracker import DeepSort


_TRACK_COLORS = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(5000)]


class DeepSortWrapper:
    """Wraps DeepSort and maintains centroid history for trail drawing."""

    def __init__(self, tracker: DeepSort):
        self._tracker = tracker
        self.centroid_history: dict = {}  # track_id -> list of (cx, cy)

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        detections : (N, 6) array  [x1, y1, x2, y2, conf, class_id]
        frame      : BGR image — required for appearance embedding

        Returns
        -------
        (M, 9) array  [x1, y1, x2, y2, class_id, 0, 0, 0, track_id]
        Columns 5-7 are zero-padded to match SORT output shape.
        """
        # Convert xyxy -> xywh expected by deep_sort_realtime
        raw = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            raw.append(([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls)))

        tracks = self._tracker.update_tracks(raw, frame=frame)

        results = []
        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = int(track.track_id)
            active_ids.add(tid)
            x1, y1, x2, y2 = track.to_ltrb()
            cls = track.get_det_class() if track.get_det_class() is not None else 0

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.centroid_history.setdefault(tid, []).append((cx, cy))

            results.append([x1, y1, x2, y2, cls, 0.0, 0.0, 0.0, tid])

        # Prune history for tracks that are no longer active
        for tid in list(self.centroid_history):
            if tid not in active_ids:
                del self.centroid_history[tid]

        return np.array(results, dtype=float) if results else np.empty((0, 9))


def init_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_thresh: float = 0.2,
    max_cosine_distance: float = 0.3,
    embedder: str = "mobilenet",
    embedder_gpu: bool = True,
) -> DeepSortWrapper:
    """Create and return a DeepSortWrapper instance."""
    ds = DeepSort(
        max_age=max_age,
        n_init=min_hits,
        nms_max_overlap=1.0,
        max_cosine_distance=max_cosine_distance,
        nn_budget=None,
        embedder=embedder,
        half=True,
        bgr=True,
        embedder_gpu=embedder_gpu,
    )
    return DeepSortWrapper(ds)


def update_tracker(
    tracker: DeepSortWrapper,
    detections: np.ndarray,
    frame: np.ndarray = None,
) -> np.ndarray:
    """Feed detections + frame into DeepSort and return tracked objects."""
    if detections is None or len(detections) == 0:
        detections = np.empty((0, 6))
    return tracker.update(detections, frame)


def draw_tracks(
    frame: np.ndarray,
    tracker: DeepSortWrapper,
    tracked_dets: np.ndarray,
    class_names: dict,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes, track IDs, and centroid trails on frame."""
    # for tid, history in tracker.centroid_history.items():
    #     color = _TRACK_COLORS[tid % len(_TRACK_COLORS)]
    #     for i in range(len(history) - 1):
    #         pt1 = (int(history[i][0]), int(history[i][1]))
    #         pt2 = (int(history[i + 1][0]), int(history[i + 1][1]))
    #         cv2.line(frame, pt1, pt2, color, thickness=3)

    if len(tracked_dets) > 0:
        bbox_xyxy = tracked_dets[:, :4]
        class_ids = tracked_dets[:, 4]
        track_ids = tracked_dets[:, 8]

        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(v) for v in box]
            tid = int(track_ids[i])
            cid = int(class_ids[i])
            name = class_names.get(cid + 1, str(cid))
            label = f"{tid} {name}"

            (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 253), line_thickness)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), (255, 144, 30), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame
