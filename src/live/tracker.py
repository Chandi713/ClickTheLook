"""
Tracker wrapper: initialisation, update, and drawing utilities for SORT.
"""
import cv2
import numpy as np
from random import randint

from src.live.sort import Sort


def init_tracker(max_age: int = 5, min_hits: int = 2, iou_thresh: float = 0.2) -> Sort:
    """Create and return a SORT tracker instance."""
    return Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)


# Pre-generate a palette of random colours for track IDs
_TRACK_COLORS = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(5000)]


def update_tracker(tracker: Sort, detections: np.ndarray, frame=None) -> np.ndarray:
    """
    Feed detections into SORT and return tracked objects.

    Parameters
    ----------
    tracker : Sort instance
    detections : np.ndarray of shape (N, 6) — [x1, y1, x2, y2, conf, class_id]
                 Pass np.empty((0, 6)) if no detections this frame.
    frame : unused — accepted for interface compatibility with deepsort_tracker.

    Returns
    -------
    np.ndarray of shape (M, 9) — [x1, y1, x2, y2, class, u_dot, v_dot, s_dot, track_id]
                 or empty array if no tracks.
    """
    if detections is None or len(detections) == 0:
        detections = np.empty((0, 6))
    return tracker.update(detections)


def draw_tracks(
    frame: np.ndarray,
    tracker: Sort,
    tracked_dets: np.ndarray,
    class_names: dict,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes with track IDs and centroid trails on the frame.

    Parameters
    ----------
    frame : BGR image (modified in-place and returned).
    tracker : Sort instance (used to read centroid history).
    tracked_dets : output of update_tracker().
    class_names : dict mapping int class_id -> str name (e.g. config.CATEGORIES).
    line_thickness : box border width.
    """
    # Draw centroid trails for all active tracks
    for track in tracker.getTrackers():
        color = _TRACK_COLORS[track.id % len(_TRACK_COLORS)]
        for i in range(len(track.centroidarr) - 1):
            pt1 = (int(track.centroidarr[i][0]), int(track.centroidarr[i][1]))
            pt2 = (int(track.centroidarr[i + 1][0]), int(track.centroidarr[i + 1][1]))
            cv2.line(frame, pt1, pt2, color, thickness=3)

    # Draw boxes + labels
    if len(tracked_dets) > 0:
        bbox_xyxy = tracked_dets[:, :4]
        track_ids = tracked_dets[:, 8]
        class_ids = tracked_dets[:, 4]

        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(v) for v in box]
            tid = int(track_ids[i])
            cid = int(class_ids[i])
            # class_names uses 1-indexed keys (CATEGORIES), class_id from YOLO is 0-indexed
            name = class_names.get(cid + 1, str(cid))
            label = f"{tid} {name}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 253), line_thickness)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), (255, 144, 30), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame
