"""
GlobalIdentityManager — re-identifies tracked objects across disappearance gaps.

Maintains a stable global ID per physical object even when the tracker drops
and re-acquires the track (e.g. object leaves frame and returns).

To disable: set LIVE_CONFIG["use_global_ids"] = False in config.py.
To remove entirely: delete this file and set use_global_ids = False.
No other files need changes — live_detect.py guards all calls behind that flag.
"""
import numpy as np


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


class GlobalIdentityManager:
    """
    Wraps raw tracker IDs with a stable global identity layer.

    Parameters
    ----------
    max_gap_s          : seconds a lost identity is kept before expiry.
    cosine_threshold   : max cosine distance to accept a re-id match.
    spatial_gate_pps   : max pixels-per-second drift allowed for a match.
    """

    def __init__(
        self,
        max_gap_s: float = 10.0,
        cosine_threshold: float = 0.15,
        spatial_gate_pps: float = 200.0,
    ):
        self.max_gap_s = max_gap_s
        self.cosine_threshold = cosine_threshold
        self.spatial_gate_pps = spatial_gate_pps

        self._next_gid: int = 1
        self._active: dict = {}
        self._identities: dict = {}

    def update(
        self,
        tracked: np.ndarray,
        embeddings: dict,
        class_names: dict,
        timestamp: float,
    ) -> dict:
        """Call once per frame immediately after update_tracker(). Returns {tracker_id: global_id}."""
        self._expire(timestamp)

        current_tids: set = set()
        id_map: dict = {}

        for row in tracked:
            tid = int(row[8])
            cid = int(row[4])
            name = class_names.get(cid + 1, str(cid))
            cx = (float(row[0]) + float(row[2])) / 2.0
            cy = (float(row[1]) + float(row[3])) / 2.0
            emb = embeddings.get(tid)

            current_tids.add(tid)

            if tid in self._active:
                gid = self._active[tid]
                self._refresh(gid, emb, name, (cx, cy), timestamp)
            else:
                gid = self._match(name, (cx, cy), emb, timestamp)
                if gid is None:
                    gid = self._mint()
                self._active[tid] = gid
                self._refresh(gid, emb, name, (cx, cy), timestamp)

            id_map[tid] = gid

        for tid in list(self._active):
            if tid not in current_tids:
                del self._active[tid]

        return id_map

    def _expire(self, now: float):
        active_gids = set(self._active.values())
        expired = [
            gid
            for gid, e in self._identities.items()
            if gid not in active_gids and (now - e["last_ts"]) > self.max_gap_s
        ]
        for gid in expired:
            del self._identities[gid]

    def _mint(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _refresh(self, gid: int, emb, class_name: str, center: tuple, ts: float):
        entry = self._identities.get(gid)
        if entry is None:
            self._identities[gid] = {
                "emb": emb,
                "class_name": class_name,
                "center": center,
                "last_ts": ts,
            }
        else:
            if emb is not None:
                prev = entry["emb"]
                # EMA on embeddings (0.7/0.3): damp single-frame appearance noise.
                entry["emb"] = emb if prev is None else (0.7 * prev + 0.3 * emb)
            entry["center"] = center
            entry["last_ts"] = ts

    def _match(self, class_name: str, center: tuple, emb, timestamp: float) -> int | None:
        """Return the best matching lost global ID, or None if no match passes gates."""
        active_gids = set(self._active.values())
        best_gid = None
        best_score = self.cosine_threshold  # lower is better; threshold is ceiling

        for gid, entry in self._identities.items():
            if gid in active_gids:
                continue

            if entry["class_name"] != class_name:
                continue

            gap = timestamp - entry["last_ts"]
            if gap > self.max_gap_s:
                continue

            ex, ey = entry["center"]
            dist_px = ((center[0] - ex) ** 2 + (center[1] - ey) ** 2) ** 0.5
            max_drift = self.spatial_gate_pps * max(gap, 0.1)
            if dist_px > max_drift:
                continue

            if emb is None or entry["emb"] is None:
                # Without appearance vectors, only allow re-id on very short occlusions.
                if gap < 2.0 and best_gid is None:
                    best_gid = gid
                continue

            score = _cosine_dist(emb, entry["emb"])
            if score < best_score:
                best_score = score
                best_gid = gid

        return best_gid
