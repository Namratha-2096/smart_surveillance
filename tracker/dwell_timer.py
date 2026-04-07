# ─────────────────────────────────────────────
#  tracker/dwell_timer.py
#  Based on: Wahyono et al. 2023 — dual threshold (time + displacement)
#  Tracks how long each person ID has been present and how much they moved.
# ─────────────────────────────────────────────
import time
import numpy as np


class DwellTimer:
    def __init__(self, time_thresh_sec=8, disp_thresh_px=60):
        self.time_thresh = time_thresh_sec
        self.disp_thresh = disp_thresh_px

        self._start_time  = {}   # id → time.time() when first seen
        self._start_pos   = {}   # id → (cx,cy) when first seen
        self._alerted     = set()

    def update(self, tracked_objects: dict):
        """
        tracked_objects: { id: (cx, cy) }  from CentroidTracker
        returns: list of IDs currently loitering (dual threshold satisfied)
        """
        now = time.time()
        loiterers = []

        for oid, (cx, cy) in tracked_objects.items():
            if oid not in self._start_time:
                self._start_time[oid] = now
                self._start_pos[oid]  = (cx, cy)

            elapsed = now - self._start_time[oid]
            sx, sy  = self._start_pos[oid]
            disp    = np.hypot(cx - sx, cy - sy)   # euclidean displacement

            if elapsed >= self.time_thresh and disp <= self.disp_thresh:
                loiterers.append(oid)

        # clean up IDs that disappeared
        gone = set(self._start_time) - set(tracked_objects)
        for oid in gone:
            self._start_time.pop(oid, None)
            self._start_pos.pop(oid, None)
            self._alerted.discard(oid)

        return loiterers

    def is_new_alert(self, oid: int) -> bool:
        """Returns True only the first time an ID crosses the threshold."""
        if oid not in self._alerted:
            self._alerted.add(oid)
            return True
        return False

    def get_dwell_sec(self, oid: int) -> float:
        if oid in self._start_time:
            return round(time.time() - self._start_time[oid], 1)
        return 0.0
