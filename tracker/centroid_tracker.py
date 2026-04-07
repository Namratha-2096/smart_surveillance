# ─────────────────────────────────────────────
#  tracker/centroid_tracker.py
#  Enhanced with IOU matching + Kalman Filter prediction
#  Based on: Wahyono et al. 2023 (MDPI) — centroid trajectory tracking
# ─────────────────────────────────────────────
import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_id         = 0
        self.objects          = OrderedDict()   # id → centroid (x,y)
        self.bboxes           = OrderedDict()   # id → (x1,y1,x2,y2)
        self.disappeared      = OrderedDict()   # id → frames missing
        self.max_disappeared  = max_disappeared
        self.kalman_filters   = OrderedDict()   # id → cv2.KalmanFilter

    # ── Kalman Filter factory ─────────────────
    def _create_kalman(self, cx, cy):
        """4-state Kalman filter: [x, y, vx, vy]."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        return kf

    # ── IOU computation ───────────────────────
    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)

    # ── public ────────────────────────────────
    def update(self, bboxes):
        """
        bboxes: list of (x1, y1, x2, y2)
        returns: dict  { id: (cx, cy) }
        """
        if len(bboxes) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                # Use Kalman prediction to update position during disappearance
                if oid in self.kalman_filters:
                    pred = self.kalman_filters[oid].predict()
                    self.objects[oid] = (int(pred[0].item()), int(pred[1].item()))
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        input_centroids = np.array([
            ((x1 + x2) // 2, (y1 + y2) // 2)
            for (x1, y1, x2, y2) in bboxes
        ])

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self._register(c, bboxes[i])
        else:
            self._match(input_centroids, bboxes)

        return self.objects

    # ── private ───────────────────────────────
    def _register(self, centroid, bbox):
        oid = self.next_id
        self.objects[oid]      = centroid
        self.bboxes[oid]       = bbox
        self.disappeared[oid]  = 0
        self.kalman_filters[oid] = self._create_kalman(int(centroid[0]), int(centroid[1]))
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        self.bboxes.pop(oid, None)
        self.kalman_filters.pop(oid, None)

    def _match(self, input_centroids, input_bboxes):
        obj_ids        = list(self.objects.keys())
        obj_centroids  = list(self.objects.values())

        # ── Hybrid cost matrix: centroid distance + IOU ──
        D_cent = dist.cdist(np.array(obj_centroids), input_centroids)

        # Normalize centroid distances to 0-1 range for blending
        max_dist = D_cent.max() if D_cent.max() > 0 else 1.0
        D_norm = D_cent / max_dist

        # IOU matrix (higher is better → invert to cost)
        iou_matrix = np.zeros((len(obj_ids), len(input_centroids)))
        for r, oid in enumerate(obj_ids):
            if oid in self.bboxes:
                for c in range(len(input_bboxes)):
                    iou_matrix[r, c] = self._iou(self.bboxes[oid], input_bboxes[c])

        # Hybrid cost: lower is better
        # 60% centroid distance + 40% (1 - IOU)
        cost = 0.6 * D_norm + 0.4 * (1.0 - iou_matrix)

        rows = cost.min(axis=1).argsort()
        cols = cost.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = obj_ids[r]
            cx, cy = input_centroids[c]
            self.objects[oid] = input_centroids[c]
            self.bboxes[oid]  = input_bboxes[c]
            self.disappeared[oid] = 0

            # Kalman correct step
            if oid in self.kalman_filters:
                self.kalman_filters[oid].correct(
                    np.array([[np.float32(cx)], [np.float32(cy)]])
                )

            used_rows.add(r)
            used_cols.add(c)

        unused_rows = set(range(len(obj_centroids))) - used_rows
        unused_cols = set(range(len(input_centroids))) - used_cols

        for r in unused_rows:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            # Kalman predict during occlusion
            if oid in self.kalman_filters:
                pred = self.kalman_filters[oid].predict()
                self.objects[oid] = (int(pred[0].item()), int(pred[1].item()))
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        for c in unused_cols:
            self._register(input_centroids[c], input_bboxes[c])
