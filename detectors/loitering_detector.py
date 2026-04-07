# ─────────────────────────────────────────────
#  detectors/loitering_detector.py
#  Technique: YOLOv8 person detection  (Wahyono et al. 2023 — MDPI)
#           + Centroid tracker (IOU + Kalman enhanced)
#           + Dual threshold           (time AND displacement)
# ─────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ultralytics import YOLO
from tracker.centroid_tracker import CentroidTracker
from tracker.dwell_timer import DwellTimer
from config import YOLO_MODEL, LOITER_TIME_SEC, LOITER_DISP_PX, MAX_DISAPPEARED


class LoiteringDetector:
    def __init__(self, use_onnx=True):
        model_path = YOLO_MODEL
        if use_onnx and os.path.exists(model_path.replace('.pt', '.onnx')):
            model_path = model_path.replace('.pt', '.onnx')
            print(f"[LoiteringDetector] YOLOv8 loaded with ONNX backend: {model_path}")
        else:
            print(f"[LoiteringDetector] YOLOv8 loaded with PyTorch backend: {model_path}")
            
        self.model   = YOLO(model_path)
        self.tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED)
        self.dwell   = DwellTimer(
            time_thresh_sec=LOITER_TIME_SEC,
            disp_thresh_px =LOITER_DISP_PX
        )

    # ── public ────────────────────────────────
    def process(self, frame):
        """
        Returns:
          tracked  : { id: (cx, cy) }
          bboxes   : { id: (x1,y1,x2,y2) }
          loiterers: list of IDs flagged as loitering
          new_alerts: list of IDs alerting for the first time this session
        """
        bboxes_list = self._detect_persons(frame)
        # Pass raw bboxes to tracker (IOU + Kalman enhanced)
        tracked    = self.tracker.update(bboxes_list)
        loiterers  = self.dwell.update(tracked)
        new_alerts = [oid for oid in loiterers if self.dwell.is_new_alert(oid)]

        # Bboxes are now stored inside the tracker
        bboxes = {}
        for oid in tracked:
            if oid in self.tracker.bboxes:
                bboxes[oid] = self.tracker.bboxes[oid]

        return tracked, bboxes, loiterers, new_alerts

    def get_dwell_sec(self, oid):
        return self.dwell.get_dwell_sec(oid)

    # ── private ───────────────────────────────
    def _detect_persons(self, frame):
        results = self.model(frame, classes=[0], verbose=False)[0]  # class 0 = person
        bboxes_list = []
        for box in results.boxes.xyxy.tolist():
            x1, y1, x2, y2 = map(int, box)
            bboxes_list.append((x1, y1, x2, y2))
        return bboxes_list
