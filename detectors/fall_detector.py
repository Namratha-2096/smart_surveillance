# ─────────────────────────────────────────────
#  detectors/fall_detector.py
#  Signal 1: Bounding box aspect ratio W/H > threshold  
#  Signal 2: Hip landmark Y-position confirmation       
#  Both signals must agree for N consecutive frames
#  Supports MULTI-PERSON fall detection
# ─────────────────────────────────────────────
import cv2
import os
from ultralytics import YOLO
from config import FALL_ASPECT_RATIO, FALL_HIP_Y_THRESH, FALL_CONFIRM_FRAMES

class FallDetector:
    # YOLOv8 COCO pose indices
    _LEFT_HIP       = 11
    _RIGHT_HIP      = 12

    def __init__(self, use_onnx=True):
        model_path = "yolov8n-pose.pt"
        if use_onnx and os.path.exists(model_path.replace('.pt', '.onnx')):
            model_path = model_path.replace('.pt', '.onnx')
            
        self.model = YOLO(model_path)
        # Per-person consecutive fall frame counters
        self._consec_fall_frames: dict[int, int] = {}
        print(f"[FallDetector] YOLO-Pose loaded: {model_path}")

    def process(self, frame):
        results = self.model(frame, classes=[0], verbose=False)[0]
        fall_detected = False
        signals = {"aspect_ratio": "?", "hip_y": "?", "pose_ok": False}

        if (results.keypoints is not None 
            and len(results.keypoints.xyn) > 0 
            and len(results.boxes.xyxyn) > 0):

            signals["pose_ok"] = True
            num_persons = min(len(results.keypoints.xyn), len(results.boxes.xyxyn))

            # Iterate over ALL detected persons
            for pid in range(num_persons):
                xyn = results.keypoints.xyn[pid]
                if len(xyn) <= self._RIGHT_HIP:
                    continue

                box = results.boxes.xyxyn[pid]
                w = float(box[2] - box[0])
                h = float(box[3] - box[1])
                aspect_ratio = (w / h) if h > 0 else 0

                hip_y = (float(xyn[self._LEFT_HIP][1]) + float(xyn[self._RIGHT_HIP][1])) / 2.0

                if aspect_ratio > FALL_ASPECT_RATIO and hip_y > FALL_HIP_Y_THRESH:
                    self._consec_fall_frames[pid] = self._consec_fall_frames.get(pid, 0) + 1
                else:
                    self._consec_fall_frames[pid] = max(0, self._consec_fall_frames.get(pid, 0) - 1)

                if self._consec_fall_frames.get(pid, 0) >= FALL_CONFIRM_FRAMES:
                    fall_detected = True
                    signals["aspect_ratio"] = round(aspect_ratio, 2)
                    signals["hip_y"] = round(hip_y, 2)

            # Clean up counters for persons no longer detected
            active_pids = set(range(num_persons))
            stale = [k for k in self._consec_fall_frames if k not in active_pids]
            for k in stale:
                del self._consec_fall_frames[k]

            frame = results.plot(boxes=False, labels=False)
        else:
            self._consec_fall_frames.clear()

        return frame, fall_detected, signals
