# ─────────────────────────────────────────────
#  main.py  —  Smart Surveillance System
#  Wires face recognition + loitering + fall detection into one live loop.
#  Run:  python main.py
# ─────────────────────────────────────────────
import cv2
import time
import os
import threading
from collections import deque

import config
from database           import db, log_event, fetch_recent
from detectors.face_detector      import FaceDetector
from detectors.loitering_detector import LoiteringDetector
from detectors.fall_detector      import FallDetector
from utils.draw_utils  import (draw_face, draw_person, draw_fall_alert,
                                draw_hud, draw_event_log)
from utils.alert_utils import trigger_alert


class SurveillanceEngine:
    def __init__(self):
        db.init()
        os.makedirs(config.CLIPS_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR,  exist_ok=True)

        self.face_det     = FaceDetector()
        self.loiter_det   = LoiteringDetector()
        self.fall_det     = FallDetector()

        self.cap = cv2.VideoCapture(config.SOURCE)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fps_timer = time.time()
        self.fps       = 0.0
        self.frame_count = 0
        self.is_running = False

        # ── Alert cooldown tracker (instance-level, not global) ──
        self._last_alert: dict[str, float] = {}

        # ── Circular frame buffer for clip saving (~5 sec at 30fps) ──
        self._clip_buffer: deque = deque(maxlen=150)

        # ── Threaded Camera Reader ──
        self.latest_frame = None
        self.latest_ret = False
        self._read_thread = threading.Thread(target=self._update_cam, daemon=True)

    # ── Alert cooldown (moved from global) ────
    def _can_alert(self, key: str) -> bool:
        now = time.time()
        if now - self._last_alert.get(key, 0) >= config.ALERT_COOLDOWN_SEC:
            self._last_alert[key] = now
            return True
        return False

    # ── Clip saving with circular buffer ──────
    def _save_clip(self, frame, event_type):
        if not config.SAVE_CLIPS or len(self._clip_buffer) == 0:
            return ""
        os.makedirs(config.CLIPS_DIR, exist_ok=True)
        fname = os.path.join(
            config.CLIPS_DIR,
            f"{event_type}_{int(time.time())}.avi"
        )
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(
            fname, cv2.VideoWriter_fourcc(*"XVID"), 20, (w, h)
        )
        # Write entire buffer (up to ~5 seconds of context)
        for buffered_frame in self._clip_buffer:
            writer.write(buffered_frame)
        # Write the current alert frame as the last frame
        writer.write(frame)
        writer.release()
        return fname

    # ── Threaded camera reader ────────────────
    def _update_cam(self):
        while self.is_running:
            ret, frame = self.cap.read()
            self.latest_ret = ret
            if ret:
                self.latest_frame = frame
            else:
                time.sleep(0.01)

    # ── Main processing loop ─────────────────
    def run_generator(self):
        self.is_running = True
        self._read_thread.start()

        # Cached results for frame-skipping
        cached_faces    = []
        cached_fall     = False
        cached_fall_sig = {}

        # How often each heavy module runs (from config)
        FACE_EVERY    = config.FACE_SKIP_FRAMES
        LOITER_EVERY  = config.LOITER_SKIP_FRAMES
        FALL_EVERY    = config.FALL_SKIP_FRAMES

        while self.is_running:
            ret, frame = self.latest_ret, self.latest_frame
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = frame.copy()
            self.frame_count += 1

            # Store frame in clip buffer (only every 5th frame to reduce overhead)
            if self.frame_count % 5 == 0:
                self._clip_buffer.append(frame.copy())

            # ── FPS ───────────────────────────────
            now = time.time()
            self.fps = 1.0 / max(now - self.fps_timer, 1e-6)
            self.fps_timer = now
            fc = self.frame_count

            try:
                # ══════════════════════════════════════
                #  MODULE 1 — Face Recognition (heavy — run rarely)
                # ══════════════════════════════════════
                if fc % FACE_EVERY == 0:
                    cached_faces = self.face_det.process(frame)
                    for r in cached_faces:
                        if r["name"] == "SPOOF" and self._can_alert("face_spoof"):
                            trigger_alert("Spoofing", "Liveness check failed — possible photo/screen attack")
                            clip = self._save_clip(frame, "face_spoof")
                            log_event("face_spoof", detail="anti-spoofing triggered", clip_path=clip)
                        elif not r["known"] and r["name"] != "SPOOF" and self._can_alert("face_unknown"):
                            trigger_alert("Intruder", "Unknown face detected")
                            clip = self._save_clip(frame, "face_unknown")
                            log_event("face_unknown", detail="dist=" + str(r["dist"]), clip_path=clip)

                # Always draw cached face labels
                for r in cached_faces:
                    draw_face(frame, r["bbox"], r["name"], r["known"], r["dist"])

                # ══════════════════════════════════════
                #  MODULE 2 — Loitering Detection
                # ══════════════════════════════════════
                if fc % LOITER_EVERY == 0:
                    tracked, bboxes, loiterers, new_alerts = self.loiter_det.process(frame)
                    for oid in new_alerts:
                        key = f"loiter_{oid}"
                        if self._can_alert(key):
                            trigger_alert("Loitering", f"Person ID {oid} stationary for {self.loiter_det.get_dwell_sec(oid)}s")
                            clip = self._save_clip(frame, "loitering")
                            log_event("loitering", detail=f"ID={oid}", clip_path=clip)

                # ══════════════════════════════════════
                #  MODULE 3 — Fall Detection (moderate — run every few frames)
                # ══════════════════════════════════════
                if fc % FALL_EVERY == 0:
                    frame, cached_fall, cached_fall_sig = self.fall_det.process(frame)
                    if cached_fall:
                        draw_fall_alert(frame, cached_fall_sig)
                        if self._can_alert("fall"):
                            trigger_alert("Fall", f"AR={cached_fall_sig['aspect_ratio']} HipY={cached_fall_sig['hip_y']}")
                            clip = self._save_clip(frame, "fall")
                            log_event("fall", detail=str(cached_fall_sig), clip_path=clip)
                else:
                    if cached_fall:
                        draw_fall_alert(frame, cached_fall_sig)

            except Exception as e:
                print(f"[Engine] Error in frame {fc}: {e}")

            # ══════════════════════════════════════
            #  HUD overlay
            # ══════════════════════════════════════
            active = ["Face Recog", "Loitering", "Fall Detect"]
            draw_hud(frame, self.fps, active)

            yield frame

        self.cap.release()

    def stop(self):
        self.is_running = False

def main():
    engine = SurveillanceEngine()
    print("=== Smart Surveillance running — press Q to quit ===")

    for frame in engine.run_generator():
        cv2.imshow("Smart Surveillance — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            engine.stop()
            break

    cv2.destroyAllWindows()
    print(f"=== Stopped. {engine.frame_count} frames processed. ===")

if __name__ == "__main__":
    main()
