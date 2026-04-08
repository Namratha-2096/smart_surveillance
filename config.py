# ─────────────────────────────────────────────
#  config.py  —  all tunable settings in one place
#  Change values here; nothing else needs editing
# ─────────────────────────────────────────────

# ── Input ──────────────────────────────────────
SOURCE = 0                        # 0 = webcam  |  "path/to/video.mp4" = file

# ── Face Recognition ───────────────────────────
KNOWN_FACES_DIR        = "data/known_faces"      # one sub-folder per person
FACE_THRESHOLD         = 0.42                    # cosine distance — lower = stricter (with confidence margin check)
FACE_DETECTOR_BACKEND  = "ssd"                   # deep-learning detector (fast + accurate, replaces Haar)
ENFORCE_DETECTION      = False                   # prevents crash if no face is in frame
ENABLE_LIVENESS        = False                   # anti-spoofing — set True only if you have a GPU

# ── Loitering Detection ────────────────────────
YOLO_MODEL        = "yolov8n.pt"         # nano = fastest; swap for yolov8s for accuracy
LOITER_TIME_SEC   = 20                   # seconds in zone before alert
LOITER_DISP_PX    = 60                   # max pixel displacement to still count as loitering
MAX_DISAPPEARED   = 30                   # frames before centroid tracker drops an ID

# ── Fall Detection ─────────────────────────────
FALL_ASPECT_RATIO = 1.1                  # W/H ratio — above this = possible fall
FALL_HIP_Y_THRESH = 0.65                 # hip landmark Y (normalised 0-1) — above = low
FALL_CONFIRM_FRAMES = 8                  # must trigger for N consecutive frames

# ── Frame Skipping (performance optimization) ──
FACE_SKIP_FRAMES     = 30   # run face detection every 30th frame (~1s at 30fps)
LOITER_SKIP_FRAMES   = 5    # run loitering every 5th frame
FALL_SKIP_FRAMES     = 8    # run fall detection every 8th frame

# ── Processing Resolution ────────────────────
FRAME_WIDTH  = 640  # keep this — optimal balance of speed vs accuracy
FRAME_HEIGHT = 360  # keep this small for performance

# ── Event Engine ─────────────────────────────
ALERT_COOLDOWN_SEC = 20                  # seconds between repeated alerts for same event
SAVE_CLIPS         = False               # set True for production, False for testing (avoids disk I/O lag)
DB_PATH            = "output/events.db"
CLIPS_DIR          = "output/clips"
LOGS_DIR           = "output/logs"
