# ─────────────────────────────────────────────
#  config.py  —  all tunable settings in one place
#  Change values here; nothing else needs editing
# ─────────────────────────────────────────────

# ── Input ──────────────────────────────────────
SOURCE = 0                        # 0 = webcam  |  "path/to/video.mp4" = file

# ── Face Recognition ───────────────────────────
KNOWN_FACES_DIR   = "data/known_faces"   # one sub-folder per person
FACE_THRESHOLD    = 0.45                 # cosine distance — lower = stricter
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"  # bundled with OpenCV

# ── Loitering Detection ────────────────────────
YOLO_MODEL        = "yolov8n.pt"         # nano = fastest; swap for yolov8s for accuracy
LOITER_TIME_SEC   = 10                   # seconds in zone before alert
LOITER_DISP_PX    = 60                   # max pixel displacement to still count as loitering
MAX_DISAPPEARED   = 30                   # frames before centroid tracker drops an ID

# ── Fall Detection ─────────────────────────────
FALL_ASPECT_RATIO = 1.1                  # W/H ratio — above this = possible fall
FALL_HIP_Y_THRESH = 0.65                 # hip landmark Y (normalised 0-1) — above = low
FALL_CONFIRM_FRAMES = 8                  # must trigger for N consecutive frames

# ── Event Engine ───────────────────────────────
ALERT_COOLDOWN_SEC = 10                  # seconds between repeated alerts for same event
SAVE_CLIPS         = True                # save 5-second clip when alert fires
DB_PATH            = "output/events.db"
CLIPS_DIR          = "output/clips"
LOGS_DIR           = "output/logs"

# ── Display ────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 360
SHOW_FPS     = True
