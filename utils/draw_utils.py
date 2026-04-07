# ─────────────────────────────────────────────
#  utils/draw_utils.py  —  OpenCV overlay helpers
# ─────────────────────────────────────────────
import cv2


# colour palette (BGR)
GREEN  = (0,  200,  80)
RED    = (0,   40, 220)
ORANGE = (0,  140, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,    0,   0)
YELLOW = (0,  220, 220)


def draw_face(frame, bbox, name, known, dist):
    x, y, w, h = bbox
    color = GREEN if known else RED
    label = f"{name}  d={dist}" if known else f"Unknown  d={dist}"
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    _label_box(frame, label, (x, y - 6), color)


def draw_person(frame, bbox, oid, dwell_sec, loitering):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    color = ORANGE if loitering else GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {oid}  {dwell_sec}s{'  LOITERING' if loitering else ''}"
    _label_box(frame, label, (x1, y1 - 6), color)


def draw_fall_alert(frame, signals):
    ar  = signals.get("aspect_ratio", "?")
    hy  = signals.get("hip_y", "?")
    txt = f"FALL DETECTED  AR={ar}  HipY={hy}"
    # red banner across the top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), RED, -1)
    cv2.putText(frame, txt, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)


def draw_hud(frame, fps, active_modules):
    h, w = frame.shape[:2]
    # semi-transparent bottom bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    mods = "  |  ".join(active_modules)
    cv2.putText(frame, f"FPS: {fps:.1f}   {mods}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_event_log(frame, events, max_lines=5):
    """Overlay last N events in top-right corner."""
    h, w = frame.shape[:2]
    y = 20
    for ev in events[-max_lines:]:
        ts, etype, detail = ev[1], ev[2], ev[3]
        txt = f"[{ts[-8:]}] {etype}: {detail}"
        cv2.putText(frame, txt, (w - 420, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
        y += 20


# ── internal ──────────────────────────────────
def _label_box(frame, text, pos, color):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 4), color, -1)
    cv2.putText(frame, text, (x + 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
