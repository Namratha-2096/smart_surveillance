# ─────────────────────────────────────────────
#  utils/alert_utils.py  —  non-blocking console alerts
#  Tkinter removed to prevent Tcl_AsyncDelete crash on Windows
# ─────────────────────────────────────────────
import threading


def trigger_alert(event_type: str, detail: str = ""):
    """Fire alert in a background thread so it never blocks the video loop."""
    t = threading.Thread(target=_alert_worker, args=(event_type, detail), daemon=True)
    t.start()


def _alert_worker(event_type: str, detail: str):
    msg = f"[ALERT] {event_type.upper()}: {detail}"
    print("\a" + msg)          # terminal bell + console log
