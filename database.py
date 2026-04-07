# ─────────────────────────────────────────────
#  database.py  —  SQLite event logging
#  Uses a persistent connection with thread-safe locking
# ─────────────────────────────────────────────
import sqlite3
import os
import threading
from datetime import datetime
from config import DB_PATH


class _Database:
    """Thread-safe persistent SQLite connection (connection pooling)."""

    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()

    def init(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT    NOT NULL,
                event_type TEXT    NOT NULL,
                detail     TEXT,
                clip_path  TEXT
            )
        """)
        self._conn.commit()

    def log_event(self, event_type: str, detail: str = "", clip_path: str = ""):
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (timestamp, event_type, detail, clip_path) VALUES (?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, detail, clip_path)
            )
            self._conn.commit()

    def fetch_recent(self, limit: int = 50):
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, timestamp, event_type, detail, clip_path FROM events ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return rows

    def close(self):
        if self._conn:
            self._conn.close()


# ── Module-level singleton ────────────────────
db = _Database()


# ── Backward-compatible free functions ────────
def init_db():
    db.init()

def log_event(event_type: str, detail: str = "", clip_path: str = ""):
    db.log_event(event_type, detail, clip_path)

def fetch_recent(limit: int = 50):
    return db.fetch_recent(limit)
