import cv2
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import config
from main import SurveillanceEngine
from database import fetch_recent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
engine_thread = None
current_frame = None
frame_lock = threading.Lock()

def engine_loop():
    global current_frame
    for frame in engine.run_generator():
        with frame_lock:
            current_frame = frame.copy()

@app.on_event("startup")
def startup_event():
    global engine, engine_thread
    engine = SurveillanceEngine()
    engine_thread = threading.Thread(target=engine_loop, daemon=True)
    engine_thread.start()

@app.on_event("shutdown")
def shutdown_event():
    if engine:
        engine.stop()
    if engine_thread:
        engine_thread.join()

def generate_frames():
    import time
    while True:
        f = None
        with frame_lock:
            if current_frame is not None:
                f = current_frame

        if f is None:
            time.sleep(0.05)
            continue
            
        ret, buffer = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if not ret:
            time.sleep(0.05)
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # limit stream to approx 30fps

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/events")
def get_events(limit: int = 20):
    events = fetch_recent(limit)
    result = []
    for e in events:
        result.append({
            "id": e[0],
            "timestamp": e[1],
            "type": e[2],
            "detail": e[3],
            "clip_path": e[4]
        })
    return {"events": result}

class ConfigModel(BaseModel):
    loiter_time_sec: int
    fall_aspect_ratio: float
    face_threshold: float

@app.get("/api/config")
def get_config():
    return {
        "loiter_time_sec": config.LOITER_TIME_SEC,
        "fall_aspect_ratio": config.FALL_ASPECT_RATIO,
        "face_threshold": config.FACE_THRESHOLD
    }

@app.post("/api/config")
def update_config(new_config: ConfigModel):
    # Update global config module
    config.LOITER_TIME_SEC = new_config.loiter_time_sec
    config.FALL_ASPECT_RATIO = new_config.fall_aspect_ratio
    config.FACE_THRESHOLD = new_config.face_threshold
    
    # ── Hot-reload: push changes into live detector instances ──
    if engine:
        # Loitering detector — update dwell timer threshold
        engine.loiter_det.dwell.time_thresh = new_config.loiter_time_sec
        # Face detector — threshold is read from config at runtime, no action needed
        # Fall detector — thresholds are imported at module level, already updated via config.*
    
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
