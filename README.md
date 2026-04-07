# Smart Surveillance System
### Edge AI for Image Processing — Multimedia Processing & Analysis Project

Real-time Python + OpenCV surveillance system implementing three research-backed detection modules, served through a FastAPI backend and React dashboard.

---

## Features
| Module | Method | Paper Reference |
|---|---|---|
| Face Recognition | Haar Cascade + DeepFace FaceNet + FAISS Vector Search | Khan et al. 2019 (IEEE), PMC 2022 |
| Loitering Detection | YOLOv8 (ONNX) + IOU-Kalman Centroid Tracker + Dual Threshold | Wahyono et al. 2023 (MDPI), IEEE 2022 |
| Fall Detection | YOLOv8-Pose (ONNX) + Multi-Person Aspect Ratio + Keypoints | Chakurkar 2025, Bugarin 2022 (IEEE) |

---

## Architecture

```
┌──────────────┐    MJPEG Stream     ┌──────────────────┐
│  FastAPI      │ ──────────────────► │  React Dashboard │
│  server.py    │ ◄── REST APIs ───── │  (Vite + JSX)    │
│  :8000        │  /api/events,config │  :5173            │
└──────┬───────┘                     └──────────────────┘
       │
┌──────┴───────┐
│  Surveillance │ ← Threaded Camera Reader (zero-lag)
│  Engine       │ ← Frame-skip optimization
│  main.py      │ ← Circular buffer clip saving
└──────┬───────┘
       │
  ┌────┴────┬──────────┐
  ▼         ▼          ▼
Face     Loitering   Fall
Recog    Detector    Detector
(FAISS)  (IOU+KF)   (Multi-person)
```

---

## Setup

```bash
# 1. Clone / download project
cd smart_surveillance

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Export ONNX models (critical for Edge AI performance)
python optimize.py

# 5. Add known faces
#    Create one folder per person inside data/known_faces/
#    e.g.  data/known_faces/Alice/img1.jpg
#          data/known_faces/Bob/img1.jpg

# 6. Run backend
python server.py

# 7. Run dashboard (in another terminal)
cd dashboard
npm install
npm run dev
```

Open **http://localhost:5173** in your browser to see the live surveillance dashboard.

---

## Configuration
All settings are in `config.py` — no other file needs editing. Settings can also be tuned live from the Dashboard's Settings panel.

| Setting | Default | Description |
|---|---|---|
| `SOURCE` | `0` | `0` = webcam, `"video.mp4"` = file |
| `FACE_THRESHOLD` | `0.55` | Lower = stricter recognition |
| `LOITER_TIME_SEC` | `10` | Seconds before loitering alert |
| `LOITER_DISP_PX` | `60` | Max pixel movement to count as stationary |
| `FALL_ASPECT_RATIO` | `1.1` | W/H ratio threshold for fall |
| `FALL_CONFIRM_FRAMES` | `8` | Consecutive frames before confirming fall |
| `ALERT_COOLDOWN_SEC` | `10` | Gap between repeated alerts |
| `FRAME_WIDTH` | `640` | Processing resolution width |
| `FRAME_HEIGHT` | `360` | Processing resolution height |

---

## Project Structure
```
smart_surveillance/
├── server.py                  # FastAPI backend + MJPEG streaming
├── main.py                    # SurveillanceEngine (threaded, frame-skip)
├── config.py                  # All tunable settings
├── database.py                # Thread-safe SQLite with connection pooling
├── optimize.py                # ONNX export script
├── benchmark.py               # PyTorch vs ONNX performance comparison
├── requirements.txt
├── detectors/
│   ├── face_detector.py       # Haar + DeepFace + FAISS vector index
│   ├── loitering_detector.py  # YOLO (ONNX) + IOU-Kalman tracker
│   └── fall_detector.py       # YOLO-Pose (ONNX) + multi-person
├── tracker/
│   ├── centroid_tracker.py    # IOU + Kalman Filter enhanced
│   └── dwell_timer.py         # Time + displacement thresholds
├── utils/
│   ├── draw_utils.py          # OpenCV overlays
│   └── alert_utils.py         # Console alert system
├── dashboard/                 # React + Vite frontend
│   └── src/components/
│       ├── LiveView.jsx       # MJPEG video stream with auto-reconnect
│       ├── EventSidebar.jsx   # Real-time alert feed
│       ├── SettingsPanel.jsx  # Live config tuning sliders
│       └── Header.jsx
├── data/
│   ├── known_faces/           # One subfolder per person with photos
│   └── test_videos/
└── output/
    ├── events.db              # SQLite event log
    ├── clips/                 # 5-second alert video clips
    └── logs/
```

---

## Performance Optimizations
- **ONNX Runtime**: Both YOLOv8 and YOLOv8-Pose are exported to ONNX for 2-3x faster inference on CPU.
- **FAISS Vector Search**: Face recognition uses FAISS `IndexFlatIP` for O(1) embedding lookup instead of O(N×M) brute-force.
- **Frame-Skip Pipeline**: Face runs every 15th frame, loitering every 2nd, fall every 3rd. Intermediate frames reuse cached results.
- **Threaded Camera**: Dedicated reader thread ensures zero-lag capture.
- **Kalman Filter**: Maintains tracking IDs during brief occlusions without re-assignment.
- **IOU Matching**: Hybrid cost matrix (60% centroid + 40% IOU) prevents ID swaps when people cross paths.

---

## Datasets Used
- **Face Recognition** — LFW (Labeled Faces in the Wild) + custom enrolled faces
- **Loitering** — VIRAT Dataset, UCF-Crime
- **Fall Detection** — UR Fall Detection Dataset, Le2i Fall Detection Dataset

---

## References
1. Khan et al. (2019). *Face Detection and Recognition Using OpenCV*. IEEE ICCCIS.
2. PMC (2022). *Automation of Surveillance Systems Using Deep Learning and Facial Recognition*.
3. Mishra et al. (2023). *Face Recognition in Real Time Using OpenCV and Python*. SSRN.
4. Wahyono et al. (2023). *Loitering Detection Using Spatial-Temporal Information*. MDPI JSAN.
5. Bugarin et al. (2022). *Machine Vision-Based Fall Detection Using MediaPipe Pose*. IEEE R10-HTC.
6. Chakurkar & Kothari (2025). *Towards Safer Environments: A YOLO and MediaPipe-Based Fall Detection System*. ScienceDirect.
7. PMC (2024). *Enhancing Elderly Care: Efficient and Reliable Real-Time Fall Detection Algorithm*.
