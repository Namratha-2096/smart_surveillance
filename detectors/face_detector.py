# ─────────────────────────────────────────────
#  detectors/face_detector.py
#  Technique 1: Haar Cascade detection  (Khan et al. 2019 — IEEE)
#  Technique 2: DeepFace cosine embedding (PMC 2022)
#  Technique 3: Threshold tuning for FAR/FRR (Mishra et al. 2023)
#  Optimization: FAISS vector index for O(1) face lookup
# ─────────────────────────────────────────────
import cv2
import os
import numpy as np
from deepface import DeepFace
from config import KNOWN_FACES_DIR, FACE_THRESHOLD, HAAR_CASCADE_PATH

# Try to use FAISS for fast vector search, fall back to brute-force
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaceDetector:
    def __init__(self):
        # Haar cascade — fast, lightweight (Khan et al. 2019)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + HAAR_CASCADE_PATH
        )
        # Build known-face database from data/known_faces/<person_name>/*.jpg
        self.known_db = self._build_db()
        self.last_faces = []  # cache to prevent DeepFace frame-lock

        # Build FAISS index for sub-millisecond face lookup
        self._faiss_index = None
        self._faiss_labels = []  # maps FAISS row → person name
        if HAS_FAISS and self.known_db:
            self._build_faiss_index()

        print(f"[FaceDetector] Loaded {len(self.known_db)} known identities. "
              f"FAISS={'enabled' if self._faiss_index else 'disabled'}")

    # ── public ────────────────────────────────
    def process(self, frame):
        """
        Returns list of dicts:
          { 'bbox': (x,y,w,h), 'name': str, 'known': bool, 'dist': float }
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces   = self.cascade.detectMultiScale(
                      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        results = []

        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2

            # Check cache to avoid massive DeepFace lag on every frame
            matched = False
            for lf in self.last_faces:
                lcx, lcy = lf['centroid']
                if abs(cx - lcx) < 50 and abs(cy - lcy) < 50:
                    results.append({
                        "bbox": (x, y, w, h),
                        "name": lf["name"],
                        "known": lf["known"],
                        "dist": lf["dist"]
                    })
                    matched = True
                    break

            if not matched:
                face_crop = frame[y:y+h, x:x+w]
                name, dist, known = self._recognize(face_crop)
                results.append({
                    "bbox": (x, y, w, h),
                    "name": name,
                    "known": known,
                    "dist": dist
                })

        # Update cache buffer
        self.last_faces = []
        for r in results:
            bx, by, bw, bh = r["bbox"]
            self.last_faces.append({
                "name": r["name"],
                "known": r["known"],
                "dist": r["dist"],
                "centroid": (bx + bw // 2, by + bh // 2)
            })

        return results

    # ── FAISS index builder ───────────────────
    def _build_faiss_index(self):
        """Build a FAISS inner-product index from all known embeddings."""
        all_embeddings = []
        self._faiss_labels = []
        for name, embeddings in self.known_db.items():
            for emb in embeddings:
                # L2-normalize for cosine similarity via inner product
                norm = np.linalg.norm(emb) + 1e-9
                all_embeddings.append(emb / norm)
                self._faiss_labels.append(name)

        if len(all_embeddings) == 0:
            return

        matrix = np.array(all_embeddings, dtype=np.float32)
        dim = matrix.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)  # Inner Product = cosine on normalized vecs
        self._faiss_index.add(matrix)

    # ── private ───────────────────────────────
    def _build_db(self):
        """
        Loads embeddings for every image in data/known_faces/<name>/*.jpg|png
        Returns: { name: [embedding, ...] }
        """
        db = {}
        if not os.path.isdir(KNOWN_FACES_DIR):
            print(f"[FaceDetector] Warning: {KNOWN_FACES_DIR} not found.")
            return db
        for person in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person)
            if not os.path.isdir(person_dir):
                continue
            embeddings = []
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                try:
                    emb = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]
                    embeddings.append(np.array(emb))
                except Exception as e:
                    print(f"[FaceDetector] Skipping {img_path}: {e}")
            if embeddings:
                db[person] = embeddings
        return db

    def _recognize(self, face_crop):
        """
        Compare face_crop against known_db.
        Uses FAISS if available, otherwise brute-force cosine distance.
        Mishra 2023: threshold directly controls FAR/FRR trade-off.
        """
        if not self.known_db:
            return "Unknown", 1.0, False
        try:
            query_emb = np.array(
                DeepFace.represent(
                    img_path=face_crop,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]
            )
        except Exception:
            return "Unknown", 1.0, False

        # ── FAISS path (O(1) lookup) ──────────
        if self._faiss_index is not None:
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            query_norm = query_norm.astype(np.float32).reshape(1, -1)
            similarities, indices = self._faiss_index.search(query_norm, 1)
            best_sim = float(similarities[0][0])
            best_dist = 1.0 - best_sim  # convert similarity to cosine distance
            best_name = self._faiss_labels[int(indices[0][0])]
            known = best_dist <= FACE_THRESHOLD
            return (best_name if known else "Unknown"), round(best_dist, 3), known

        # ── Brute-force fallback (O(N×M)) ─────
        best_name, best_dist = "Unknown", float("inf")
        for name, embeddings in self.known_db.items():
            for ref_emb in embeddings:
                cos_dist = 1 - np.dot(query_emb, ref_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(ref_emb) + 1e-9
                )
                if cos_dist < best_dist:
                    best_dist = cos_dist
                    best_name = name

        known = best_dist <= FACE_THRESHOLD
        return (best_name if known else "Unknown"), round(best_dist, 3), known
