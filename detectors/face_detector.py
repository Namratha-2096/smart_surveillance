# ─────────────────────────────────────────────
#  detectors/face_detector.py
#  Technique 1: SSD deep-learning detection (replaces Haar Cascade)
#  Technique 2: DeepFace/Facenet cosine embedding (PMC 2022)
#  Technique 3: Threshold tuning for FAR/FRR (Mishra et al. 2023)
#  Technique 4: Liveness / anti-spoofing detection (optional)
#  Optimization: FAISS vector index for O(1) face lookup
# ─────────────────────────────────────────────
import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from config import (
    KNOWN_FACES_DIR, FACE_THRESHOLD,
    FACE_DETECTOR_BACKEND, ENFORCE_DETECTION, ENABLE_LIVENESS,
)

# Try to use FAISS for fast vector search, fall back to brute-force
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaceDetector:
    def __init__(self):
        # Build known-face database from data/known_faces/<person_name>/*.jpg
        self.known_db = self._build_db()
        self.last_faces = []  # centroid cache to avoid re-running embedding

        # Build FAISS index for sub-millisecond face lookup
        self._faiss_index = None
        self._faiss_labels = []  # maps FAISS row → person name
        if HAS_FAISS and self.known_db:
            self._build_faiss_index()

        print(f"[FaceDetector] Backend: {FACE_DETECTOR_BACKEND.upper()} | "
              f"Liveness: {'ON' if ENABLE_LIVENESS else 'OFF'} | "
              f"Threshold: {FACE_THRESHOLD}")
        print(f"[FaceDetector] Loaded {len(self.known_db)} known identities. "
              f"FAISS={'enabled' if self._faiss_index else 'disabled'}")

    # ── public ────────────────────────────────
    def process(self, frame):
        """
        Returns list of dicts:
          { 'bbox': (x,y,w,h), 'name': str, 'known': bool, 'dist': float }
        Pipeline: Detection → Alignment → (Liveness) → Embedding → Matching
        """
        results = []
        t0 = time.perf_counter()

        # ── Step 1: Detection + Alignment via DeepFace ──
        try:
            extract_kwargs = dict(
                img_path=frame,
                detector_backend=FACE_DETECTOR_BACKEND,
                enforce_detection=ENFORCE_DETECTION,
                align=True,  # auto-rotate faces so eyes are level
            )
            # Only pass anti_spoofing if liveness is enabled
            if ENABLE_LIVENESS:
                extract_kwargs["anti_spoofing"] = True

            face_objs = DeepFace.extract_faces(**extract_kwargs)
        except Exception as e:
            # No faces or detection failure — return empty
            if "Face could not be detected" not in str(e):
                print(f"[FaceDetector] Detection error: {e}")
            return results

        for face_obj in face_objs:
            # ── Step 2: Liveness check ──
            if ENABLE_LIVENESS:
                is_real = face_obj.get("is_real", True)
                if not is_real:
                    region = face_obj.get("facial_area", {})
                    x = region.get("x", 0)
                    y = region.get("y", 0)
                    w = region.get("w", 100)
                    h = region.get("h", 100)
                    results.append({
                        "bbox": (x, y, w, h),
                        "name": "SPOOF",
                        "known": False,
                        "dist": 0.0,
                    })
                    continue

            # ── Step 3: Extract bbox from DeepFace result ──
            region = face_obj.get("facial_area", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 100)
            h = region.get("h", 100)
            confidence = face_obj.get("confidence", 0.0)

            # Skip very low-confidence detections
            if confidence < 0.5:
                continue

            cx, cy = x + w // 2, y + h // 2

            # ── Step 4: Check centroid cache to skip embedding ──
            matched = False
            for lf in self.last_faces:
                lcx, lcy = lf['centroid']
                if abs(cx - lcx) < 50 and abs(cy - lcy) < 50:
                    results.append({
                        "bbox": (x, y, w, h),
                        "name": lf["name"],
                        "known": lf["known"],
                        "dist": lf["dist"],
                    })
                    matched = True
                    break

            if not matched:
                # ── Step 5: Embedding + Matching ──
                # Crop from ORIGINAL frame with generous padding.
                # Let DeepFace.represent() with SSD handle detection+alignment
                # on this crop — consistent with how enrollment images were processed.
                pad = int(max(w, h) * 0.25)
                fh, fw = frame.shape[:2]
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(fw, x + w + pad)
                y2 = min(fh, y + h + pad)
                face_crop = frame[y1:y2, x1:x2]

                name, dist, known = self._recognize(face_crop)
                results.append({
                    "bbox": (x, y, w, h),
                    "name": name,
                    "known": known,
                    "dist": dist,
                })

        # Update centroid cache
        self.last_faces = []
        for r in results:
            bx, by, bw, bh = r["bbox"]
            self.last_faces.append({
                "name": r["name"],
                "known": r["known"],
                "dist": r["dist"],
                "centroid": (bx + bw // 2, by + bh // 2),
            })

        dt = (time.perf_counter() - t0) * 1000
        if results:
            names = ", ".join(f"{r['name']}({r['dist']})" for r in results)
            print(f"[FaceDetector] {len(results)} face(s) [{names}] in {dt:.0f}ms")

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
                        detector_backend=FACE_DETECTOR_BACKEND,
                        enforce_detection=ENFORCE_DETECTION,
                        align=True,
                    )[0]["embedding"]  # type: ignore
                    embeddings.append(np.array(emb))
                except Exception as e:
                    print(f"[FaceDetector] Skipping {img_path}: {e}")
            if embeddings:
                db[person] = embeddings
                print(f"  → {person}: {len(embeddings)} embedding(s)")
        return db

    def _recognize(self, face_crop):
        """
        Compare face_crop against known_db.
        Uses FAISS if available, otherwise brute-force cosine distance.
        Added confidence margin: best match must be significantly better than second best.
        """
        if not self.known_db:
            return "Unknown", 1.0, False
        try:
            query_emb = np.array(
                DeepFace.represent(
                    img_path=face_crop,
                    model_name="Facenet",
                    detector_backend=FACE_DETECTOR_BACKEND,  # Re-detect in crop to ensure proper eye alignment
                    enforce_detection=False,
                    align=True,
                )[0]["embedding"]  # type: ignore
            )
        except Exception:
            return "Unknown", 1.0, False

        all_distances = []
        if self._faiss_index is not None and self._faiss_index.ntotal > 0:
            # ── FAISS: Fast O(1) Lookup ─────
            norm = np.linalg.norm(query_emb) + 1e-9
            q_norm = (query_emb / norm).astype(np.float32).reshape(1, -1)
            
            # Fetch more neighbors (e.g., 30) so we can group by person correctly
            k = min(30, self._faiss_index.ntotal)
            similarities, indices = self._faiss_index.search(q_norm, k)
            
            for i in range(k):
                idx = indices[0][i]
                if idx != -1:
                    # Convert Cosine Similarity back to Cosine Distance for your threshold logic
                    cos_dist = 1.0 - similarities[0][i]
                    name = self._faiss_labels[idx]
                    all_distances.append((cos_dist, name))
        else:
            # ── Brute-force: Fallback if FAISS is missing ─────
            for name, embeddings in self.known_db.items():
                for ref_emb in embeddings:
                    cos_dist = 1 - np.dot(query_emb, ref_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(ref_emb) + 1e-9
                    )
                    all_distances.append((cos_dist, name))

        # ── Group by person to find their best matching image ──
        person_best = {}
        for dist, name in all_distances:
            if name not in person_best or dist < person_best[name]:
                person_best[name] = dist
        
        # Convert to list and sort by distance (ascending)
        unique_distances = [(dist, name) for name, dist in person_best.items()]
        unique_distances.sort()

        if not unique_distances:
            return "Unknown", 1.0, False

        if len(unique_distances) < 2:
            # Only one person in the database (or only one matched)
            best_dist, best_name = unique_distances[0]
            known = best_dist <= FACE_THRESHOLD
            return (best_name if known else "Unknown"), round(best_dist, 3), known

        # Get top 2 distinct matches
        best_dist, best_name = unique_distances[0]
        second_dist, second_name = unique_distances[1]

        # ── Confidence margin check ──
        # Margin to ensure the best match is distinctly better than the second best
        confidence_margin = 0.05
        
        if best_dist <= FACE_THRESHOLD and (second_dist - best_dist) >= confidence_margin:
            # Clear winner
            print(f"[DEBUG] Best: {best_name}({round(best_dist, 3)}), Second: {second_name}({round(second_dist, 3)}), Margin: {round(second_dist - best_dist, 3)}")
            return best_name, round(best_dist, 3), True
        else:
            # Too close to second match or threshold exceeded - reject
            print(f"[DEBUG] Rejected: Best: {best_name}({round(best_dist, 3)}), Second: {second_name}({round(second_dist, 3)}), Margin: {round(second_dist - best_dist, 3)}")
            return "Unknown", round(best_dist, 3), False
