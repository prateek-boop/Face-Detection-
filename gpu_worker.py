"""
High-Speed GPU AI Inference Service (PyTorch CUDA + InsightFace)

Powered by NVIDIA GeForce RTX 4070 SUPER GPU.
Features:
- Dual GPU Face Detection (PyTorch FP16 + InsightFace RetinaFace)
- Padded facial landmark embedding extraction with adaptive thresholding
- In-memory vectorized cosine similarity over 4,767 student identities
- Full metadata resolution (Name, Roll, Branch) from StudentMetadataDB
"""

import os
import sys
import time
import yaml
import cv2
import pickle
import numpy as np
import torch
import csv
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add CUDA / cuDNN DLL paths for Windows ONNX Runtime & PyTorch
if os.name == 'nt':
    dll_paths = [
        os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages", "torch", "lib"),
    ]
    for p in dll_paths:
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            try:
                os.add_dll_directory(p_abs)
                os.environ["PATH"] = p_abs + os.pathsep + os.environ["PATH"]
            except Exception:
                pass

from inference.detector import FaceDetector
from insightface.app import FaceAnalysis

print("[GPU Worker] Initializing on NVIDIA GeForce RTX 4070 SUPER...")

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 1. Load PyTorch Face Detector on CUDA (FP16)
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = FaceDetector("checkpoints/latest.pt", cfg, device=device)
print(f"[GPU Worker] PyTorch FaceDetector active on: {device.upper()} (FP16={detector.fp16})")

# 2. Load InsightFace with CUDA Execution Provider
try:
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    print("[GPU Worker] InsightFace GPU Embedding Engine ready (ctx_id=0)")
except Exception as e:
    face_app = None
    print(f"[GPU Worker] InsightFace fallback warning: {e}")

# 3. Load Student Metadata DB
class StudentMetadataDB:
    def __init__(self, csv_paths):
        self._by_roll = {}
        self._by_name = {}
        for path in csv_paths:
            if not Path(path).exists():
                continue
            self._load(path)
        print(f"[GPU Worker] MetaDB loaded {len(self._by_roll)} roll records, {sum(len(v) for v in self._by_name.values())} name records")

    def _col(self, fields, candidates):
        for c in candidates:
            for f in fields:
                if f.lower() == c.lower():
                    return f
        return None

    def _load(self, path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return
            fields = [c.strip() for c in reader.fieldnames]
            name_col = self._col(fields, ("Student Name", "Name", "name"))
            roll_col = self._col(fields, ("Roll No.", "Roll Number", "Roll", "UID", "ID", "Reg No."))
            branch_col = self._col(fields, ("Branch", "Stream", "Course", "Department"))
            for raw in reader:
                row = {k.strip(): (v or "").strip() for k, v in raw.items()}
                record = {
                    "Name": row.get(name_col, "N/A") if name_col else "N/A",
                    "Roll": row.get(roll_col, "N/A") if roll_col else "N/A",
                    "Branch": row.get(branch_col, "N/A") if branch_col else "N/A",
                    **{k: v for k, v in row.items()},
                }
                roll = record["Roll"]
                name = record["Name"]
                if roll and roll != "N/A":
                    self._by_roll[roll] = record
                if name and name != "N/A":
                    first_token = name.split()[0].lower() if name.split() else ""
                    if first_token:
                        self._by_name.setdefault(first_token, []).append(record)

    def lookup(self, identity_string):
        parts = [p for p in identity_string.split("_") if p]
        if not parts:
            return {"Name": "Unknown", "Roll": "N/A", "Branch": "N/A"}
        first = parts[0]
        last = parts[-1] if len(parts) > 1 else ""

        if last and last in self._by_roll:
            return self._by_roll[last]
        candidates = self._by_name.get(first.lower(), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            for rec in candidates:
                if rec.get("Roll", "") == last:
                    return rec
            return candidates[0]
        return {"Name": " ".join(parts[:-1]) if len(parts) > 1 else first, "Roll": last or "N/A", "Branch": "N/A"}

meta_db = StudentMetadataDB(["output.csv", "output2.csv"])

# 4. Load Encodings Database
enc_db = {}
if os.path.exists("encodings.pkl"):
    try:
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
        for name, embeddings in data.items():
            if len(embeddings) > 0:
                arr = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-7
                enc_db[name] = arr / norms
        print(f"[GPU Worker] Loaded {len(enc_db)} student identities from encodings.pkl")
    except Exception as e:
        print(f"[GPU Worker] Encodings load error: {e}")

IDENTIFY_THRESHOLD = 0.40

def extract_face_embedding(frame, box):
    """Extract 512-d normalized embedding with margin padding & adaptive det_thresh."""
    if face_app is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    pad = int(max(x2 - x1, y2 - y1) * 0.25)
    
    crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None

    faces = face_app.get(crop)
    if faces and len(faces) > 0:
        return faces[0].normed_embedding

    # Adaptive fallback with lower detection threshold
    try:
        det_model = face_app.models.get("detection")
        if det_model is not None:
            old_thresh = det_model.det_thresh
            det_model.det_thresh = 0.20
            try:
                faces = face_app.get(crop)
                if faces and len(faces) > 0:
                    return faces[0].normed_embedding
            finally:
                det_model.det_thresh = old_thresh
    except Exception:
        pass
    return None

def match_embedding(query_emb):
    if query_emb is None or len(enc_db) == 0:
        return "Unknown", 0.0
    best_name = "Unknown"
    best_sim = 0.0
    for name, master_embs in enc_db.items():
        sim = float(np.max(np.dot(master_embs, query_emb)))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name, best_sim

class InferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/infer":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            t0 = time.perf_counter()
            nparr = np.frombuffer(body, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid image payload"}')
                return

            h, w = img.shape[:2]

            # 1. Run GPU Face Detection
            det_res = detector.detect(img)
            boxes = det_res.get("boxes", np.zeros((0, 4)))
            scores = det_res.get("scores", np.zeros((0,)))

            # If no boxes from PyTorch detector, run InsightFace detection on full image
            if len(boxes) == 0 and face_app is not None:
                try:
                    detected_faces = face_app.get(img)
                    if detected_faces:
                        boxes = np.array([f.bbox for f in detected_faces])
                        scores = np.array([getattr(f, 'det_score', 0.95) for f in detected_faces])
                except Exception:
                    pass

            faces_out = []

            for i, box in enumerate(boxes):
                score = float(scores[i]) if i < len(scores) else 0.9
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))

                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                emb = extract_face_embedding(img, [x1, y1, x2, y2])
                raw_identity, similarity = match_embedding(emb)

                is_known = similarity >= IDENTIFY_THRESHOLD
                if is_known:
                    meta = meta_db.lookup(raw_identity)
                    student_name = meta.get("Name", raw_identity)
                    roll_no = meta.get("Roll", "N/A")
                    branch = meta.get("Branch", "N/A")
                else:
                    raw_identity = "Unknown"
                    student_name = "Unknown"
                    roll_no = "N/A"
                    branch = "N/A"

                faces_out.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "identity": raw_identity,
                    "student_name": student_name,
                    "roll_no": roll_no,
                    "branch": branch,
                    "similarity": similarity,
                    "is_known": is_known,
                })

            latency = (time.perf_counter() - t0) * 1000.0

            response = {
                "status": "success",
                "faces": faces_out,
                "latency_ms": latency,
                "frame_width": w,
                "frame_height": h,
                "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }

            resp_bytes = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "gpu_worker_ready", "device": "cuda"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def run_server(port=5000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, InferenceHandler)
    print(f"[GPU Worker] HTTP AI Server running on http://127.0.0.1:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
