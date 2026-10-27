
import os
import pickle
import cv2
import numpy as np
import yaml
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.build_encodings import EncodingBuilder

def calculate_accuracy(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    enc_path = cfg["paths"].get("encodings_file", "encodings.pkl")
    known_dir = cfg["paths"].get("known_students", "known_students")

    if not os.path.exists(enc_path):
        print(f"❌ Encodings file not found: {enc_path}")
        return

    # Load Database
    with open(enc_path, "rb") as f:
        database = pickle.load(f)
    
    # Calculate Mean Embeddings for matching (same as face_recog.py)
    known_db = {}
    for name, embeddings in database.items():
        if len(embeddings) > 0:
            avg = np.mean(embeddings, axis=0)
            known_db[name] = avg / np.linalg.norm(avg)

    # Initialize InsightFace for testing
    if os.name == 'nt':
        # Add CUDA/cuDNN DLL paths for ONNX Runtime
        dll_paths = [
            os.path.join(os.path.dirname(__file__), "..", "venv", "Lib", "site-packages", "nvidia", "cudnn", "bin"),
            os.path.join(os.path.dirname(__file__), "..", "venv", "Lib", "site-packages", "nvidia", "cublas", "bin"),
            os.path.join(os.path.dirname(__file__), "..", "venv", "Lib", "site-packages", "torch", "lib"),
        ]
        for p in dll_paths:
            p_abs = os.path.abspath(p)
            if os.path.exists(p_abs):
                os.add_dll_directory(p_abs)
                os.environ["PATH"] = p_abs + os.pathsep + os.environ["PATH"]

    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    # Set strict threshold for testing
    if hasattr(face_app, 'models') and 'detection' in face_app.models:
        face_app.models['detection'].det_thresh = 0.2

    total_images = 0
    correct = 0
    wrong = 0
    unknown = 0
    
    threshold = 0.50 # The strict threshold we set earlier
    
    print(f"📊 Starting Accuracy Calculation for {len(known_db)} identities...")
    print(f"🔍 Threshold: {threshold}")

    results = []

    # Loop through folders
    identities = sorted(os.listdir(known_dir))
    for identity_name in tqdm(identities):
        identity_dir = os.path.join(known_dir, identity_name)
        if not os.path.isdir(identity_dir):
            continue
            
        for filename in os.listdir(identity_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            img_path = os.path.join(identity_dir, filename)
            img = cv2.imread(img_path)
            if img is None: continue
            
            total_images += 1
            
            # Extract embedding
            faces = face_app.get(img)
            if not faces:
                unknown += 1
                results.append((identity_name, "None (No Face Detected)"))
                continue
                
            emb = faces[0].normed_embedding
            
            # Match against DB
            best_name = "Unknown"
            best_score = 0.0
            
            for name, master_emb in known_db.items():
                score = float(np.dot(emb, master_emb))
                if score > best_score:
                    best_score = score
                    best_name = name
            
            if best_score < threshold:
                unknown += 1
                results.append((identity_name, f"Unknown (Score: {best_score:.2f})"))
            elif best_name == identity_name:
                correct += 1
            else:
                wrong += 1
                results.append((identity_name, f"WRONG: {best_name} (Score: {best_score:.2f})"))

    # Final Stats
    accuracy = (correct / total_images) * 100 if total_images > 0 else 0
    error_rate = (wrong / total_images) * 100 if total_images > 0 else 0
    unknown_rate = (unknown / total_images) * 100 if total_images > 0 else 0

    print("\n" + "="*40)
    print("📈 ACCURACY REPORT")
    print("="*40)
    print(f"Total Images Tested: {total_images}")
    print(f"Total Identities:    {len(known_db)}")
    print(f"Correct:             {correct} ({accuracy:.2f}%)")
    print(f"Wrong Identity:      {wrong} ({error_rate:.2f}%)")
    print(f"Unknown/Not Found:   {unknown} ({unknown_rate:.2f}%)")
    print("="*40)
    
    if wrong > 0:
        print("\n❌ Top Misidentifications:")
        for actual, pred in results[-10:]: # Show last 10 issues
            if "WRONG" in pred:
                print(f"  • {actual} was seen as {pred}")

if __name__ == "__main__":
    calculate_accuracy()
