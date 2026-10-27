import pandas as pd
import os
import requests
import cv2
import numpy as np
import yaml
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())
from inference.detector import FaceDetector

def download_image(url, save_path):
    if os.path.exists(save_path):
        return True
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def process_student(row, detector, known_dir):
    name = str(row['Student Name']).strip().replace(" ", "_")
    roll = str(row['Roll No.']).strip()
    photo_url = row['Photo']
    
    if not photo_url or not isinstance(photo_url, str) or not photo_url.startswith('http'):
        return
    
    student_dir = os.path.join(known_dir, f"{name}_{roll}")
    os.makedirs(student_dir, exist_ok=True)
    
    temp_img_path = os.path.join(student_dir, "temp_source.jpg")
    if download_image(photo_url, temp_img_path):
        img = cv2.imread(temp_img_path)
        if img is not None:
            # Detect face and save crop
            res = detector.detect(img)
            if len(res['boxes']) > 0:
                # Take the highest scoring face
                idx = np.argmax(res['scores'])
                x1, y1, x2, y2 = res['boxes'][idx].astype(int)
                
                # Add padding
                w, h = x2 - x1, y2 - y1
                pw, ph = int(w * 0.15), int(h * 0.15)
                x1 = max(0, x1 - pw)
                y1 = max(0, y1 - ph)
                x2 = min(img.shape[1], x2 + pw)
                y2 = min(img.shape[0], y2 + ph)
                
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(student_dir, f"{name}_crop.jpg"), crop)
                    # Remove temp source to save space
                    os.remove(temp_img_path)
                    return True
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output2.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of students to import")
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    known_dir = cfg['paths'].get('known_students', 'known_students')
    os.makedirs(known_dir, exist_ok=True)
    
    detector = FaceDetector(args.checkpoint, cfg)
    df = pd.read_csv(args.csv)
    
    if args.limit:
        df = df.head(args.limit)
    
    print(f"📋 Processing {len(df)} students from {args.csv}...")
    
    success_count = 0
    # Using thread pool for faster downloads, but detector is serial (or handle device locking)
    # To keep it simple and safe for CPU/Memory, we'll loop
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if process_student(row, detector, known_dir):
            success_count += 1
            
    print(f"\n✅ Finished! Successfully imported {success_count} student faces.")
    print(f"🔄 Now run 'python app/build_encodings.py' to update the model database.")

if __name__ == "__main__":
    main()
