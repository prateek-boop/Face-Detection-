"""
High-Performance Asynchronous Webcam Client for Rust Face Server

Features:
- Dedicated capture & render thread: Runs at full 30-60 FPS with ZERO video lag
- Background async inference worker: Streams frames to Rust server in non-blocking queue
- Real-time HUD telemetry: Displays Live Camera FPS and Server Inference Latency
- High-visibility corner-bracket target boxes with student metadata
"""

import cv2
import requests
import time
import sys
import threading
import queue
import argparse
import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from datetime import datetime

SERVER_URL = "http://127.0.0.1:8080/api/v1/recognize"
CAMERA_ID = 1
CAMERA_NAME = "Webcam Gate Feed"
SCREENSHOT_DIR = Path("outputs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
screenshot_cooldowns = {}
SCREENSHOT_COOLDOWN = 60.0
REENTRY_ALERT_MINUTES = 20.0

def load_today_attendance():
    records = {}
    att_path = Path("attendance.csv")
    if att_path.exists():
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            import csv
            with open(att_path, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) >= 5 and row[4].startswith(today):
                        name = row[1].strip()
                        roll = row[3].strip()
                        key = f"{name}_{roll}" if (roll and roll != "N/A") else name
                        try:
                            dt = datetime.strptime(row[4].strip(), "%Y-%m-%d %H:%M:%S")
                            if key not in records:
                                records[key] = dt.timestamp()
                        except Exception:
                            pass
        except Exception:
            pass
    return records

entry_records = load_today_attendance()
entry_screenshots_taken = set()
reentry_screenshots_taken = set()

def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, corner_len=18):
    """Draw sleek corner brackets around bounding box."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    # Top-left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

class AsyncInferenceWorker:
    def __init__(self, server_url):
        self.server_url = server_url
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_result = None
        self.latest_latency = None
        self.latest_error = None
        self.lock = threading.Lock()
        self.running = True
        self.session = requests.Session()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="InferenceWorker")
        self.thread.start()

    def submit_frame(self, frame):
        """Submit frame non-blockingly; replaces older unprocessed frame."""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    def get_results(self):
        with self.lock:
            return self.latest_result, self.latest_latency, self.latest_error

    def _worker_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Scale to fast inference size (640px)
            h, w = frame.shape[:2]
            scale = 640.0 / max(w, h)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
            else:
                small = frame

            _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])

            try:
                t0 = time.perf_counter()
                files = {'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                data = {'camera_id': str(CAMERA_ID), 'camera_name': CAMERA_NAME}
                resp = self.session.post(self.server_url, files=files, data=data, timeout=1.0)
                latency = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    res_json = resp.json()
                    analysis = res_json.get('analysis', {})
                    # Rescale bounding boxes back to original camera resolution
                    if scale < 1.0 and 'faces' in analysis:
                        for face in analysis['faces']:
                            if 'bbox' in face:
                                b = face['bbox']
                                b['x1'] /= scale
                                b['y1'] /= scale
                                b['x2'] /= scale
                                b['y2'] /= scale
                    
                    with self.lock:
                        self.latest_result = analysis
                        self.latest_latency = latency
                        self.latest_error = None
                else:
                    with self.lock:
                        self.latest_latency = latency
                        self.latest_error = f"HTTP {resp.status_code}"
            except Exception as error:
                with self.lock:
                    self.latest_latency = None
                    self.latest_error = str(error)[:48]

    def stop(self):
        self.running = False

def main():
    parser = argparse.ArgumentParser(description="Webcam Client for Face Server")
    parser.add_argument("--reset-attendance", action="store_true",
                        help="Reset/archive today's attendance CSV for a clean session")
    args, _ = parser.parse_known_args()

    if args.reset_attendance:
        att_path = Path("attendance.csv")
        if att_path.exists():
            import shutil
            b_path = Path(f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            try:
                shutil.copyfile(att_path, b_path)
                print(f"[Webcam] Backed up attendance to {b_path}")
            except Exception:
                pass
            with open(att_path, "w", encoding="utf-8") as f:
                f.write("Sr.no,Name,Branch,Roll Number,Timestamp\n")
            print("[Webcam] Reset attendance.csv to clean header.")
            global entry_records, entry_screenshots_taken, reentry_screenshots_taken
            entry_records = {}
            entry_screenshots_taken.clear()
            reentry_screenshots_taken.clear()

    print("=" * 65)
    print("[Webcam] High-Performance Real-Time Face Recognition Client")
    print(f"[Webcam] Connected to Server: {SERVER_URL}")
    print("[Webcam] Press 'q' or 'ESC' to exit")
    print("=" * 65)

    # Open camera with DirectShow for instant startup on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    worker = AsyncInferenceWorker(SERVER_URL)

    fps = 0.0
    frame_times = []

    try:
        while True:
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read frame from webcam.")
                break

            h, w = frame.shape[:2]

            # Submit frame asynchronously to background worker
            worker.submit_frame(frame)

            # Get latest server analysis (non-blocking)
            analysis, server_lat, server_error = worker.get_results()

            if analysis and 'faces' in analysis:
                for face in analysis['faces']:
                    bbox = face.get('bbox', {})
                    x1 = int(bbox.get('x1', 0))
                    y1 = int(bbox.get('y1', 0))
                    x2 = int(bbox.get('x2', 0))
                    y2 = int(bbox.get('y2', 0))

                    # Clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    is_known = face.get('is_known', False)
                    name = face.get('student_name') or face.get('identity', 'Unknown')
                    roll = face.get('roll_no', '')
                    branch = face.get('branch', '')
                    score = face.get('similarity', 0.0)

                    # Check 20-minute re-entry alert
                    now_ts = time.time()
                    is_reentry = False
                    elapsed_min = 0.0
                    if is_known:
                        entry_key = f"{name}_{roll}" if (roll and roll != "N/A") else name
                        if entry_key not in entry_records:
                            entry_records[entry_key] = now_ts
                        else:
                            elapsed_min = (now_ts - entry_records[entry_key]) / 60.0
                            is_reentry = (elapsed_min >= REENTRY_ALERT_MINUTES)

                    if is_reentry:
                        color = (0, 0, 255)  # RED: Re-entry alert after 20 minutes!
                    elif is_known:
                        color = (0, 255, 128)  # GREEN: Normal verified entry
                    else:
                        color = (0, 165, 255)  # Orange for unknown

                    draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2)

                    # Info badge
                    if is_reentry:
                        label_title = f"{name} [RE-ENTRY]"
                        label_sub = f"Alert: Entered {int(elapsed_min)}m ago!"
                    else:
                        label_title = f"{name}"
                        label_sub = f"Roll: {roll} | {branch}" if roll and roll != "N/A" else f"Conf: {score:.1%}"
                    
                    cv2.rectangle(frame, (x1, max(0, y1 - 42)), (x1 + 240, y1), (20, 20, 20), -1)
                    cv2.rectangle(frame, (x1, max(0, y1 - 42)), (x1 + 240, y1), color, 1)

                    cv2.putText(frame, label_title, (x1 + 6, max(14, y1 - 24)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1)
                    cv2.putText(frame, label_sub, (x1 + 6, max(28, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

                    # Proof screenshot capture: ONLY for KNOWN faces
                    # Normal entry: 1 screenshot when verified
                    # Re-entry alert: 1 screenshot when 20m re-entry triggers
                    should_capture = False
                    if is_known:
                        if is_reentry:
                            if entry_key not in reentry_screenshots_taken:
                                reentry_screenshots_taken.add(entry_key)
                                should_capture = True
                        else:
                            if entry_key not in entry_screenshots_taken:
                                entry_screenshots_taken.add(entry_key)
                                should_capture = True

                    if should_capture:
                        ss_frame = frame.copy()
                        ss_color = color
                        cv2.rectangle(ss_frame, (x1, y1), (x2, y2), ss_color, 2)
                        if is_reentry:
                            ss_label = f"{name} ({roll}) [RE-ENTRY {int(elapsed_min)}m]" if (roll and roll != "N/A") else f"{name} [RE-ENTRY {int(elapsed_min)}m]"
                        else:
                            ss_label = f"{name} ({roll})" if (roll and roll != "N/A") else name
                        cv2.putText(ss_frame, ss_label, (x1, max(y1 - 10, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ss_color, 2)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
                        safe_roll = "".join(c for c in str(roll) if c.isalnum() or c in ("-", "_")).strip()
                        if is_reentry:
                            filename = f"{safe_name}_{safe_roll}_REENTRY_{timestamp}.jpg"
                        else:
                            filename = f"{safe_name}_{safe_roll}_{timestamp}.jpg"
                        cv2.imwrite(str(SCREENSHOT_DIR / filename), ss_frame)
                        if is_reentry:
                            print(f"[Webcam] ⚠️ RE-ENTRY ALERT: {name} ({roll}) entered {int(elapsed_min)}m ago! Screenshot: {filename}")
                        else:
                            print(f"[Webcam] Proof screenshot saved with decision box: {filename}")

            # Calculate smooth rolling FPS
            frame_times.append(time.perf_counter() - t_start)
            if len(frame_times) > 15:
                frame_times.pop(0)
            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0.0

            # Render HUD Glass Card
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 15), (340, 95), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (15, 15), (340, 95), (60, 60, 60), 1)

            cv2.putText(frame, "RUST FACE RECOGNITION SERVER", (25, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 200), 1)
            cv2.putText(frame, f"Live Camera FPS: {fps:.1f} (Smooth)", (25, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100) if fps >= 25 else (220, 220, 220), 1)
            
            if server_error:
                latency_text = f"Server Error: {server_error}"
                lat_color = (0, 100, 255)
            elif server_lat is None:
                latency_text = "Server Latency: waiting..."
                lat_color = (220, 220, 220)
            else:
                latency_text = f"Server Latency: {server_lat:.1f} ms"
                lat_color = (100, 255, 100) if server_lat < 30 else (0, 220, 255)
            cv2.putText(frame, latency_text, (25, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, lat_color, 1)

            cv2.imshow("Live Face Recognition - CCTV / Webcam Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[Webcam] Stream stopped cleanly.")

if __name__ == "__main__":
    main()
