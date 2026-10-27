"""
Face enrollment — captures and saves face crops with quality gating.

Replaces MediaPipe with our custom detector. Enforces blur, confidence,
and minimum-size thresholds before saving crops.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.detector import FaceDetector


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def check_quality(face_crop: np.ndarray, score: float,
                  blur_threshold: float = 100.0,
                  min_confidence: float = 0.8,
                  min_face_size: int = 64) -> tuple:
    """Check if a face crop passes quality requirements.

    Returns (passed: bool, reason: str).
    """
    h, w = face_crop.shape[:2]

    # Size check
    if h < min_face_size or w < min_face_size:
        return False, f"Too small ({w}×{h} < {min_face_size}px)"

    # Confidence check
    if score < min_confidence:
        return False, f"Low confidence ({score:.2f} < {min_confidence})"

    # Blur check (Laplacian variance — higher = sharper)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_threshold:
        return False, f"Blurry (score={blur_score:.1f} < {blur_threshold})"

    return True, "OK"


# ---------------------------------------------------------------------------
# Enrollment loop
# ---------------------------------------------------------------------------

def run_enrollment(detector: FaceDetector, cfg: dict):
    """Interactive webcam enrollment with quality gating."""
    cap_cfg = cfg.get("capture", {})
    blur_threshold = cap_cfg.get("blur_threshold", 100.0)
    min_confidence = cap_cfg.get("min_confidence", 0.8)
    min_face_size = cap_cfg.get("min_face_size", 64)
    num_crops = cap_cfg.get("num_crops", 5)
    save_root = cfg["paths"].get("known_students", "known_students")

    name = input("Enter the person's name: ").strip()
    if not name:
        print("❌ Name cannot be empty")
        return

    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    count = 0
    print(f"\n📸 Enrollment for '{name}'")
    print(f"   Press 's' to save a crop (need {num_crops})")
    print(f"   Quality gate: blur>{blur_threshold}, conf>{min_confidence}, size>{min_face_size}px")
    print(f"   Press 'q' to quit\n")

    while count < num_crops:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        result = detector.detect(frame)
        display = frame.copy()

        status_text = f"Captured: {count}/{num_crops}"

        if len(result["scores"]) > 0:
            # Use the highest-confidence detection
            best_idx = np.argmax(result["scores"])
            box = result["boxes"][best_idx].astype(int)
            score = result["scores"][best_idx]

            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_crop = frame[y1:y2, x1:x2]

            # Quality check
            if face_crop.size > 0:
                passed, reason = check_quality(
                    face_crop, score,
                    blur_threshold=blur_threshold,
                    min_confidence=min_confidence,
                    min_face_size=min_face_size,
                )

                if passed:
                    color = (0, 255, 0)   # green — good to save
                    status_text += " | ✅ READY (press 's')"
                else:
                    color = (0, 165, 255)  # orange — quality issue
                    status_text += f" | ⚠️ {reason}"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            status_text += " | No face detected"

        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(result["scores"]) > 0:
            best_idx = np.argmax(result["scores"])
            box = result["boxes"][best_idx].astype(int)
            score = result["scores"][best_idx]

            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                passed, reason = check_quality(
                    face_crop, score,
                    blur_threshold=blur_threshold,
                    min_confidence=min_confidence,
                    min_face_size=min_face_size,
                )
                if passed:
                    img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
                    cv2.imwrite(img_path, face_crop)
                    count += 1
                    print(f"  💾 Saved {img_path} ({count}/{num_crops})")
                else:
                    print(f"  ❌ Rejected: {reason}")

    cap.release()
    cv2.destroyAllWindows()

    if count >= num_crops:
        print(f"\n✅ Enrollment complete! {count} crops saved for '{name}'")
    else:
        print(f"\n⚠️  Enrollment incomplete: {count}/{num_crops} crops saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Face enrollment with quality gate")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    detector = FaceDetector(args.checkpoint, cfg)
    run_enrollment(detector, cfg)


if __name__ == "__main__":
    main()
