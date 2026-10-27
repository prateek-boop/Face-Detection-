import cv2
import os
import sys

files = [
    r"known_students\MANAN_GROVER_2823831\MANAN_GROVER_crop.jpg",
    r"known_students\MOHAMMD_ARMAN_2823400\MOHAMMD_ARMAN_crop.jpg",
    r"known_students\TEST_PIET-TEST-01\TEST_crop.jpg"
]

print("📊 AI-Powered Failure Analysis:")
print("-" * 50)

for f in files:
    if not os.path.exists(f):
        print(f"❌ {f}: File is missing entirely!")
        continue
        
    img = cv2.imread(f)
    if img is None:
        print(f"❌ {f}: Image is corrupted or invalid!")
        continue
        
    h, w = img.shape[:2]
    print(f"📁 {f}")
    print(f"   • Resolution: {w}x{h}")
    
    # Check for too small
    if w < 40 or h < 40:
        print("   ⚠️ REASON: Face is too small! (InsightFace ignores faces below 40x40)")
        continue
        
    # Check for extreme blur (Variance of Laplacian)
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    print(f"   • Clarity Score: {blur_score:.2f}")
    if blur_score < 15.0:
        print("   ⚠️ REASON: Image is extremely blurry!")
    else:
        print("   ⚠️ REASON: Potential False Detection. Your detector saw a 'face', but the recognizer disagreed.")

print("-" * 50)
