"""
Inference testing — FPS benchmarking, GT vs prediction comparison,
landmark visualisation, and edge-case testing.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.detector import FaceDetector
from data.dataset import parse_wider_face_annotations


# ---------------------------------------------------------------------------
# FPS benchmark
# ---------------------------------------------------------------------------

def benchmark_fps(detector: FaceDetector, img_size: int, device: str,
                  warmup: int = 20, iters: int = 100):
    """Measure inference FPS.

    Parameters
    ----------
    detector : FaceDetector
    img_size : int
    device : str ('cpu' or 'cuda')
    warmup : int  warmup iterations
    iters : int  timed iterations
    """
    # Create a dummy image
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        detector.detect(dummy)

    # Timed
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        detector.detect(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    fps = 1000.0 / np.mean(times)
    p95 = np.percentile(times, 95)

    print(f"\n[FPS Benchmark] ({device})")
    print(f"   Mean latency: {np.mean(times):.2f} ms")
    print(f"   P95 latency:  {p95:.2f} ms")
    print(f"   FPS:          {fps:.1f}")

    return fps


# ---------------------------------------------------------------------------
# GT vs prediction comparison grid
# ---------------------------------------------------------------------------

def comparison_grid(detector: FaceDetector, images_root: str,
                    entries: list, output_path: str, num_images: int = 8):
    """Create side-by-side GT vs predicted boxes grid.

    Parameters
    ----------
    detector : FaceDetector
    images_root : str
    entries : list  — from parse_wider_face_annotations
    output_path : str — save path for the grid image
    num_images : int
    """
    import random
    selected = random.sample(entries, min(num_images, len(entries)))

    panels = []
    for entry in selected:
        img_path = os.path.join(images_root, entry["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        # GT panel
        gt_panel = img.copy()
        for box in entry["boxes"]:
            x, y, w, h = map(int, box)
            cv2.rectangle(gt_panel, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(gt_panel, "Ground Truth", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Prediction panel
        pred_panel = detector.detect_and_draw(img)
        cv2.putText(pred_panel, "Prediction", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Resize both to same size
        target_h = 300
        scale = target_h / img.shape[0]
        target_w = int(img.shape[1] * scale)
        gt_panel = cv2.resize(gt_panel, (target_w, target_h))
        pred_panel = cv2.resize(pred_panel, (target_w, target_h))

        # Horizontal concat
        pair = np.hstack([gt_panel, pred_panel])
        panels.append(pair)

    if not panels:
        print("⚠️  No valid images found for comparison")
        return

    # Stack vertically
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 128, dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    grid = np.vstack(padded)
    cv2.imwrite(output_path, grid)
    print(f"📸 Comparison grid saved to {output_path}")


# ---------------------------------------------------------------------------
# Edge case testing
# ---------------------------------------------------------------------------

def test_edge_cases(detector: FaceDetector, output_dir: str):
    """Test on synthetic edge cases: dark, partial, crowd."""
    os.makedirs(output_dir, exist_ok=True)

    # Dark image
    dark = np.full((640, 640, 3), 20, dtype=np.uint8)  # very dark
    result = detector.detect(dark)
    print(f"🌑 Dark image: {len(result['scores'])} detections")

    # Half image (partial face simulation)
    half = np.random.randint(100, 200, (320, 640, 3), dtype=np.uint8)
    half = np.vstack([half, np.full((320, 640, 3), 114, dtype=np.uint8)])
    result = detector.detect(half)
    print(f"🔲 Partial image: {len(result['scores'])} detections")

    # Random noise (should produce few/no detections)
    noise = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result = detector.detect(noise)
    print(f"🎲 Random noise: {len(result['scores'])} detections")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference testing")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None, help="Single image to test")
    parser.add_argument("--benchmark", action="store_true", help="Run FPS benchmark")
    parser.add_argument("--compare", action="store_true",
                        help="Generate GT vs pred comparison grid")
    parser.add_argument("--edge-cases", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    detector = FaceDetector(args.checkpoint, cfg)

    # Single image test
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"[Error] Could not read {args.image}")
            return
        result = detector.detect(img)
        print(f"Detected {len(result['scores'])} faces:")
        for i, (box, score) in enumerate(zip(result['boxes'], result['scores'])):
            print(f"  [{i}] score={score:.3f} box={box.astype(int).tolist()}")

        out_path = os.path.join(args.output_dir, "detection_result.jpg")
        drawn = detector.detect_and_draw(img)
        cv2.imwrite(out_path, drawn)
        print(f"[Saved] Saved to {out_path}")

    # Benchmark
    if args.benchmark:
        img_size = cfg["model"]["image_size"]
        benchmark_fps(detector, img_size, "cpu")
        if torch.cuda.is_available():
            benchmark_fps(detector, img_size, "cuda")

    # Comparison grid
    if args.compare:
        paths = cfg["paths"]
        entries = parse_wider_face_annotations(paths["val_ann"])
        grid_path = os.path.join(args.output_dir, "comparison_grid.jpg")
        comparison_grid(detector, paths["val_images"], entries, grid_path)

    # Edge cases
    if args.edge_cases:
        test_edge_cases(detector, args.output_dir)


if __name__ == "__main__":
    main()
