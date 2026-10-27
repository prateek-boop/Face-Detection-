# Face Detection & Recognition System

A high-performance face detection and recognition pipeline combining:
- **Custom object detector** trained on WIDER FACE dataset
- **Real-time SORT tracking** with IoU-based matching
- **InsightFace embeddings** for identity recognition
- **Live webcam pipeline** with 25+ FPS throughput

---

## 🎯 Project Overview

This system performs end-to-end face detection and recognition on video streams:

1. **Detection**: Custom neural network detects faces in images/video
2. **Tracking**: SORT tracker assigns persistent IDs across frames
3. **Recognition**: Pre-computed face embeddings identify known individuals
4. **Output**: Annotated video with bounding boxes and names

**Key Features:**
- Track-before-recognize optimization (embeddings extracted every 5 frames per track)
- Soft-NMS for crowded scenes
- Automatic encoding updates when new students are added
- Confidence calibration and edge-case handling
- CUDA FP16 acceleration for speed

---

## 📁 Project Structure

```
Face/
├── README.md                    # This file
├── GETTING_STARTED.md           # Detailed setup guide
├── config.yaml                  # Single source of truth for all hyperparameters
├── requirements.txt             # Python dependencies (PyTorch, InsightFace, etc.)
│
├── app/                         # Application layer (inference + enrollment)
│   ├── face_recog.py           # Live webcam recognition pipeline
│   ├── build_encodings.py      # Build embedding database from face crops
│   ├── capture_photo.py        # Enroll new students (capture + save crops)
│   ├── import_students.py      # Batch import students from external data
│   ├── generate_failed_list.py # Debug script to find recognition failures
│   ├── debug_failures.py       # Analyze failed face detections
│   └── __init__.py
│
├── inference/                   # Inference utilities
│   ├── detector.py             # FaceDetector class (load model + predict)
│   ├── inference_test.py       # Benchmark FPS, test single images, edge cases
│   ├── export.py               # Export model to ONNX / TorchScript
│   ├── profiler.py             # Profile bottlenecks (Chrome trace)
│   └── __init__.py
│
├── data/                        # Data pipeline
│   ├── dataset.py              # PyTorch dataset for training
│   ├── augment.py              # Image augmentation (mosaic, color jitter, cutout)
│   ├── anchors.py              # Anchor generation (data-driven via K-means)
│   └── __init__.py
│
├── model/                       # Neural network architecture
│   ├── backbone.py             # MobileNetV2-style backbone
│   ├── neck.py                 # Feature Pyramid Network (FPN)
│   ├── head.py                 # Detection head (classification + regression)
│   └── __init__.py
│
├── training/                    # Training pipeline
│   ├── train.py                # Main training loop (FP16, early stopping, checkpointing)
│   └── __init__.py
│
├── checkpoints/                 # Saved model weights
│   ├── latest.pt               # Latest checkpoint (auto-loaded on resume)
│   ├── epoch_0020_mAP_0.8084.pt
│   └── ...
│
├── known_students/             # Enrolled student face crops (organized by name)
│   ├── AACHAL_KARN_241703/     # Sub-directory per identity
│   ├── AADARSH_DESHWAL_28250239/
│   └── ...
│
├── outputs/                     # Inference results
│   ├── detections/             # Annotated images/video
│   ├── profiling/              # Chrome trace files
│   └── ...
│
├── attendance.csv              # Sample attendance records
├── config.yaml                 # Training & inference hyperparameters
├── encodings.pkl               # Face embeddings database (binary)
├── failed_encodings.txt        # List of crops that failed encoding
├── failed_students_list.csv    # Recognition failures for debugging
└── cctv_adaption.md           # Notes on CCTV/surveillance adaptation

```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **CUDA GPU** with ≥6 GB VRAM (recommended; CPU mode slower)
- **Virtual Environment** (required for GPU detection)

### Setup (2 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run live recognition (requires pre-trained checkpoint)
venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/latest.pt
```

**That's it!** Press `q` to quit.

### Common Commands

| Task | Command |
|------|---------|
| **Live webcam** | `app/face_recog.py --checkpoint checkpoints/latest.pt` |
| **Single image** | `inference/inference_test.py --image photo.jpg` |
| **Benchmark FPS** | `inference/inference_test.py --benchmark` |
| **Enroll student** | `app/capture_photo.py --checkpoint checkpoints/latest.pt` |
| **Build embeddings** | `app/build_encodings.py` |
| **Train model** | `training/train.py --config config.yaml` |

---

## 📊 System Architecture

### 1. Detection Model
- **Input**: 640×640 RGB images
- **Backbone**: MobileNetV2 (configurable width multiplier)
- **Neck**: Feature Pyramid Network (256 channels)
- **Head**: Multi-scale anchor-based detector
  - Classification branch: per-anchor face probability
  - Regression branch: bounding box adjustments
  - Landmarks branch: (optional) 5-point face landmarks

**Training:**
- Loss: Focal loss (classification) + Smooth L1 (regression)
- Data: WIDER FACE (393K images, diverse poses/lighting)
- Augmentation: Mosaic, color jitter, random crop, cutout

### 2. Tracking (SORT + IoU)
- Matches detections frame-to-frame using **Intersection over Union (IoU)**
- No Kalman filter — simplicity over precision
- Maintains track ID across frames for stable recognition
- Auto-expires tracks after 10 frames without detection

### 3. Recognition (InsightFace)
- Pre-computed 512-D embeddings per identity
- **Cosine similarity** threshold: 0.4 (configurable)
- Embeddings extracted every 5 frames per track (not every frame)
- Exponential moving average smoothing for stable identity

### 4. End-to-End Workflow
```
Webcam Frame
    ↓
[Detection] → Face Bounding Boxes + Scores
    ↓
[Tracking] → Persistent Track IDs (IoU matching)
    ↓
[Embedding] → Every 5 frames: extract 512-D vector
    ↓
[Recognition] → Cosine similarity to known identities
    ↓
[Render] → Draw boxes + names + IDs
    ↓
Output Video
```

---

## 🔧 Core Components & Files

### Detection (`inference/detector.py`)
- Loads trained checkpoint and prepares model for inference
- Handles **Soft-NMS** (Gaussian-weighted score decay for overlaps)
- Confidence calibration via Platt scaling
- Returns: boxes, scores, landmarks (if enabled)

**Usage:**
```python
from inference.detector import FaceDetector

detector = FaceDetector("checkpoints/latest.pt", cfg)
boxes, scores, landmarks = detector.predict(frame)  # (N, 4), (N,), (N, 10)
```

### Tracking (`app/face_recog.py` → `Track` class)
- **Track object** stores: box, embedding, identity, track ID
- **SORT update** matches detections to tracks via IoU
- Exponential moving average of embeddings for stability
- Tracks aged out after 10 frames without new detections

### Encoding Builder (`app/build_encodings.py`)
- Scans `known_students/{name}/` directories
- Extracts face crops and computes embeddings via InsightFace
- Incremental mode: only re-processes modified directories
- Output: `encodings.pkl` = `{identity: [embed1, embed2, ...]}`

### Face Recognition (`app/face_recog.py`)
- Main loop: detect → track → embed → recognize
- Performs 1:N similarity search against known embeddings
- Returns top match if confidence > 0.4
- Falls back to "Unknown" otherwise

### Enrollment (`app/capture_photo.py`)
- Webcam loop with quality gates:
  - **Blur check**: Laplacian variance > 100
  - **Confidence**: Detection score > 0.8
  - **Size**: Face bounding box > 64×64 pixels
- Saves 5 crops per person to `known_students/{name}/`

---

## ⚙️ Configuration (`config.yaml`)

All hyperparameters live in `config.yaml`:

| Section | Key | Purpose |
|---------|-----|---------|
| `model` | `image_size`, `anchor_scales`, `anchor_ratios` | Model architecture |
| `train` | `batch_size`, `lr`, `max_epochs` | Training params |
| `loss` | `focal_alpha`, `reg_weight` | Loss function tuning |
| `augment` | `mosaic_prob`, `flip_prob` | Data augmentation |
| `inference` | `conf_threshold`, `nms_iou` | Detection thresholds |
| `capture` | `blur_threshold`, `min_confidence` | Enrollment quality gates |
| `paths` | `dataset_root`, `known_students` | Data directories |

**Edit before training/running:**
- Lower `train.batch_size` if OOM errors
- Increase `max_epochs` for longer training
- Raise `inference.conf_threshold` to filter low-confidence detections
- Tune `capture.blur_threshold` if enrollment photos are noisy

---

## 📚 Workflow Examples

### Scenario 1: Run Live Recognition (Already Trained)
```bash
# 1. Make sure known_students/ has enrolled faces
ls known_students/  # Should see directories like JOHN_DOE/

# 2. Build embeddings
venv\Scripts\python.exe app/build_encodings.py

# 3. Run recognition
venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/latest.pt

# 4. Press 'q' to quit
```

### Scenario 2: Enroll a New Student
```bash
# 1. Capture 5 face crops
venv\Scripts\python.exe app/capture_photo.py --checkpoint checkpoints/latest.pt
# Enter name: JOHN_DOE
# Press 's' to save good crops (5 total)

# 2. Rebuild encodings (auto-updates to include new faces)
venv\Scripts\python.exe app/build_encodings.py

# 3. Run recognition again
venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/latest.pt
```

### Scenario 3: Train a New Model
```bash
# 1. Download & organize WIDER FACE dataset (see GETTING_STARTED.md)
# 2. Compute data-driven anchors
venv\Scripts\python.exe data/anchors.py --config config.yaml --update-config

# 3. Train
venv\Scripts\python.exe training/train.py --config config.yaml

# 4. Test inference
venv\Scripts\python.exe inference/inference_test.py --checkpoint checkpoints/latest.pt --image test.jpg

# 5. Once satisfied, link to known_students/ and run recognition
```

---

## 🎓 Key Technical Details

### Soft-NMS
Non-Maximum Suppression with Gaussian score decay instead of hard deletion. Critical in crowded scenes where overlapping faces shouldn't be suppressed.

### Track-Before-Recognize
Embedding extraction runs every 5 frames per track (not every frame). This 5× speedup is safe because tracks are stable within short time windows. Provides 25+ FPS instead of ~5 FPS.

### Exponential Moving Average (EMA) Embeddings
When the same track gets a new embedding, blend it with the previous one:
```
new_embedding = 0.5 * new + 0.5 * old
```
This smooths jittery identity predictions across frames.

### Anchor Generation
K-means clustering on WIDER FACE bounding box aspect ratios → data-driven anchors instead of generic ones. Improves detection of faces with diverse poses.

### Focal Loss
Addresses class imbalance (most anchors are negatives). Reduces loss contribution from easy examples, focuses on hard ones.

---

## 🛠️ Debugging & Troubleshooting

| Issue | Solution |
|-------|----------|
| Low FPS | Reduce `capture.blur_threshold` or skip embedding for some tracks |
| Poor recognition | Add more enrollment crops (5+ per person); check lighting conditions |
| Missing detections | Lower `inference.conf_threshold`; retrain on harder examples |
| CUDA not detected | Always use `venv\Scripts\python.exe` (not global `python`) |
| OOM during training | Reduce `train.batch_size` in `config.yaml` |

See [failed_students_list.csv](failed_students_list.csv) for recognition failures and [debug_failures.py](app/debug_failures.py) to analyze them.

---

## 📖 Additional Resources

- **[GETTING_STARTED.md](GETTING_STARTED.md)** — Detailed step-by-step setup (training, inference, enrollment)
- **[cctv_adaption.md](cctv_adaption.md)** — Notes on adapting for surveillance systems
- **Model files**: See `model/` for backbone, FPN, and detection head implementations
- **Data pipeline**: See `data/` for augmentation and dataset classes

---

## 🎬 Output Examples

### Live Recognition Output
- Bounding box around each face (green = confident, yellow = low confidence)
- Track ID (persistent across frames)
- Identity name + confidence score
- FPS counter in top-left

### Saved Results
- Images: `outputs/detections/` (annotated JPEG)
- Profiles: `outputs/profiling/trace.json` (Chrome trace)
- Metrics: `outputs/evaluation.txt` (mAP, per-class accuracy)

---

## 📋 Training Summary

| Metric | Value |
|--------|-------|
| Dataset | WIDER FACE (393K images) |
| Model | MobileNetV2 backbone + FPN |
| Input | 640×640 RGB |
| Training time | ~40 hours (RTX 3090) |
| Inference speed | 25+ FPS on NVIDIA GPU |
| Accuracy (mAP@0.5) | ~81% (varies by training run) |

---

## 🔐 License & Attribution

- **WIDER FACE dataset**: Yang et al. (2016) — [Paper](https://arxiv.org/abs/1511.06523)
- **InsightFace**: Jiankang Deng et al. — [GitHub](https://github.com/deepinsight/insightface)
- **SORT Tracker**: Bewley et al. (2016) — [Paper](https://arxiv.org/abs/1602.00763)

---

## 📝 Quick Reference

**Installation:**
```bash
pip install -r requirements.txt
```

**Run Recognition:**
```bash
venv\Scripts\python.exe app/face_recog.py --checkpoint checkpoints/latest.pt
```

**Train Model:**
```bash
venv\Scripts\python.exe training/train.py --config config.yaml
```

**For detailed guides, see [GETTING_STARTED.md](GETTING_STARTED.md)**
