# Security & Code Audit Report

## [FIXED] [CRITICAL] Missing undeclared dependencies causing fatal import error
- File: app/generate_failed_list.py:6
- Issue: Imports `fitz` and `pillow_heif` which are not declared in requirements.txt, causing immediate runtime crash.
- Fix: Add `pymupdf` and `pillow-heif` to requirements.txt or wrap in conditional imports.

## [LOW] Bare exception handler hides runtime exceptions
- File: app/generate_failed_list.py:20
- Issue: Bare `except:` swallows all exceptions including keyboard interrupts and system exits.
- Fix: Catch `requests.RequestException` explicitly.

## [LOW] Hardcoded Windows backslash file paths
- File: app/debug_failures.py:6
- Issue: Hardcoded backslash path literals cause file resolution failures on POSIX platforms.
- Fix: Use `os.path.join` or `pathlib.Path` for cross-platform path resolution.

## [FIXED] [HIGH] AttendanceLogger daily cache not refreshed on midnight rollover
- File: app/face_recog.py:628
- Issue: `logged_today` is populated only at startup, causing attendance to be permanently blocked on subsequent days in long-running processes.
- Fix: Track current date on each `log` call and reset `logged_today` when the calendar day changes.

## [FIXED] [HIGH] Unhandled empty array in database embedding normalization
- File: app/face_recog.py:872
- Issue: Calling `np.linalg.norm(arr, axis=1)` on an empty embedding list raises `AxisError` and crashes database loading.
- Fix: Validate that `len(embeddings) > 0` before computing numpy array norm.

## [FIXED] [MEDIUM] Hardcoded webcam device index ignoring configuration
- File: app/face_recog.py:1043
- Issue: `cv2.VideoCapture(0)` hardcodes camera index 0 and ignores camera source settings in config.yaml.
- Fix: Pass `args.source` or `cfg['camera']['source']` to `cv2.VideoCapture`.

## [LOW] Obsolete commented code block at start of module
- File: app/face_recog.py:1
- Issue: Over 400 lines of obsolete commented code left at the top of the file.
- Fix: Delete lines 1 to 438.

## [FIXED] [HIGH] Identity truncation on multi-token names in flat folder layout
- File: app/build_encodings.py:120
- Issue: `identity_name.split("_")[0]` truncates multi-word student names to the first token only.
- Fix: Use `os.path.splitext(identity_name)[0]` or `rsplit("_", 1)[0]` to preserve full student names.

## [FIXED] [MEDIUM] Unprotected detector threshold state mutation
- File: app/build_encodings.py:193
- Issue: `det_thresh` is not restored if `self.face_app.get(img)` raises an unhandled exception.
- Fix: Enclose model inference in a `try...finally` block to guarantee restoring `det_thresh`.

## [FIXED] [MEDIUM] Non-atomic file persistence for embeddings database
- File: app/build_encodings.py:210
- Issue: Direct write to target pickle file without atomic rename risks corrupting database on process interruption.
- Fix: Write pickle data to a temporary file and atomically rename to destination path.

## [FIXED] [HIGH] Insecure model deserialization allows arbitrary code execution
- File: inference/detector.py:53
- Issue: `torch.load(..., weights_only=False)` unpickles arbitrary Python objects from untrusted checkpoint files.
- Fix: Set `weights_only=True` in `torch.load`.

## [LOW] Unused variable allocation in Soft-NMS
- File: inference/detector.py:170
- Issue: `remaining = np.array([indices])` allocates memory but is never referenced.
- Fix: Remove the unused assignment line.

## [LOW] Quadratic complexity from list removal in Soft-NMS loop
- File: inference/detector.py:163
- Issue: `indices.remove()` inside the while loop induces $O(N^2)$ array shifts during box suppression.
- Fix: Use a boolean suppression mask array instead of mutating a Python list.

## [FIXED] [HIGH] Export failure when landmark prediction head is disabled
- File: inference/export.py:42
- Issue: Declaring 3 output names raises RuntimeError when the model is configured with 2 heads (`landmark_enabled: false`).
- Fix: Dynamically set `output_names` based on `cfg['model']['landmark_enabled']`.

## [FIXED] [MEDIUM] Missing pre-benchmark CUDA synchronization
- File: inference/inference_test.py:46
- Issue: Timing starts before synchronizing CUDA stream, measuring asynchronous kernel launch instead of execution latency.
- Fix: Invoke `torch.cuda.synchronize()` immediately before recording start timestamp.

## [FIXED] [HIGH] Negative class index out of bounds in FocalLoss one-hot encoding
- File: model/loss.py:83
- Issue: Negative target labels such as `-1` (ignored anchors) produce invalid or out-of-bounds indices in `F.one_hot`.
- Fix: Mask out or clamp ignored anchor labels (`target < 0`) before one-hot transformation.

## [LOW] Module import inside hot training loss computation
- File: model/loss.py:60
- Issue: `import math` is executed inside `_box_ciou` on every forward batch pass.
- Fix: Relocate `import math` to top of file.

## [FIXED] [HIGH] Checkpoint loader vulnerable to arbitrary code execution
- File: training/checkpoint.py:132
- Issue: `torch.load(..., weights_only=False)` allows unpickling arbitrary bytecode from checkpoint files.
- Fix: Set `weights_only=True` in `torch.load`.

## [LOW] Bare exception in checkpoint directory scanning
- File: training/checkpoint.py:43
- Issue: Bare `except:` clause silently swallows unhandled parsing and filesystem errors.
- Fix: Replace with `except (IndexError, ValueError):`.

## [FIXED] [HIGH] Validation loop unpack crash when landmarks are disabled
- File: training/train.py:526
- Issue: `cls_preds, reg_preds, _ = val_model(images)` crashes with `ValueError` when model returns only 2 outputs.
- Fix: Unpack dynamically based on `self.cfg['model']['landmark_enabled']`.

## [FIXED] [MEDIUM] EarlyStopping score tracking bypasses minimum mode during warmup
- File: training/scheduler.py:109
- Issue: Epoch warmup guard updates `best_score` only when `self.mode == "max"`, ignoring `min` loss mode.
- Fix: Update `best_score` during warmup for both `max` and `min` optimization targets.

## [FIXED] [MEDIUM] Fragile annotation parser on zero-face WIDER FACE entries
- File: data/dataset.py:77
- Issue: Unconditionally skips line on `num_faces == 0`, causing parser desynchronization if dummy row is omitted.
- Fix: Peek line content to verify if next row is a dummy box before consuming.

## [FIXED] [MEDIUM] Inverted color channels in color jitter HSV conversion
- File: data/augment.py:172
- Issue: `COLOR_RGB2HSV` applied to BGR OpenCV images distorts hue and saturation transformations.
- Fix: Convert BGR image to RGB prior to HSV color perturbation.

## [LOW] Unused math import in augmentation module
- File: data/augment.py:9
- Issue: `import math` is imported but never referenced.
- Fix: Remove unused `import math` statement.

## [FIXED] [HIGH] Test assertion fails on default configuration without landmarks
- File: tests/sanity_check.py:95
- Issue: Forward pass shape test unpacks 3 tensors, crashing when `landmark_enabled: false`.
- Fix: Unpack 2 or 3 tensors conditionally depending on `cfg['model']['landmark_enabled']`.

## [FIXED] [CRITICAL] SQL injection vulnerability in student search query
- File: server/src/db/students.rs:47
- Issue: User search string is interpolated directly into raw SQL query via `format!` without parameter binding.
- Fix: Use sqlx parameterized query `$1` / `$2` with `.bind(format!("%{}%", s))`.

## [FIXED] [CRITICAL] SQL injection vulnerability in attendance query filters
- File: server/src/db/attendance.rs:47
- Issue: Date and roll_no parameters are directly concatenated into SQL query string without parameter binding.
- Fix: Bind parameters using parameterized placeholders instead of string formatting.

## [FIXED] [CRITICAL] Unbounded memory consumption on continuous camera streams
- File: server/src/camera/stream.rs:25
- Issue: `reqwest::get(&self.url).await?.bytes()` attempts to load unbounded/infinite MJPEG stream into memory, triggering OOM.
- Fix: Process frame stream via chunked multipart boundary reader rather than full body buffering.

## [FIXED] [HIGH] Face detection engine returns mock center box instead of inference output
- File: server/src/engine/mod.rs:125
- Issue: `detect_faces_internal` returns a synthetic center bounding box instead of running detector inference.
- Fix: Pass preprocessed tensor through ONNX model session and decode output anchors.

## [FIXED] [HIGH] Missing face embedding extraction causes all tracks to remain Unknown
- File: server/src/engine/mod.rs:77
- Issue: Track embedding is never extracted from image crops, leaving `track.embedding` as `None` permanently.
- Fix: Extract crop from image, run embedding model, and update `track.embedding`.

## [FIXED] [HIGH] Unbounded multipart body upload allows memory exhaustion
- File: server/src/api/handlers/inference.rs:32
- Issue: Multipart image endpoint buffers entire upload into RAM without enforcing maximum request size limits.
- Fix: Apply `DefaultBodyLimit` middleware and validate content length before reading bytes.

## [FIXED] [MEDIUM] Brittle positional column indexing in CSV import handler
- File: server/src/api/handlers/students.rs:90
- Issue: Hardcoded column indexes `record.get(1)` cause data corruption if CSV column order changes.
- Fix: Deserialize CSV records by column header names using serde.

## [FIXED] [MEDIUM] Redundant duplicate file processing in database seeder
- File: server/src/bin/seed_db.rs:43
- Issue: Path list contains both relative and parent paths, executing duplicate insertions if both paths resolve.
- Fix: Deduplicate canonical paths before iterating through CSV files.

## [FIXED] [MEDIUM] Plaintext database credentials committed in configuration file
- File: server/.env:6
- Issue: Database connection string containing cleartext password is stored in tracked `.env` file.
- Fix: Remove `.env` from repository tracking and provide only `.env.example`.

TOTAL ISSUES: 34 (CRITICAL: 4, HIGH: 12, MEDIUM: 10, LOW: 8)
