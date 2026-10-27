"""
Robust Student Face Importer.

Extracts student face crops from Excel (.xlsx, .xls) and CSV files.
Supports any input media format:
- Images: JPG, JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC/HEIF
- Documents: PDF (full page rendering at 2x DPI + embedded image extraction)
- Videos: MP4, MOV, AVI, MKV, WEBM, FLV, 3GP (multi-frame sampling)
- Cloud Storage: Direct links, Okiedokie S3, Google Drive links (with virus-scan confirmation handler)
- Local Files: Absolute or relative file paths on disk

Saves cropped, aligned faces directly into `known_students/{name}_{roll}/`.
"""

import os
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import io
import re
import tempfile
import argparse
import zipfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from PIL import Image
import requests
import yaml
from tqdm import tqdm

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import pandas as pd

sys.path.insert(0, os.getcwd())
from inference.detector import FaceDetector


# ---------------------------------------------------------------------------
# URL & Download Utilities
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def normalize_url(url: str) -> str:
    """Fix common malformed URL patterns."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url or url.lower() == "nan":
        return ""

    # Fix missing slash after .com (e.g. okiedokie-erp.comPIET -> .com/PIET)
    if ".comPIET" in url:
        url = url.replace(".comPIET", ".com/PIET")
    elif ".inPIET" in url:
        url = url.replace(".inPIET", ".in/PIET")

    # Normalize Google Drive view links to direct download links
    if "drive.google.com" in url:
        # Match /file/d/<ID>/view or open?id=<ID> or id=<ID>
        file_id_match = re.search(r"/(?:file/d/|open\?id=|id=)([a-zA-Z0-9_-]+)", url)
        if file_id_match:
            file_id = file_id_match.group(1)
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Properly encode spaces and other URL characters in path if needed
    if "://" in url:
        try:
            parts = urllib.parse.urlsplit(url)
            quoted_path = urllib.parse.quote(urllib.parse.unquote(parts.path))
            url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, quoted_path, parts.query, parts.fragment))
        except Exception:
            pass

    return url


def download_file(url: str, session: requests.Session = None, timeout: int = 20) -> tuple[bytes | None, str]:
    """Download content from URL, handling Google Drive confirmation tokens and redirects."""
    url = normalize_url(url)
    if not url:
        return None, "Empty URL"

    # Local file support
    if os.path.exists(url) and os.path.isfile(url):
        try:
            with open(url, "rb") as f:
                return f.read(), "local_file"
        except Exception as e:
            return None, f"Local read error: {e}"

    if not url.startswith("http://") and not url.startswith("https://"):
        return None, "Invalid URL schema"

    sess = session or requests.Session()
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = sess.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"

        content = resp.content
        content_type = resp.headers.get("Content-Type", "").lower()

        # Handle Google Drive virus-scan / large file confirmation page
        if "drive.google.com" in url and "text/html" in content_type:
            # Check for confirm token in cookies or HTML
            confirm_token = None
            for k, v in resp.cookies.items():
                if k.startswith("download_warning"):
                    confirm_token = v
                    break

            if not confirm_token:
                match = re.search(r"confirm=([0-9A-Za-z_]+)", resp.text)
                if match:
                    confirm_token = match.group(1)

            if confirm_token:
                confirm_url = f"{url}&confirm={confirm_token}"
                resp2 = sess.get(confirm_url, headers=headers, timeout=timeout, allow_redirects=True)
                if resp2.status_code == 200 and "text/html" not in resp2.headers.get("Content-Type", "").lower():
                    return resp2.content, resp2.headers.get("Content-Type", "")

        return content, content_type

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Media Extraction: Any Image, PDF, Video, Zip
# ---------------------------------------------------------------------------

def extract_from_pdf(pdf_bytes: bytes) -> list[np.ndarray]:
    """Extract rendered pages and embedded images from PDF bytes."""
    images = []
    if fitz is None:
        return images

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_idx in range(len(doc)):
            page = doc[page_idx]

            # 1. Render page at 2x resolution (captures ID cards, forms, student badges)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            pil_page = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR))

            # 2. Extract embedded images
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                    raw_bytes = base_img["image"]
                    pil_emb = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    emb_arr = cv2.cvtColor(np.array(pil_emb), cv2.COLOR_RGB2BGR)
                    if emb_arr.shape[0] >= 64 and emb_arr.shape[1] >= 64:
                        images.append(emb_arr)
                except Exception:
                    continue

    except Exception as e:
        print(f"  [PDF Error] {e}")

    return images


def extract_from_video(video_bytes: bytes, max_frames: int = 15) -> list[np.ndarray]:
    """Sample multiple evenly distributed frames from video bytes."""
    frames = []
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_file = f.name

        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100

        # Sample frame indices (skip first/last 5% to avoid black transitions)
        start_f = max(0, int(total_frames * 0.05))
        end_f = max(start_f + 1, int(total_frames * 0.95))
        step = max(1, (end_f - start_f) // max_frames)
        sample_indices = set(range(start_f, end_f, step))

        current_frame = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in sample_indices:
                # Filter out completely dark or flat blank frames
                if frame is not None and frame.std() > 12.0:
                    frames.append(frame)
            current_frame += 1

        cap.release()

    except Exception as e:
        print(f"  [Video Error] {e}")
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    return frames


def bytes_to_images(content: bytes, url_or_filename: str) -> list[np.ndarray]:
    """Decode raw bytes into a list of candidate OpenCV BGR images."""
    if not content:
        return []

    hint = url_or_filename.lower().split("?")[0]
    images = []

    # 1. Try standard Image decoding (Pillow with HEIF + OpenCV)
    try:
        from PIL import ImageOps
        pil_img = Image.open(io.BytesIO(content))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        images.append(bgr)
        return images
    except Exception:
        pass

    # OpenCV imdecode fallback
    try:
        arr = np.frombuffer(content, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is not None:
            images.append(cv_img)
            return images
    except Exception:
        pass

    # 2. Check for PDF
    if hint.endswith(".pdf") or content.startswith(b"%PDF"):
        pdf_imgs = extract_from_pdf(content)
        if pdf_imgs:
            return pdf_imgs

    # 3. Check for Video
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".3gp", ".m4v", ".wmv")
    is_video = (
        any(hint.endswith(ext) for ext in video_exts)
        or (len(content) > 12 and content[4:8] == b"ftyp")
        or content.startswith(b"RIFF")
        or content.startswith(b"\x1a\x45\xdf\xa3")
    )
    if is_video:
        vid_imgs = extract_from_video(content)
        if vid_imgs:
            return vid_imgs

    # 4. Check for Zip archive
    if content.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    if any(name.lower().endswith(e) for e in (".jpg", ".jpeg", ".png", ".webp", ".heic")):
                        with zf.open(name) as zf_entry:
                            sub_imgs = bytes_to_images(zf_entry.read(), name)
                            images.extend(sub_imgs)
            if images:
                return images
        except Exception:
            pass

    # 5. Last resort: try video parser in case binary format wasn't recognized
    vid_imgs = extract_from_video(content)
    if vid_imgs:
        return vid_imgs

    return []


# ---------------------------------------------------------------------------
# Adaptive Face Detection Pipeline
# ---------------------------------------------------------------------------

def adaptive_face_detect(detector: FaceDetector, img: np.ndarray) -> tuple[dict | None, np.ndarray]:
    """Multi-pass adaptive face detection:
    Pass 1: Normal orientation with descending thresholds (0.50, 0.35, 0.20, 0.15)
    Pass 2: Rotations 90, 180, 270 deg (for sideways/rotated phone photos)
    Pass 3: 1.5x upscaling for small or distant photos
    Returns: (detection_result, working_image)
    """
    # 1. Normal orientation
    for conf in [0.50, 0.35, 0.28]:
        res = detector.detect(img, conf_threshold=conf)
        if len(res.get("boxes", [])) > 0:
            return res, img

    # 2. Rotations (90 deg CW, 180 deg, 270 deg CW)
    for rot in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        rot_img = cv2.rotate(img, rot)
        for conf in [0.40, 0.28]:
            res = detector.detect(rot_img, conf_threshold=conf)
            if len(res.get("boxes", [])) > 0:
                return res, rot_img

    # 3. Upscaling for small images (< 400px)
    h, w = img.shape[:2]
    if max(h, w) < 400:
        scaled = cv2.resize(img, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
        for conf in [0.30]:
            res = detector.detect(scaled, conf_threshold=conf)
            if len(res.get("boxes", [])) > 0:
                return res, scaled

    return None, img


# ---------------------------------------------------------------------------
# Student Face Processing
# ---------------------------------------------------------------------------

def process_single_student(row_data: dict, detector: FaceDetector, known_dir: str, max_crops: int = 5) -> tuple[bool, str]:
    """Process a single student record: download, decode, detect face, save crop."""
    name = str(row_data.get("name", "")).strip().replace(" ", "_")
    roll = str(row_data.get("roll", "")).strip()

    if not name or not roll or roll.lower() == "nan":
        return False, "Missing student name or roll number"

    student_dir = os.path.join(known_dir, f"{name}_{roll}")
    primary_crop = os.path.join(student_dir, f"{name}_crop.jpg")

    # If primary crop already exists, skip
    if os.path.exists(primary_crop):
        return True, "Already exists"

    urls = row_data.get("urls", [])
    if not urls:
        return False, "Missing photo URL"

    os.makedirs(student_dir, exist_ok=True)

    # Try each available URL (e.g. Okiedokie first, Google Drive fallback)
    last_err = "Download failed"
    for raw_url in urls:
        url = normalize_url(raw_url)
        if not url:
            continue

        content, err = download_file(url)
        if not content:
            last_err = f"Download failed: {err}"
            continue

        candidate_imgs = bytes_to_images(content, url)
        if not candidate_imgs:
            last_err = "Could not decode media (image/pdf/video)"
            continue

        saved_crops = 0
        for cand_idx, img in enumerate(candidate_imgs):
            if saved_crops >= max_crops:
                break

            res, working_img = adaptive_face_detect(detector, img)
            if res is None:
                continue

            boxes = res.get("boxes", np.array([]))
            scores = res.get("scores", np.array([]))
            if len(boxes) == 0:
                continue

            # Pick highest confidence face
            best_idx = int(np.argmax(scores))
            x1, y1, x2, y2 = boxes[best_idx].astype(int)
            w, h = x2 - x1, y2 - y1

            # 18% margin padding around face box
            pw, ph = int(w * 0.18), int(h * 0.18)
            x1 = max(0, x1 - pw)
            y1 = max(0, y1 - ph)
            x2 = min(working_img.shape[1], x2 + pw)
            y2 = min(working_img.shape[0], y2 + ph)

            crop = working_img[y1:y2, x1:x2]
            if crop.shape[0] < 12 or crop.shape[1] < 12:
                continue

            # Upscale small thumbnail crops to standard 112x112 using Lanczos interpolation
            # so facial recognition models have sufficient resolution for embedding generation
            if crop.shape[0] < 112 or crop.shape[1] < 112:
                crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LANCZOS4)

            crop_name = f"{name}_crop.jpg" if saved_crops == 0 else f"{name}_crop_{saved_crops + 1}.jpg"
            save_path = os.path.join(student_dir, crop_name)
            cv2.imwrite(save_path, crop)
            saved_crops += 1

        if saved_crops > 0:
            return True, f"Success ({saved_crops} crop{'s' if saved_crops > 1 else ''})"

        last_err = "No face detected in candidate images"

    return False, last_err


# ---------------------------------------------------------------------------
# Dataframe Ingestion: Excel & CSV
# ---------------------------------------------------------------------------

def load_records_from_file(file_path: str) -> list[dict]:
    """Load and normalize student records from an Excel or CSV file."""
    if not os.path.exists(file_path):
        print(f"  [Warning] File not found: {file_path}")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    df = None

    if ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(file_path)
        # Select best sheet: 'All Student Photo Links' if present, else largest sheet
        sheet_to_use = xl.sheet_names[0]
        for s in xl.sheet_names:
            if "student" in s.lower() and "link" in s.lower():
                sheet_to_use = s
                break
        df = xl.parse(sheet_to_use)
        print(f"[Loaded] Loaded {len(df)} rows from Excel sheet '{sheet_to_use}' in {os.path.basename(file_path)}")
    else:
        df = pd.read_csv(file_path)
        print(f"[Loaded] Loaded {len(df)} rows from CSV {os.path.basename(file_path)}")

    # Column name mapping
    cols = {str(c).strip(): c for c in df.columns}
    name_col = next((cols[c] for c in cols if re.search(r"student.*name|^name$", c, re.IGNORECASE)), None)
    roll_col = next((cols[c] for c in cols if re.search(r"roll.*no|^roll$", c, re.IGNORECASE)), None)

    # Collect all photo URL columns (prioritize download links over view links)
    url_cols = []
    for c in cols:
        if re.search(r"photo.*download.*link", c, re.IGNORECASE):
            url_cols.insert(0, cols[c])
        elif re.search(r"google.*drive.*view.*link", c, re.IGNORECASE):
            url_cols.append(cols[c])
        elif re.search(r"^photo$|photo.*url|original.*okiedokie", c, re.IGNORECASE):
            url_cols.append(cols[c])

    if not name_col or not roll_col:
        print(f"  [Failed] Required columns (Name, Roll) not found in {file_path}. Columns: {list(df.columns)}")
        return []

    records = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        roll = str(row[roll_col]).strip()
        if not name or name.lower() == "nan" or not roll or roll.lower() == "nan":
            continue

        candidate_urls = []
        for uc in url_cols:
            val = str(row[uc]).strip()
            if val and val.lower() != "nan":
                parts = [p.strip() for p in val.split(" | ")] if " | " in val else [val]
                for p in parts:
                    if p and p.lower() != "nan" and p not in candidate_urls:
                        candidate_urls.append(p)

        records.append({
            "name": name,
            "roll": roll,
            "urls": candidate_urls,
        })

    return records


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Universal Student Photo Importer")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "PIET - All Students - Okiedokie and Drive Links.xlsx",
            "output.csv",
            "output2.csv",
        ],
        help="Path to Excel (.xlsx, .xls) or CSV files",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt", help="Path to face detector checkpoint")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent download threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of students to process (for testing)")
    parser.add_argument("--force", action="store_true", help="Re-process even if crop already exists")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    known_dir = cfg.get("paths", {}).get("known_students", "known_students")
    os.makedirs(known_dir, exist_ok=True)

    print("=" * 60)
    print("UNIVERSAL STUDENT FACE IMPORTER")
    print("Supports: Images (JPG/PNG/HEIC/WEBP), PDFs, Videos, Drive Links")
    print("=" * 60)

    # 1. Load records from all input files
    all_records = []
    seen_rolls = set()
    for fpath in args.input:
        if os.path.exists(fpath):
            recs = load_records_from_file(fpath)
            for r in recs:
                if r["roll"] not in seen_rolls:
                    seen_rolls.add(r["roll"])
                    all_records.append(r)

    if not all_records:
        print("[Failed] No valid student records found across input files.")
        return

    if args.limit:
        all_records = all_records[:args.limit]

    print(f"\n[Records] Total unique student records to check: {len(all_records)}")

    # 2. Check already processed students
    to_process = []
    already_done = 0
    for r in all_records:
        name_clean = r["name"].strip().replace(" ", "_")
        crop_path = os.path.join(known_dir, f"{name_clean}_{r['roll']}", f"{name_clean}_crop.jpg")
        if os.path.exists(crop_path) and not args.force:
            already_done += 1
        else:
            to_process.append(r)

    print(f"   [Skipped]  Already existing face crops: {already_done}")
    print(f"   [Download] New / Missing students to download: {len(to_process)}")

    if not to_process:
        print("\n[Success] All students already have face crops! Run 'app/build_encodings.py' to update embeddings.")
        return

    # 3. Initialize detector
    print(f"\n[Model] Loading Face Detector from {args.checkpoint}...")
    detector = FaceDetector(args.checkpoint, cfg)

    # 4. Multi-threaded download & detection
    success_count = 0
    failed_records = []
    failed_reasons = {}

    print(f"[Running] Processing with {args.workers} download worker threads...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_rec = {
            executor.submit(process_single_student, rec, detector, known_dir): rec
            for rec in to_process
        }

        with tqdm(total=len(to_process), desc="Importing students") as pbar:
            for future in as_completed(future_to_rec):
                rec = future_to_rec[future]
                try:
                    success, reason = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                        failed_records.append({
                            "Roll No": rec["roll"],
                            "Student Name": rec["name"],
                            "Photo URLs": " | ".join(rec["urls"]),
                            "Reason": reason,
                        })
                except Exception as exc:
                    err_msg = str(exc)
                    failed_reasons[err_msg] = failed_reasons.get(err_msg, 0) + 1
                    failed_records.append({
                        "Roll No": rec["roll"],
                        "Student Name": rec["name"],
                        "Photo URLs": " | ".join(rec["urls"]),
                        "Reason": err_msg,
                    })
                finally:
                    pbar.update(1)

    # 5. Output Summary & Save Failures
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"  [Success] Successfully extracted: {success_count}")
    print(f"  [Skipped]  Previously existing:   {already_done}")
    print(f"  [Failed] Failed to extract:       {len(failed_records)}")

    if failed_reasons:
        print("\nBreakdown of failures:")
        for r, cnt in sorted(failed_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {r}: {cnt}")

    if failed_records:
        fail_df = pd.DataFrame(failed_records)
        fail_df.to_csv("failed_students_list.csv", index=False)
        print("\n[File] Detailed list of failures saved to: failed_students_list.csv")

    print("\n[Next] NEXT STEP:")
    print("Rebuild face encodings for live recognition:")
    print("  .\\venv\\Scripts\\python.exe app/build_encodings.py --config config.yaml")


if __name__ == "__main__":
    main()
