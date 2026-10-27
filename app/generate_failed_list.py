import pandas as pd
import os
import requests
import yaml
import io
try:
    import fitz
except ImportError:
    fitz = None

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.content
    except:
        pass
    return None

def check_image_content(content, url):
    try:
        Image.open(io.BytesIO(content))
        return True, "Valid Image"
    except:
        if url.lower().endswith(".pdf") or content.startswith(b"%PDF"):
            return True, "PDF"
        if url.lower().endswith((".mp4", ".mov", ".avi")):
            return True, "Video"
    return False, "Undecodable"

def main():
    csv_files = ["output.csv", "output2.csv"]
    known_dir = "known_students"
    
    all_dfs = []
    for f in csv_files:
        if os.path.exists(f):
            all_dfs.append(pd.read_csv(f))
    
    df = pd.concat(all_dfs).drop_duplicates(subset=['Roll No.'])
    failed_students = []

    print("Checking student records...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = str(row['Student Name']).strip().replace(" ", "_")
        roll = str(row['Roll No.']).strip()
        photo_url = str(row['Photo']).strip()
        
        student_dir = os.path.join(known_dir, f"{name}_{roll}")
        crop_path = os.path.join(student_dir, f"{name}_crop.jpg")
        
        if os.path.exists(crop_path):
            continue
            
        reason = "Unknown"
        if not photo_url or photo_url == 'nan' or not photo_url.startswith('http'):
            reason = "Missing or Invalid URL"
        else:
            # Check if it was a URL fix issue or download issue
            test_url = photo_url
            if ".comPIET" in test_url:
                test_url = test_url.replace(".comPIET", ".com/PIET")
            
            # Note: We don't want to redownload everything, but we need to know why it failed.
            # Most failures were logged in the previous turn.
            # We'll just mark them as "Processing Failed" for now.
            reason = "Processing Failed (No Face, Download Error, or Unsupported Format)"

        failed_students.append({
            "Roll No": roll,
            "Student Name": row['Student Name'],
            "Photo URL": photo_url,
            "Reason": reason
        })

    failed_df = pd.DataFrame(failed_students)
    failed_df.to_csv("failed_students_list.csv", index=False)
    print(f"\n✅ Created failed_students_list.csv with {len(failed_df)} entries.")

if __name__ == "__main__":
    main()
