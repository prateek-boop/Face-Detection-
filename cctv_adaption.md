
  Here is the plan:

  CCTV Camera Integration Plan

  Objective
  Integrate support for CCTV cameras (RTSP/HTTP streams) into the live face recognition pipeline, ensuring robust
  operation with automatic reconnection if the stream drops.

  Key Files & Context
   - config.yaml: Central configuration file. Needs a new section to define the camera source.
   - app/face_recog.py: The main live recognition script. Needs to be updated to accept the new source and handle
     potential disconnections gracefully.

  Implementation Steps

  1. Update Configuration (config.yaml)
  Add a new camera section to hold the video source. This can be an integer (for webcams) or a string (URL) for CCTV
  cameras.

   1 # Add this section to config.yaml
   2 camera:
   3   source: 0  # 0 for webcam, or "rtsp://user:pass@ip:port/stream" for CCTV

  2. Update CLI Arguments (app/face_recog.py)
  Add a --source argument to the main() function in app/face_recog.py to allow users to override the configured source
  directly from the command line.

  3. Implement Robust Reconnection Logic (app/face_recog.py)
  Modify the LiveRecognizer.run method to handle stream disconnections:
   - Read the source from args.source or cfg['camera']['source'].
   - If the source is an integer string (e.g., "0"), convert it to an int.
   - Wrap the video capture logic in a loop that attempts to reconnect if the stream is lost or cap.isOpened() returns
     False.
   - Add a small delay between reconnection attempts to avoid spamming connection requests.
   - Use cv2.VideoCapture with the provided source. Remove the hardcoded cv2.VideoCapture(0).

  Verification & Testing
   1. Run the system with the default webcam (source 0) to ensure backward compatibility.
   2. If a CCTV stream is available, run the system with --source "rtsp://..." and verify it connects and processes
      frames.
   3. Simulate a disconnection (e.g., by disabling the network temporarily) and observe if the system attempts to
      reconnect and successfully resumes when the connection is restored.

  Does this plan look good to you? If so, I will proceed to formally request your approval to begin implementation.