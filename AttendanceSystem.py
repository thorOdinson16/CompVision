import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TF logs

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp
from deepface import DeepFace
from scipy.spatial.distance import cosine
from datetime import datetime
import numpy as np

# -------------------- Setup --------------------
path = "facerec"
if not os.path.exists(path):
    os.makedirs(path)

# Mediapipe face detector
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# Load known faces and compute embeddings
known_faces = {}
known_embeddings = {}
for file in os.listdir(path):
    img_path = os.path.join(path, file)
    if os.path.isfile(img_path):
        name = os.path.splitext(file)[0]
        known_faces[name] = img_path
        emb = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        known_embeddings[name] = emb

print("Loaded known faces:", list(known_faces.keys()))

# -------------------- Attendance Function --------------------
def markAttendance(name):
    with open("Attendance.csv", "a+") as f:
        f.seek(0)
        data = f.readlines()
        nameList = [line.split(",")[0] for line in data]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtString}")

# -------------------- Webcam Loop --------------------
cap = cv2.VideoCapture(0)
frame_count = 0
skip_frames = 10  # Compute embeddings every 5 frames

# To keep track of already recognized faces for smoothing
recent_faces = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Resize frame for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for det in results.detections:
            bboxC = det.location_data.relative_bounding_box
            h, w, _ = small_frame.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Scale bounding box back to original frame size
            x, y, w_box, h_box = int(x/0.3), int(y/0.3), int(w_box/0.3), int(h_box/0.3)

            # Crop detected face
            face = frame[y:y+h_box, x:x+w_box]
            recognized_name = "Unknown"

            if face.size > 0 and frame_count % skip_frames == 0:
                try:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    temp_emb = DeepFace.represent(face_rgb, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                    for name, known_emb in known_embeddings.items():
                        dist = cosine(temp_emb, known_emb)
                        if dist < 0.6:
                            recognized_name = name
                            recent_faces[name] = frame_count
                            markAttendance(name)
                            break
                except Exception as e:
                    print("Error in DeepFace:", e)

            # Use recent recognition to keep text stable
            for name in list(recent_faces):
                if frame_count - recent_faces[name] > 15:  # forget after 15 frames
                    recent_faces.pop(name)

            # If any recent face is in this bbox, use its name
            if recognized_name == "Unknown":
                for name in recent_faces:
                    recognized_name = name
                    break

            # Draw bounding box and name
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
            cv2.putText(frame, recognized_name.upper(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Attendance System - DeepFace", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()