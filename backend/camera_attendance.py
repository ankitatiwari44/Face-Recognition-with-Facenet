# camera_attendance.py
import cv2
import os
import numpy as np
from datetime import datetime
from keras_facenet import FaceNet
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

embedder = FaceNet()
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

path = 'Training_images'
known_embeddings = []
classNames = []

for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue
    faces = face_app.get(img)
    if not faces:
        continue
    face_crop = faces[0].crop_bgr
    if face_crop is None or face_crop.size == 0:
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        face_crop = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (160, 160))
    embedding = embedder.embeddings([face_resized])[0]
    known_embeddings.append(embedding)
    classNames.append(os.path.splitext(filename)[0])

attendance_marked = set()

def markAttendance(name, frame):
    filename = 'Attendance.csv'

    # Check if already marked
    marked_names = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            marked_names = set(line.split(',')[0] for line in lines)

    if name in marked_names:
        print(f"Attendance already marked for {name}")
        return  # Skip

    # Continue with marking and screenshot
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    screenshot_dir = "Screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_path = os.path.join(screenshot_dir, f"{name}_{now.replace(':', '-')}.jpg")
    cv2.imwrite(screenshot_path, frame)

    with open(filename, 'a') as f:
        f.write(f"{name},{now}\n")
    print(f"Attendance marked for {name}")
