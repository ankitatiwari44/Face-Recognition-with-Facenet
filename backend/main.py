from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import threading
import os
import cv2
import numpy as np
from datetime import datetime
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
CORS(app)
is_running = False
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame to avoid NoneType errors

embedder = FaceNet()
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

path = 'Training_images'
known_embeddings = []
classNames = []

# Load single image per person (flat structure)
for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    person_name = os.path.splitext(filename)[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Cannot read image: {filename}")
        continue

    faces = face_app.get(img)
    if not faces:
        print(f"[!] No face detected in {filename}")
        continue

    face_crop = faces[0].crop_bgr
    if face_crop is None or face_crop.size == 0:
        print(f"[!] face_crop empty for {filename}, trying bbox")
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        face_crop = img[y1:y2, x1:x2]

    if face_crop is None or face_crop.size == 0:
        print(f"[!] Failed to crop {filename}")
        continue

    face_resized = cv2.resize(face_crop, (160, 160))
    embedding = embedder.embeddings([face_resized])[0]
    known_embeddings.append(embedding)
    classNames.append(person_name.lower())
    print(f"[+] Encoded: {filename}")

known_embeddings = normalize(np.array(known_embeddings))
print("Loaded embeddings:", len(known_embeddings))

THRESHOLD = 0.5
attendance_data = []  # will be used for frontend sync
attendance_marked = set()

# Load attendance from CSV on startup
if os.path.exists('Attendance.csv'):
    with open('Attendance.csv', 'r') as f:
        lines = f.read().splitlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                name, time_str = parts
                attendance_marked.add(name.lower())
                attendance_data.append({"name": name.lower(), "time": time_str})

def markAttendance(name, frame):
    normalized_name = name.strip().lower()
    if normalized_name in attendance_marked:
        print(f"[INFO] Attendance already marked for {normalized_name}")
        return

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    screenshot_name = f"{normalized_name}_{now.replace(':', '-').replace(' ', '_')}.jpg"
    screenshot_path = os.path.join("Screenshots", screenshot_name)

    os.makedirs("Screenshots", exist_ok=True)
    cv2.imwrite(screenshot_path, frame)
    print(f"📸 Screenshot saved: {screenshot_path}")

    filename = 'Attendance.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,Time\n")

    with open(filename, 'a', newline='') as f:
        f.write(f"{normalized_name},{now}\n")
        f.flush()

    print(f"✅ Attendance marked: {normalized_name} at {now}")
    attendance_marked.add(normalized_name)
    attendance_data.append({"name": normalized_name, "time": now})

@app.route('/start-recognition', methods=['GET'])
def start_recognition():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=run_recognition).start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

def run_recognition():
    global is_running, last_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        is_running = False
        return

    while is_running:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame")
            continue

        last_frame = frame.copy()
        faces = face_app.get(frame)
        for face in faces:
            face_crop = face.crop_bgr
            if face_crop is None or face_crop.size == 0:
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_crop = frame[y1:y2, x1:x2]

            if face_crop is None or face_crop.size == 0:
                continue

            try:
                resized = cv2.resize(face_crop, (160, 160))
                embedding = embedder.embeddings([resized])[0]
                embedding = normalize(embedding.reshape(1, -1))[0]
            except:
                continue

            if not known_embeddings.any():
                continue

            sims = cosine_similarity(embedding.reshape(1, -1), known_embeddings)[0]
            best_idx = np.argmax(sims)
            score = sims[best_idx]
            print(f"[MATCH] {classNames[best_idx]} with score {score}")

            if score > THRESHOLD:
                name = classNames[best_idx].lower()
                markAttendance(name, frame)

        time.sleep(0.05)

    cap.release()

@app.route('/stop-recognition', methods=['GET'])
def stop_recognition():
    global is_running
    is_running = False
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global last_frame
    while True:
        if last_frame is not None:
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/get-attendance', methods=['GET'])
def get_attendance():
    return jsonify(attendance_data)

@app.route('/upload-photo', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['photo']
    filename = secure_filename(file.filename)
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Failed to read image"}), 400

    faces = face_app.get(img)
    if not faces:
        return jsonify({"message": "No face detected"}), 200

    for face in faces:
        face_crop = face.crop_bgr
        if face_crop is None or face_crop.size == 0:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = img[y1:y2, x1:x2]
        if face_crop is None or face_crop.size == 0:
            continue

        try:
            resized = cv2.resize(face_crop, (160, 160))
            embedding = embedder.embeddings([resized])[0]
            embedding = normalize(embedding.reshape(1, -1))[0]
        except:
            continue

        if not known_embeddings.any():
            continue

        sims = cosine_similarity(embedding.reshape(1, -1), known_embeddings)[0]
        best_idx = np.argmax(sims)
        score = sims[best_idx]

        if score > THRESHOLD:
            name = classNames[best_idx].lower()
            markAttendance(name, img)
            return jsonify({"message": f"Attendance marked for {name}"}), 200

    return jsonify({"message": "No known face recognized"}), 200

if __name__ == '__main__':
    app.run(debug=True)
