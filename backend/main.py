# app.py
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from keras_facenet import FaceNet
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Face Attendance (Streamlit)", layout="wide")
st.title("Face Recognition Attendance (Streamlit)")

# -----------------------------
# Utility paths
# -----------------------------
TRAIN_DIR = "Training_images"     # one image per person, filename = person_name.jpg/png
SHOT_DIR  = "Screenshots"
CSV_PATH  = "Attendance.csv"
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(SHOT_DIR, exist_ok=True)


# -----------------------------
# Caching heavy resources
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    embedder = FaceNet()  # Keras-FaceNet
    face_app = FaceAnalysis(name="buffalo_l")
    # ctx_id = -1 for CPU, >=0 for GPU; set 0 if you have a CUDA GPU
    face_app.prepare(ctx_id=0 if st.sidebar.checkbox("Use GPU (ctx_id=0)?", value=False) else -1)
    return embedder, face_app


@st.cache_resource(show_spinner=True)
def build_known_embeddings(train_dir: str):
    """
    Reads one face image per person from `train_dir` and returns:
    - class_names: list[str]
    - known_embeddings: np.ndarray [N, D] normalized
    """
    embedder, face_app = load_models()

    class_names = []
    embs = []

    for filename in sorted(os.listdir(train_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        person_name = os.path.splitext(filename)[0].strip().lower()
        img_path = os.path.join(train_dir, filename)

        img = cv2.imread(img_path)
        if img is None:
            st.warning(f"Cannot read image: {filename}")
            continue

        faces = face_app.get(img)
        if not faces:
            st.warning(f"No face detected in training image: {filename}")
            continue

        face = faces[0]
        face_crop = face.crop_bgr
        if face_crop is None or face_crop.size == 0:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = img[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            st.warning(f"Failed to crop face for: {filename}")
            continue

        face_resized = cv2.resize(face_crop, (160, 160))
        emb = embedder.embeddings([face_resized])[0]
        embs.append(emb)
        class_names.append(person_name)

    if len(embs) == 0:
        return [], np.empty((0, 512), dtype=np.float32)

    known_embeddings = normalize(np.asarray(embs))
    return class_names, known_embeddings


# -----------------------------
# Attendance helpers
# -----------------------------
def load_attendance(csv_path: str):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["Name"] = df["Name"].astype(str).str.lower()
        return df
    return pd.DataFrame(columns=["Name", "Time"])


def save_attendance_row(csv_path: str, name: str, when: str):
    header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        if header:
            f.write("Name,Time\n")
        f.write(f"{name},{when}\n")


def mark_attendance(name: str, frame_bgr: np.ndarray, shot_dir: str, csv_path: str, marked_set: set):
    nm = name.strip().lower()
    if nm in marked_set:
        return False

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shot_name = f"{nm}_{now_str.replace(':', '-').replace(' ', '_')}.jpg"
    shot_path = os.path.join(shot_dir, shot_name)
    cv2.imwrite(shot_path, frame_bgr)
    save_attendance_row(csv_path, nm, now_str)
    marked_set.add(nm)
    return True


# -----------------------------
# Matching logic
# -----------------------------
def detect_and_match(bgr_img: np.ndarray, threshold: float,
                     class_names, known_embeddings, face_app, embedder):
    """
    Returns (best_name, best_score, boxed_image) OR (None, None, boxed_image)
    """
    draw = bgr_img.copy()
    faces = face_app.get(bgr_img)
    if not faces or known_embeddings.size == 0:
        return None, None, draw

    best_name, best_score = None, -1.0

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_crop = face.crop_bgr
        if face_crop is None or face_crop.size == 0:
            face_crop = bgr_img[y1:y2, x1:x2]
        if face_crop is None or face_crop.size == 0:
            continue

        try:
            resized = cv2.resize(face_crop, (160, 160))
            emb = embedder.embeddings([resized])[0]
            emb = normalize(emb.reshape(1, -1))[0]
        except Exception:
            continue

        sims = cosine_similarity(emb.reshape(1, -1), known_embeddings)[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])

        label = f"{class_names[idx]} ({score:.2f})"
        cv2.putText(draw, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if score > best_score:
            best_score = score
            best_name  = class_names[idx]

    if best_score is not None and best_score < threshold:
        return None, best_score, draw
    return best_name, best_score, draw


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")
THRESHOLD = st.sidebar.slider("Similarity threshold", 0.10, 0.95, 0.50, 0.01)
st.sidebar.info("Tip: 0.45–0.60 usually works for FaceNet cosine similarity.")

# Load embeddings
with st.spinner("Loading training images and building embeddings..."):
    classNames, known_embs = build_known_embeddings(TRAIN_DIR)

if len(classNames) == 0:
    st.warning(f"No valid faces loaded from `{TRAIN_DIR}`. "
               f"Add one clear image per person (filename = person name).")

st.caption(f"Loaded identities: {', '.join(classNames) if classNames else 'none'}")

# Keep an in-memory set of already marked names this session, seeded from CSV.
attendance_df = load_attendance(CSV_PATH)
marked_names = set(attendance_df["Name"].tolist())


# -----------------------------
# Tabs: Live camera / Upload / Attendance
# -----------------------------
tab_live, tab_upload, tab_log = st.tabs([" Live Camera", "Upload Photo", "Attendance Log"])

# --- Live Camera (using Streamlit's camera_input for snapshots) ---
with tab_live:
    st.subheader("Live Capture (snapshot mode)")
    st.write("Click *Take Photo* to capture. For true video streaming, use streamlit-webrtc (optional).")

    cam = st.camera_input("Camera", key="cam_input")
    if cam is not None:
        file_bytes = np.asarray(bytearray(cam.getvalue()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        name, score, annotated = detect_and_match(
            bgr, THRESHOLD, classNames, known_embs, load_models()[1], load_models()[0]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)

        if name:
            if mark_attendance(name, bgr, SHOT_DIR, CSV_PATH, marked_names):
                st.success(f"Attendance marked for **{name}** (score {score:.2f})")
            else:
                st.info(f"Already marked for **{name}** (score {score:.2f})")
        else:
            if score is None:
                st.warning("No face or no known embeddings.")
            else:
                st.warning(f"No match above threshold (best score {score:.2f} < {THRESHOLD:.2f}).")


# --- Upload Photo ---
with tab_upload:
    st.subheader("Upload a Photo")
    up = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if up is not None:
        file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        name, score, annotated = detect_and_match(
            bgr, THRESHOLD, classNames, known_embs, load_models()[1], load_models()[0]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)

        if name:
            if mark_attendance(name, bgr, SHOT_DIR, CSV_PATH, marked_names):
                st.success(f"Attendance marked for **{name}** (score {score:.2f})")
            else:
                st.info(f" Already marked for **{name}** (score {score:.2f})")
        else:
            if score is None:
                st.warning("No face or no known embeddings.")
            else:
                st.warning(f"No match above threshold (best score {score:.2f} < {THRESHOLD:.2f}).")


# --- Attendance Log ---
with tab_log:
    st.subheader("Attendance Log")
    attendance_df = load_attendance(CSV_PATH)
    st.dataframe(attendance_df, use_container_width=True, height=350)
    if not attendance_df.empty:
        csv_bytes = attendance_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Attendance CSV", data=csv_bytes,
                           file_name="Attendance.csv", mime="text/csv")

st.caption("💡 Add or update images in `Training_images/` and press 'Rerun' to refresh embeddings (top-right ↻).")
