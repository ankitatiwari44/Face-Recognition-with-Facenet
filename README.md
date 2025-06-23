
# Face Recognition with Attendance System using FaceNet and InsightFace


##The Face Recognition Attendance System uses a webcam to detect and recognize faces in real time, marking attendance automatically by comparing faces with stored images. It is built using Python, OpenCV,InsightFace and the FaceNet library, making attendance fast, secure, and contactless.

cv2: OpenCV for image processing and webcam access.

numpy: Numerical operations.

FaceNet: to extract facial embeddings.

InsightFace: to detect faces in images.

os: To access file directories.

datetime: To get the current time/date for marking attendance.


## Model Used
1. Face Detection: InsightFace
insightface.app.FaceAnalysis(name="buffalo_l"):

High-accuracy model for detecting face bounding boxes.

Also returns aligned face crops used for embedding.

2. Face Embedding: Keras FaceNet
keras_facenet.FaceNet():

Pre-trained deep CNN that maps faces to 128D embeddings.

Embeddings are then compared using cosine similarity.

3. Similarity & Recognition
cosine_similarity from sklearn.metrics.pairwise checks similarity between new face and stored embeddings.

## Steps performed in the Face Recognition Attendance System project, step by step:

This is a Flask-based real-time face recognition attendance system. Key components:

1. Uses webcam or uploaded photo to recognize faces.

2. Detects faces using InsightFace, and extracts embeddings using FaceNet.

3. Matches detected embeddings against known ones using cosine similarity.

4. If a face is recognized (above threshold), it:
Saves a screenshot
Marks attendance in a CSV file

5. API endpoints allow:

Starting/stopping recognition
Uploading photos for recognition
Getting live video feed
Fetching attendance records

