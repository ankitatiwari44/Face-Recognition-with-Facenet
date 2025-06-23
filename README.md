<<<<<<< HEAD
# Face Recognition with Attendance System

=======

# Face Recognition with Attendance System

>>>>>>> c67dea2 (changes done)
##The Face Recognition Attendance System uses a webcam to detect and recognize faces in real time, marking attendance automatically by comparing faces with stored images. It is built using Python, OpenCV, and the face_recognition library, making attendance fast, secure, and contactless.

cv2: OpenCV for image processing and webcam access.

numpy: Numerical operations.

face_recognition: Python library to detect and recognize faces.

os: To access file directories.

datetime: To get the current time/date for marking attendance.

## Steps performed in the Face Recognition Attendance System project, step by step:

1. Load Training Images

Reads images from a folder and extracts names from filenames.

2. Encode Faces

Converts each image into a 128-dimensional face encoding.

3. Capture Real-Time Video

Uses webcam to capture live video frames.

4. Detect and Recognize Faces

Detects faces in each frame and compares them to known encodings.

5. Draw Rectangle and Name

Highlights recognized faces with boxes and labels in the video feed.

<<<<<<< HEAD
6. Mark Attendance

Logs name and timestamp in a CSV file if not already recorded.

7. Exit Mechanism

Pressing 'q' cleanly stops the webcam and closes the window.
=======
6. Mark Attendance
>>>>>>> c67dea2 (changes done)
