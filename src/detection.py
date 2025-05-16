import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from urllib.request import urlretrieve

# =======================
# Paths and Initialization
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize models
yolo_model = None
face_detection = None
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# =======================
# Init Functions
# =======================
def init_object_detector():
    global yolo_model

    if not os.path.isfile(MODEL_PATH):
        print("Downloading YOLOv8n weights...")
        urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            MODEL_PATH,
        )
        print("Download complete.")

    yolo_model = YOLO(MODEL_PATH)


def init_face_detector():
    global face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# =======================
# Detection Functions
# =======================
def detect_objects_from_frame(frame):
    if yolo_model is None:
        raise ValueError("YOLO model not initialized.")

    results = yolo_model(frame)[0]

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0].item()
        cls_id = int(result.cls[0])
        label = f"{yolo_model.names[cls_id]} ({conf:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def detect_faces_from_image(image_bgr):
    if face_detection is None:
        raise ValueError("Face detection not initialized.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image_bgr, detection)

    return image_bgr


def detect_faces_from_frame(frame):
    return detect_faces_from_image(frame)



