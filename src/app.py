import streamlit as st
import cv2
from PIL import Image
import numpy as np
from detection import (
    init_object_detector,
    init_face_detector,
    detect_faces_from_frame,
    detect_objects_from_frame
)

# Initialize models
init_object_detector()
init_face_detector()

# Streamlit UI
st.title("🎯 Real-Time Face and Object Detection with MediaPipe")

st.sidebar.header("🧠 Detection Settings")
detect_faces = st.sidebar.checkbox("Enable Face Detection", value=True)
detect_objects = st.sidebar.checkbox("Enable Object Detection", value=True)

source_type = st.sidebar.radio("Choose Input Source:", ["Webcam", "Upload Image"])

# =============== Webcam Mode ===============
if source_type == "Webcam":
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)

    while run and cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)

        if detect_faces:
            frame = detect_faces_from_frame(frame)

        if detect_objects:
            frame = detect_objects_from_frame(frame)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    if cap:
        cap.release()

# =============== Upload Mode ===============
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        if detect_faces:
            frame = detect_faces_from_frame(frame)

        if detect_objects:
            frame = detect_objects_from_frame(frame)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Processed Image")
