# src/utils.py

import cv2
import numpy as np
from PIL import Image

# Convert BGR (OpenCV) to RGB for Streamlit display
def convert_cv2_to_pil(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# Convert PIL to OpenCV format (if needed)
def convert_pil_to_cv2(pil_image):
    image = np.array(pil_image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Resize image with aspect ratio preserved
def resize_with_aspect_ratio(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Optional: Draw label on image (for future extension)
def draw_label(image, text, position, color=(0, 255, 0)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

