import base64
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = os.getenv("MODEL_PATH", "./model/best.pth")
CLASS_NAMES = ["downdog", "goddess", "plank", "tree", "warrior2"]

CONF_THRESHOLD = 0.6
PADDING = 0.15

DISPLAY_NAMES = {
    "downdog": "Downdog",
    "goddess": "Goddess",
    "plank": "Plank",
    "tree": "Tree",
    "warrior2": "Warrior",
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = models.resnet18()
model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, len(CLASS_NAMES)))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)
print("Model loaded.")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_body_bbox(landmarks, h, w, padding=0.15):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw = x_max - x_min
    bh = y_max - y_min
    x1 = max(0.0, x_min - bw * padding)
    y1 = max(0.0, y_min - bh * padding)
    x2 = min(1.0, x_max + bw * padding)
    y2 = min(1.0, y_max + bh * padding)
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def classify_crop(crop_bgr):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx = probs.argmax()
    return CLASS_NAMES[idx], float(probs[idx])


def process_frame(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    label, conf = None, 0.0

    if not results.pose_landmarks:
        cv2.putText(
            frame,
            "No person detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame, label, conf

    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
    )

    x1, y1, x2, y2 = get_body_bbox(results.pose_landmarks, h, w, PADDING)
    crop = frame[y1:y2, x1:x2]

    if crop.size <= 0:
        return frame, label, conf

    label, conf = classify_crop(crop)
    color = (42, 212, 115) if conf >= CONF_THRESHOLD else (0, 178, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{DISPLAY_NAMES.get(label, label)}  {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 6, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    return frame, label, conf


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10e6)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("frame")
def handle_frame(data):
    img_bytes = base64.b64decode(data["image"])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    result_frame, label, conf = process_frame(frame)
    _, buf = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64_result = base64.b64encode(buf).decode("utf-8")

    emit(
        "result_frame",
        {
            "image": b64_result,
            "label": label,
            "display_name": DISPLAY_NAMES.get(label, label),
            "conf": conf,
            "server_time": time.time(),
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running at http://0.0.0.0:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)