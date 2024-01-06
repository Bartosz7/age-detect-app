import os
import sys
import time
import logging
from functools import reduce

import cv2
import mediapipe as mp
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap, QIcon
from PyQt6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget, QFileDialog,
                               QGraphicsView, QGraphicsScene, QSplitter, QSpacerItem)

from config import Config
from face_detection import DetectedFace, Detectors


MEAN_VALUE = 37.4
STD_VALUE = 14.5
MEAN_COLORS = [0.5646, 0.4326, 0.3711]
STD_COLORS = [0.2495, 0.2205, 0.2173]


def detect_face_with_open_cv(full_image):
    min_neighbors = 4
    clasifier = "haarcascade_frontalface_default.xml"

    gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + clasifier)
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(40, 40)
    )
    faces = []
    for x, y, w, h in face:
        face = full_image[y : y + h, x : x + w]
        faces.append(DetectedFace(face, x, y, w, h))
    return faces


def detect_faces_with_mediapipe(image):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return []

        faces = []

        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )
            face = image[y : y + h, x : x + w]
            detected_face = DetectedFace(face, x, y, w, h)
            faces.append(detected_face)
    return faces


def predict_age_resnet50(model_path, image):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging.debug(device)
    # model_path = os.path.join("data", "checkpoints", model_path)
    # model = resnet50(model_path).to(device)
    # model.eval()
    # print(reduce(lambda x, y: x * y, image.shape))
    if reduce(lambda x, y: x * y, image.shape) == 0:
        return 0
    
    image = Image.fromarray(image)  # Convert the NumPy array to PIL Image
    
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
    ])

    # image = get_face(image_path)
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    age = output.item() * STD_VALUE + MEAN_VALUE # denormalization
    return int(age)

def inference(run_path: str, image_path: str, model_id: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(run_path, "best_balancing_both.pth")
    if model_id is not None:
        model_path = os.path.join(run_path, "models", f"epoch={model_id}.pth")
    model = resnet50(model_path).to(device)
    model.eval()

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_COLORS, std=STD_COLORS),
    ])

    # image = get_face(image_path)
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image = Image.open(image_path)
    image = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    age = output.item() * STD_VALUE + MEAN_VALUE # denormalization
    return int(age)

def resnet50(weights_path: str | None = None, drop: float = 0.0):
    model = torchvision.models.resnet50(pretrained=weights_path is None)
    last_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(last_in_features, 128),
        nn.ReLU(),
        nn.Dropout(drop),
        nn.Linear(128, 1),
    )

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    
    return model

# TODO: remove this after incorporating
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("data", "checkpoints", "best_balancing_both.pth")
model = resnet50(model_path).to(device)
model.eval()


class ProcessingThread(QThread):
    """
    This is a thread that starts a VideoCapture and updates MainWindow
    """
    updateFrame = pyqtSignal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.fd_model_file = None
        self.ad_model_file = None
        self.status = True
        self.camera = True
        self.age_model = None
        self.selected_files = []
        self.age_info = []

    def set_fd_file(self, fname):
        # The data comes with the 'opencv-python' module
        # TODO: change to account for internal models and/or MediaPipe
        logging.debug(f"Face detection model set: {fname}")
        if fname.startswith("HaarCascade"):
            self.fd_detector = Detectors.OPEN_CV
            self.fd_model_file = os.path.join(cv2.data.haarcascades, Config.FACE_DETECTION_MODELS.get(fname))
        elif fname == "MediaPipe":
            self.fd_detector = Detectors.MEDIAPIPE
            self.fd_model_file = None
        else:
            self.fd_model_file = None

    def set_ad_file(self, fname):
        logging.debug(f"Age detection model set: {fname}")
        self.ad_model_file = fname

    def run(self):
        self.camera = cv2.VideoCapture(0) # web camera input
        while self.status:
            
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            detected_faces = []
            if self.fd_detector == Detectors.OPEN_CV:
                detected_faces = detect_face_with_open_cv(frame)
            elif self.fd_detector == Detectors.MEDIAPIPE:
                detected_faces = detect_faces_with_mediapipe(frame)
            
            # Adjusting image on the live video
            for face in detected_faces:
                self.draw_face_bounding_box(frame, face)
                age = self.predict_age(face)
                age_text = f"Age: {age}"
                self.draw_age_annotation(frame, face, text=age_text)

            # Postprocessing the image
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # back to RGB
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            scaled_img = img.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            self.updateFrame.emit(scaled_img)

    def predict_age(self, face):
        if self.ad_model_file == "Sweet18 Model":
            return 18
        elif self.ad_model_file == "ResNet-based Model":
            return predict_age_resnet50("best.pth", face.image)

    def detect_faces_with_haarcascades(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(self.fd_model_file)
        detections = cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(40, 40))
        return detections
    
    def load_age_model(self, model_path):
        self.age_model = torch.load(model_path)
        self.age_model.eval()

    def draw_face_bounding_box(self, frame, face):
        x, y, w, h = face.x, face.y, face.w, face.h
        pos_ori = (x, y)
        pos_end = (x + w, y + h)
        color = (0, 255, 0)
        cv2.rectangle(frame, pos_ori, pos_end, color, 2)

    def draw_age_annotation(self, frame, face, text):
        cv2.putText(
            frame,
            text,
            (face.x, face.y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(36, 255, 12),
            thickness=2
        )
