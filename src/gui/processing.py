import os
import logging

import cv2
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from config import Config
from face_detection import DetectedFace, Detectors
from face_detection import (detect_face_with_open_cv,
                            detect_faces_with_mediapipe)
from age_detection import resnet50, predict_age_resnet50


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.age_model = None
        self.load_age_model("")
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
        self.camera = cv2.VideoCapture(0)  # web camera input
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
            return predict_age_resnet50(self.age_model, self.device, face.image)

    def load_age_model(self, model_path):
        model_path = os.path.join("data", "checkpoints", "best_balancing_both.pth")
        self.age_model = resnet50(model_path).to(self.device)
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
