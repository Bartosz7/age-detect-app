import os
import logging

import cv2
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from config import Config
from face_detection import DetectedFace, FaceDetectors, FaceDetectorFactory
from age_detection import resnet50, predict_age_resnet50


class ProcessingThread(QThread):
    """
    This is a thread that starts a VideoCapture and updates MainWindow
    """
    updateFrame = pyqtSignal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True  # live camera status
        self.camera = True
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.ad_model_file = None
        self.age_model = None
        self.load_age_model("")

    def run(self):
        self.camera = cv2.VideoCapture(0)  # web camera input
        while self.status:

            # Get current frame
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Detect faces
            detected_faces = self.fd_detector.detect_faces(frame)

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

    def set_fd_model(self, face_detector_name: str):
        face_detector = Config.FACE_DETECTION_MODELS.get(face_detector_name)
        face_detector_factory = FaceDetectorFactory()
        self.fd_detector = face_detector_factory.create_model(face_detector)

    def set_ad_file(self, fname):
        logging.debug(f"Age detection model set: {fname}")
        self.ad_model_file = fname

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
