import os
import logging
import time

import cv2
import torch
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPainter, QColor

from config import Config
from face_detection import DetectedFace, FaceDetectors, FaceDetectorFactory
from age_detection import resnet50, predict_age_resnet50


class ImageProcessingThread(QThread):
    """
    This is a thread that processes images, saves them and updates MainWindow
    """
    imagesProcessed = pyqtSignal(str)  # path to dir with processed images
    progress = pyqtSignal(int)  # progress bar value

    def __init__(self, images_paths_list):
        super().__init__()
        self.images_paths_list = images_paths_list
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.age_model = None
        self.face_detector = None
        self.load_age_model("")

    def run(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("output", timestamp)
        os.makedirs(output_dir, exist_ok=True)

        total_images = len(self.images_paths_list)
        for i, image_path in enumerate(self.images_paths_list):
            image = cv2.imread(image_path)
            detected_faces = self.fd_detector.detect_faces(image)

            for face in detected_faces:
                age = self.predict_age(face)
                age_text = f"Age: {age}"
                self.draw_age_annotation(image, face, text=age_text)

            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, image)

            progress = int((i + 1) / total_images * 100)
            self.progress.emit(progress)

        self.imagesProcessed.emit(output_dir)  # Emit the signal with the saved_images_dir_path

    def set_fd_model(self, face_detector_name: str):
        face_detector = Config.FACE_DETECTION_MODELS.get(face_detector_name)
        face_detector_factory = FaceDetectorFactory()
        self.fd_detector = face_detector_factory.create_model(face_detector)

    def set_images_paths_list(self, images_paths_list):
        self.images_paths_list = images_paths_list

    def draw_age_annotation(self, image, face, text):
        cv2.rectangle(image, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
        cv2.putText(image, text, (face.x, face.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
