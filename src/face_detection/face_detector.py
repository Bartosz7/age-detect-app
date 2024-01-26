import os
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from enum import Enum


class FaceDetectors(Enum):
    HAAR_CASCADE = 0
    MEDIAPIPE = 1


class DetectedFace:

    def __init__(self, image, x, y, w, h) -> None:
        self.image = image
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class FaceDetector:

    def __init__(self) -> None:
        pass

    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        # to be overridden by subclasses
        pass

    def detect_faces_from_path(self, input_path: str,
                               output_path: str | None) -> List[DetectedFace]:
        """
        Loads data from file or directory specified in `input_path` and returns
        the list of detected faces. The detected faces are saved in `output_path` if specified.
        """
        if input_path == "":
            return []

        # Loading images
        images = []
        if self.is_path_file(input_path):
            img = self.get_cv2_image_from_file(input_path)
            images = [img] if img is not None else []
        else:
            images = self.get_cv2_images_from_dir(input_path)

        result = []
        for image in images:
            result += self.detect_faces(image)

        # (optional) Saving results
        if output_path is not None:
            absolute_path = os.path.abspath(output_path)
            if not os.path.exists(absolute_path):
                final_path = os.path.join(absolute_path, "faces")
                os.makedirs(final_path, exist_ok=True)

            for i, face in enumerate(result):
                path = os.path.join(
                    absolute_path, f"face_{i}_{face.x}_{face.y}.png"
                )
                cv2.imwrite(path, face.image)

    def get_cv2_images_from_dir(self, dirpath) -> list:
        images = []
        IMAGE_FILES = next(os.walk(dirpath, (None, None, [])))[2]
        for _, file in enumerate(IMAGE_FILES):
            probable_image = self.get_cv2_image_from_file(os.path.join(dirpath, file))
            if probable_image is not None:
                images.append(probable_image)
        return images

    def get_cv2_image_from_file(self, filepath: str):
        if not self.is_path_file(filepath):
            return None
        return cv2.imread(filepath)

    def is_path_file(self, path):
        if not path.endswith(".jpg") and not path.endswith(".png"):
            return False
        return True


class HaarCascadeFaceDetector(FaceDetector):

    def __init__(self) -> None:
        super().__init__()

    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        min_neighbors = 4
        classifier_filename = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(classifier_filename)
        faces = face_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(40, 40)
        )
        faces_list = []
        for (x, y, w, h) in faces:
            face_crop = image[y:y+h, x:x+w]
            faces_list.append(DetectedFace(face_crop, x, y, w, h))
        return faces_list


class MediaPipeFaceDetector(FaceDetector):

    def __init__(self) -> None:
        super().__init__()

    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
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
                face = image[y:y + h, x:x + w]
                detected_face = DetectedFace(face, x, y, w, h)
                faces.append(detected_face)

        return faces


class FaceDetectorFactory:

    installed_face_detectors = {
        FaceDetectors.MEDIAPIPE: MediaPipeFaceDetector,
        FaceDetectors.HAAR_CASCADE: HaarCascadeFaceDetector
    }

    @classmethod
    def create_model(cls, model_type):
        print(model_type)
        face_detector = cls.installed_face_detectors.get(model_type)
        if face_detector:
            return face_detector()
        else:
            raise ValueError("Model not found")
