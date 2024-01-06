import cv2
from enum import Enum
import os
from mediapipe.python import *
import mediapipe as mp


class Detectors(Enum):
    OPEN_CV = 1
    MEDIAPIPE = 2


class DetectedFace:
    def __init__(self, image, x, y, w, h) -> None:
        self.image = image
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def run_detection(
    detector=Detectors.MEDIAPIPE,
    input_path="",
    output_path="",
    full_image=None,
) -> list:  # list of faces and coords
    """
    detector is either MEDIAPIPE or OPEN_CV
    input_path is path to folder with images or path to single image with extension of .jpg or .png
    output_path is path to folder where cropped faces will be saved, if null then faces won't be saved
    full_image is image in cv2 format, if null then input_path must be provided
    function returns list of DetectedFace objects which contain image of face and coordinates of face in full_image
    """
    images = []

    if input_path == "" and full_image == None:
        return []

    # add images to list
    if full_image != None:
        images.append(full_image)
    elif input_path != "":
        if is_path_file(input_path):
            images.append(get_cv2_image_from_file(input_path))
        else:
            images = get_cv2_images_from_folder(input_path)

    list_of_faces_and_coordinates = []
    # get faces and coordinates
    if detector == Detectors.MEDIAPIPE:
        for image in images:
            list_of_faces_and_coordinates += detect_faces_with_mediapipe(image)
    else:
        for image in images:
            list_of_faces_and_coordinates += detect_face_with_open_cv(image)

    if output_path != "":
        absolute_path = os.path.abspath(output_path)
        if not os.path.exists(absolute_path):
            # os.makedirs(absolute_path)
            os.makedirs(absolute_path + "/faces/", exist_ok=True)

        for i, face in enumerate(list_of_faces_and_coordinates):
            path = os.path.join(
                absolute_path, "faces", f"face_{i}_{face.x}_{face.y}.png"
            )
            cv2.imwrite(path, face.image)

    return list_of_faces_and_coordinates


def is_path_file(path):
    if not path.endswith(".jpg") and not path.endswith(".png"):
        return False
    return True


def get_cv2_image_from_file(full_path: str):
    if not is_path_file(full_path):
        return None
    return cv2.imread(full_path)


def get_cv2_images_from_folder(folder_path) -> list:
    images = []
    IMAGE_FILES = next(os.walk(folder_path, (None, None, [])))[2]
    for _, file in enumerate(IMAGE_FILES):
        probable_image = get_cv2_image_from_file(os.path.join(folder_path, file))
        if probable_image is not None:
            images.append(probable_image)
    return images


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


def detect_face_with_open_cv(full_image):
    min_neighbors = 4
    clasifier = "haarcascade_frontalface_alt_tree.xml"

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
