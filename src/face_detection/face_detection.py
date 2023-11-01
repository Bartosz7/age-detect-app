from enum import Enum
import os

from mediapipe_detection import (
    detect_faces_from_path_mediapipe,
    detect_faces_from_single_image,
)


class Detectors(Enum):
    OPEN_CV = 1
    MEDIAPIPE = 2


def detect_faces(
    output_folder_path,
    input_folder_path=None,
    detector=Detectors.MEDIAPIPE,
    image=None,
):
    if detector == Detectors.MEDIAPIPE:
        # if images from folder
        if input_folder_path != None:
            detect_faces_from_path_mediapipe(
                input_folder_path,
                output_folder_path,
            )
        # if images from camera TODO
        if image != None:
            detect_faces_from_single_image(image, output_folder_path)
        pass

    if detector == Detectors.OPEN_CV:
        # TODO if we want to use opencv
        pass
