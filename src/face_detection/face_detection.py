from enum import Enum
import os

from mediapipe_detection import detect_faces_from_path_mediapipe


class Detectors(Enum):
    OPEN_CV = 1
    MEDIAPIPE = 2


def detect_faces(
    inputFolderPath=None,
    outputFolderPath=None,
    detector=Detectors.MEDIAPIPE,
):
    if detector == Detectors.MEDIAPIPE:
        # if images from folder
        if inputFolderPath != None:
            detect_faces_from_path_mediapipe(
                inputFolderPath,
                outputFolderPath,
            )
        # if images from camera TODO
        pass

    if detector == Detectors.OPEN_CV:
        # TODO if we want to use opencv
        pass


if __name__ == "__main__":
    print(os.getcwd())
    detect_faces(
        detector=Detectors.MEDIAPIPE,
        inputFolderPath="data/train_data",
        outputFolderPath="data/detections",
    )
