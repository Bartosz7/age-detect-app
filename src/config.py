import os
import cv2
from face_detection import FaceDetectors


class Config:

    # General
    VERSION = 0.1
    TITLE = "age-detect-app"

    # Set app general style (OS-dependent!)
    # Choices: [Windows, WindowsVista, Macintosh, Breeze, WindowsXP, Fusion, GTK]
    APP_STYLE = "Fusion"

    # Models
    CHECKPOINTS_DIR = "data/checkpoints"

    # Frame settings
    FRAME_MINSIZE = (40, 40)

    # Models (this may be a path to a file or un/loaded model handler)
    # Note: they will appear in the GUI in the same order (first one is default)
    FACE_DETECTION_MODELS = {
        # GUI/text name : FaceDetectors.Enum
        "MediaPipe": FaceDetectors.MEDIAPIPE,
        "HaarCascade Frontal Face default": FaceDetectors.HAAR_CASCADE
    }
    AGE_DETECTION_MODELS = {
        # GUI/text name : AgeDetector.Enum
        "Sweet18 Model": "model18",
        "ResNet-based Model": "resnetmodel"
    }

    # Directory for GUI static elements (must be absolute path from project root)
    STATIC_DIR_PATH = "src/gui/static"
    WELCOME_IMAGE = ""  # must be located in STATIC_DIR_PATH
