import os
import cv2


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
        "HaarCascade Frontal Face (default)": os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
        "HaarCascade Frontal Face (alt)": os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml"),
        "HaarCascade Frontal Face (alt2)": os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml"),
        "HaarCascade Profile Face": os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml"),
        "MediaPipe": "mediapipe",
    }
    AGE_DETECTION_MODELS = {
        "Sweet18 Model": "model18",
        "ResNet-based Model": "resnetmodel"
    }

    # Directory for GUI static elements (must be absolute path from project root)
    STATIC_DIR_PATH = "src/gui/static"