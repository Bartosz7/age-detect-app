from src.face_detection.face_detection import Detectors, detect_faces

if __name__ == "__main__":
    # face detection
    detect_faces(
        detector=Detectors.MEDIAPIPE,
        inputFolderPath="data/train_data",
        outputFolderPath="data/detections",
    )
    # age detection
    pass
