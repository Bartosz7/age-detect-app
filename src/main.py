from src.face_detection.face_detection import Detectors, detect_faces

if __name__ == "__main__":
    # face detection

    # detect_faces_from_folder
    detect_faces(
        detector=Detectors.MEDIAPIPE,
        inputFolderPath="data/train_data",
        outputFolderPath="data/detections",
    )

    # detect_faces_from_camera

    # detect_faces(
    #     outputFolderPath="data/camera",
    #     detector=Detectors.MEDIAPIPE,
    #     image=None,  # change to image from camera
    # )

    # age detection
    pass
