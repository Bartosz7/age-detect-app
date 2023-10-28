import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from typing import List, Mapping, Optional, Tuple, Union


# For static images:
# IMAGE_FILES = ["face_detection/face_images/20230527_224925.jpg"]

face_images_dir = "../data/test_data/face_images/"
print(os.listdir(face_images_dir))
IMAGE_FILES = next(os.walk(face_images_dir, (None, None, [])))[2]

# IMAGE_FILES = ["face_detection/face_images/20230528_032416.jpg"]

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
) as face_detection:
    directory = "../data/detections/media_pipe2/"
    os.makedirs(directory + "detected/", exist_ok=True)
    os.makedirs(directory + "not_detected/", exist_ok=True)

    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(os.path.join(face_images_dir, file))
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            not_detected_path = os.path.join(directory, "not_detected", file)
            print(not_detected_path)
            cv2.imwrite(not_detected_path, image)
            continue
        annotated_image = image.copy()
        for detection in results.detections:
            print("Nose tip:")
            print(
                mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
                )
            )
            mp_drawing.draw_detection(
                annotated_image,
                detection,
                bbox_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                ),
            )

        cv2.imwrite(
            os.path.join(directory, "detected", file),
            annotated_image,
        )
