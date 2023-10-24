import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
from typing import List, Mapping, Optional, Tuple, Union


# For static images:
# IMAGE_FILES = ["face_detection/face_images/20230527_224925.jpg"]

IMAGE_FILES = next(os.walk("face_detection/face_images/", (None, None, [])))[2]

# IMAGE_FILES = ["face_detection/face_images/20230528_032416.jpg"]

with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
) as face_detection:
    directory = "face_detection/media_pipe/"
    os.makedirs(directory + "detected/", exist_ok=True)
    os.makedirs(directory + "not_detected/", exist_ok=True)

    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread("face_detection/face_images/" + file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            print(directory + "not_detected/" + file)
            cv2.imwrite(directory + "not_detected/" + file, image)
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
            "./face_detection/media_pipe/detected/" + file,
            annotated_image,
        )
