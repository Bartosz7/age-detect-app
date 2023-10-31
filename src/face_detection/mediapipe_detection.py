import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def detect_faces_from_path_mediapipe(input_path, output_path):
    IMAGE_FILES = next(os.walk(input_path, (None, None, [])))[2]
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        os.makedirs(output_path + "/faces/", exist_ok=True)

        for _, file in enumerate(IMAGE_FILES):
            if not file.endswith(".jpg") and not file.endswith(".png"):
                continue

            image = cv2.imread(os.path.join(input_path, file))
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Piece of code if we would like to save images where faces weren't found
            if not results.detections:
                # piece of code if we would like to save images where faces haven't been detected

                # not_detected_path = os.path.join(output_path, "not_detected", file)
                # print(not_detected_path)
                # cv2.imwrite(not_detected_path, image)
                continue

            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # Crop the face from the original image
                face = image[y : y + h, x : x + w]
                path = os.path.join(output_path, "faces", f"face_{i}_{file}")
                # Save the cropped face
                cv2.imwrite(path, face)
